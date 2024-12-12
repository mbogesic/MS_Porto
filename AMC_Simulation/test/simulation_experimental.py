import os
import pandas as pd
import networkx as nx
from mesa import Model
from mesa.space import NetworkGrid
from mesa.time import RandomActivation
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.exception import NetworkXNoPath
from scipy.spatial import KDTree
import numpy as np

def order_route_graph(graph, start_node, end_node, node_positions=None):
    """
    Orders the nodes and edges in an nx.MultiDiGraph between start and end nodes.
    Handles cases where no path exists by connecting to the nearest node.

    Parameters:
    - graph: nx.MultiDiGraph - The graph to process.
    - start_node: Node - The starting node ID.
    - end_node: Node - The ending node ID.
    - node_positions: dict (optional) - Dictionary of node positions {node_id: (x, y)}.
      Required if nearest node handling is needed.

    Returns:
    - ordered_nodes: List of nodes in order from start to end.
    - ordered_edges: List of edges in order from start to end.
    """
    try:
        # Find the shortest path from start to end
        ordered_nodes = shortest_path(graph, source=start_node, target=end_node)
    except NetworkXNoPath:
        # Handle the case where no valid path exists
        if not node_positions:
            raise ValueError("No valid path and node_positions are required for nearest node handling.")

        # Use KDTree to find the nearest connected node to the start or end
        unconnected_nodes = {start_node, end_node}
        connected_nodes = list(graph.nodes)

        # Build KDTree for connected nodes
        connected_positions = [node_positions[node] for node in connected_nodes]
        tree = KDTree(connected_positions)

        # Find nearest connected nodes for each unconnected node
        for unconnected_node in unconnected_nodes:
            if unconnected_node not in connected_nodes:
                nearest_idx = tree.query(node_positions[unconnected_node])[1]
                nearest_node = connected_nodes[nearest_idx]
                graph.add_edge(unconnected_node, nearest_node)  # Add edge to nearest node

        # Retry finding the shortest path
        ordered_nodes = shortest_path(graph, source=start_node, target=end_node)

    # Generate ordered edges based on the ordered nodes
    ordered_edges = list(zip(ordered_nodes[:-1], ordered_nodes[1:]))

    return ordered_nodes, ordered_edges


def extract_route_lengths(nodes_and_edges_folder):
    route_lengths = {}
    edges_files = [f for f in os.listdir(nodes_and_edges_folder) if f.endswith("_edges.csv")]

    for edges_file in edges_files:
        edges_df = pd.read_csv(os.path.join(nodes_and_edges_folder, edges_file))

        # Calculate total length based on traversal order
        total_length = edges_df["length"].sum()
        route_key = edges_file.replace("_edges.csv", "")
        route_lengths[route_key] = total_length

    return route_lengths

class CongestionNetworkGrid(NetworkGrid):
    def __init__(self, graph):
        super().__init__(graph)
        self.edge_congestion = {}  # Track agents on edges

    def place_agent_on_edge(self, agent, u, v):
        """
        Place an agent on an edge (u, v) and track congestion.
        """
        edge_key = (u, v)
        if edge_key not in self.edge_congestion:
            self.edge_congestion[edge_key] = []
        self.edge_congestion[edge_key].append(agent)

    def remove_agent_from_edge(self, agent, u, v):
        """
        Remove an agent from an edge (u, v) and update congestion.
        """
        edge_key = (u, v)
        if edge_key in self.edge_congestion and agent in self.edge_congestion[edge_key]:
            self.edge_congestion[edge_key].remove(agent)
            if not self.edge_congestion[edge_key]:
                del self.edge_congestion[edge_key]

    def get_edge_congestion(self, u, v):
        """
        Get the number of agents currently on edge (u, v).
        """
        edge_key = (u, v)
        return len(self.edge_congestion.get(edge_key, []))

class TrafficModel(Model):
    def __init__(self, nodes_and_edges_folder, num_agents, agent_speed=10, step_time=10, combined_nodes_file=None, combined_edges_file=None):
        """
        Initialize the traffic model.

        Parameters:
            nodes_and_edges_folder: Folder containing route CSVs
            num_agents: Number of agents in the simulation
            agent_speed: Speed of the agents (meters/second)
            step_time: Time per step (seconds)
            combined_nodes_file: Path to the combined subgraph nodes CSV file (optional)
            combined_edges_file: Path to the combined subgraph edges CSV file (optional)
        """
        super().__init__()
        
        # Load the combined subgraph
        # self.load_combined_subgraph(combined_nodes_file, combined_edges_file)

        self.nodes_and_edges_folder = nodes_and_edges_folder
        self.num_agents = num_agents
        self.agent_speed = agent_speed
        self.routes = []  # List of routes (subgraphs)
        self.route_names = []  # List of route names
        self.route_lengths = extract_route_lengths(nodes_and_edges_folder)  # Extract lengths here
        self.graph = nx.MultiDiGraph()
        self.schedule = RandomActivation(self)
        self.grid = CongestionNetworkGrid(self.graph)
        self.completed_agents = 0
        self.simulation_finished = False
        self.step_time = step_time
        self.step_count = 0
        self.routes_visuals = []
        self.node_positions = []
        
        # Load all routes (subgraphs) from the folder
        self.load_routes()
        
        # After loading routes
        print(f"Main graph has {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges.")
        print(f"Routes loaded: {len(self.routes)}")
        print(f"First few nodes: {list(self.graph.nodes)[:5]}")
        # Print the extracted route lengths
        for route_name, length in self.route_lengths.items():
            print(f"Route: {route_name}, Total Length: {length:.2f} meters")
            
        for route_graph in self.routes:
            for node in route_graph.nodes:
                if node not in self.graph.nodes:
                    print(f"Node {node} from subgraph not in main graph!")

        # Add agents to the model
        self.add_agents()
    
    def agent_completed(self, agent_id):
        """
        Notify the model that an agent has completed its route.
        This ensures the model's completed_agents counter is updated accurately.
        """
        print(f"Agent {agent_id} has completed its journey.")
        self.completed_agents += 1

    def get_ordered_edges(self, graph, start_node, end_node):
        """
        Return edges of the graph sorted in path-like order from start_node to end_node.
        Falls back to unsorted edges if no path is found.
        """
        if nx.has_path(graph, start_node, end_node):
            path = nx.shortest_path(graph, source=start_node, target=end_node)
            ordered_edges = [
                (path[i], path[i + 1], graph.get_edge_data(path[i], path[i + 1])[0])
                for i in range(len(path) - 1)
            ]
        else:
            print(f"Warning: Falling back to unsorted edges for {graph}.")
            ordered_edges = [
                (u, v, data)
                for u, v, data in graph.edges(data=True)
            ]
        return ordered_edges

    def load_routes(self):
        """
        Load all routes from the specified folder into the model.
        """
        nodes_files = [f for f in os.listdir(self.nodes_and_edges_folder) 
               if f.endswith("_nodes.csv") and not f.startswith("all")]
        edges_files = [f for f in os.listdir(self.nodes_and_edges_folder) 
               if f.endswith("_edges.csv") and not f.startswith("all")]

        self.route_names = []  # Store route names for reference
        self.all_ordered_routes = []
        
        for nodes_file, edges_file in zip(sorted(nodes_files), sorted(edges_files)):
            nodes_df = pd.read_csv(os.path.join(self.nodes_and_edges_folder, nodes_file))
            edges_df = pd.read_csv(os.path.join(self.nodes_and_edges_folder, edges_file))

            route_graph = nx.MultiDiGraph()
            route_graph_visuals = nx.MultiDiGraph()
            
            for _, row in nodes_df.iterrows():
                route_graph.add_node(row["node"], **row.to_dict())

            for _, row in edges_df.iterrows():
                edge_attributes = row.to_dict()
                start_node = edge_attributes.pop("start_node")
                end_node = edge_attributes.pop("end_node")
                length = edge_attributes.pop("length", None)

                route_graph.add_edge(
                    start_node,
                    end_node,
                    length=length,
                    **edge_attributes
                )
            
            # Determine the route's start and end nodes
            if "Asprela" in nodes_file:
                start_node, end_node = 4523960189, 479183608
            elif "Campo_Alegre" in nodes_file:
                start_node, end_node = 479183608, 4523960189
            else:
                print(f"Unknown route type in {nodes_file}")
                continue

            # Get the ordered edges for the route
            ordered_edges = self.get_ordered_edges(route_graph, start_node, end_node)
            self.all_ordered_routes.append(ordered_edges)
            
            self.routes.append(route_graph)
            self.route_names.append(nodes_file.split("_nodes.csv")[0])  # Extract route name from file

            for node, data in route_graph.nodes(data=True):
                self.graph.add_node(node, **data)
            for u, v, key, data in route_graph.edges(keys=True, data=True):
                self.graph.add_edge(u, v, key=key, **data)
            
            # Add nodes with geographical positions
            for _, row in nodes_df.iterrows():
                route_graph_visuals.add_node(row["node"], x=row["x"], y=row["y"])

            # Add edges
            for _, row in edges_df.iterrows():
                edge_attributes = row.to_dict()
                start_node = edge_attributes.pop("start_node")
                end_node = edge_attributes.pop("end_node")
                route_graph_visuals.add_edge(start_node, end_node, **edge_attributes)
                        
            self.routes_visuals.append(route_graph_visuals)
            
            print(f"Outgoing edges from {start_node}:")
            for _, v, data in route_graph.out_edges(start_node, data=True):
                print(f"  {start_node} -> {v}, Length: {data.get('length', 0)}")
      
        # Precompute node positions for visualization
        for route_graph_visuals in self.routes_visuals:
            node_positions = {
                node: (data["x"], data["y"])
                for node, data in route_graph_visuals.nodes(data=True)
            }
            route_graph_visuals = order_route_graph(route_graph_visuals, start_node, end_node)
            self.node_positions.append(node_positions)

        # Compute bounds for visualization
        x_coords = []
        y_coords = []
        for positions in self.node_positions:  # Iterate over the list of position dictionaries
            x_coords.extend(pos[0] for pos in positions.values())
            y_coords.extend(pos[1] for pos in positions.values())

        self.min_x, self.max_x = min(x_coords), max(x_coords)
        self.min_y, self.max_y = min(y_coords), max(y_coords)
        
        for node in self.graph.nodes:
            if "agent" not in self.graph.nodes[node]:
                self.graph.nodes[node]["agent"] = []
                
            for route_graph in self.routes:
                for node in route_graph.nodes:
                    if node not in self.graph.nodes:
                        print(f"Node {node} from subgraph not in main graph!")

        # Normalize the node positions for alignment with image dimensions
        self.scaled_positions = {}
        for positions in self.node_positions:  # Iterate over all node positions
            for node, pos in positions.items():
                self.scaled_positions[node] = (
                    (pos[0] - self.min_x) / (self.max_x - self.min_x),  # Normalize x
                    (pos[1] - self.min_y) / (self.max_y - self.min_y)   # Normalize y
                )
                
        self.normalized_route_edges = []  # Add a container for normalized route edges
        
        # Normalize edges for visualization
        normalized_edges = []
        for route in self.all_ordered_routes:
            for u, v, data in route:
                normalized_edges.append({
                    "start_node": u,
                    "end_node": v,
                    "start_pos": self.scaled_positions[u],
                    "end_pos": self.scaled_positions[v],
                    "length": data.get("length", 0),
                })
            self.normalized_route_edges.append(normalized_edges)
                
        # for route_graph_visuals in self.routes_visuals:
        #     normalized_edges = []
        #     for u, v, data in route_graph_visuals.edges(data=True):
        #         if u in self.scaled_positions and v in self.scaled_positions:
        #             normalized_edges.append({
        #                 "start_node": u,
        #                 "end_node": v,
        #                 "start_pos": self.scaled_positions[u],
        #                 "end_pos": self.scaled_positions[v],
        #                 "length": data.get("length", 0),
        #             })
        #     self.normalized_route_edges.append(normalized_edges)
        
        print(f"Loaded {len(self.routes)} routes from {self.nodes_and_edges_folder}")

    def load_combined_subgraph(self, nodes_file, edges_file):
        """
        Load the combined subgraph directly from nodes and edges CSV files.
        """
        nodes_df = pd.read_csv(nodes_file)
        edges_df = pd.read_csv(edges_file)

        self.combined_subgraph = nx.MultiDiGraph()

        # Add nodes with geographical positions
        for _, row in nodes_df.iterrows():
            self.combined_subgraph.add_node(row["node"], x=row["x"], y=row["y"])

        # Add edges
        for _, row in edges_df.iterrows():
            edge_attributes = row.to_dict()
            start_node = edge_attributes.pop("start_node")
            end_node = edge_attributes.pop("end_node")
            self.combined_subgraph.add_edge(start_node, end_node, **edge_attributes)

        # Precompute node positions for visualization
        self.node_positions = {
            node: (data["x"], data["y"])
            for node, data in self.combined_subgraph.nodes(data=True)
        }

        print(f"Combined subgraph loaded with {len(self.combined_subgraph.nodes)} nodes and {len(self.combined_subgraph.edges)} edges.")

    def add_agents(self):
        """
        Add agents to the model, assigning fixed start and end nodes based on the route.
        """
        for i in range(self.num_agents):
            # Assign a random route to the agent
            route_index = self.random.randint(0, len(self.routes) - 1)
            route_graph = self.routes[route_index]
            route_name = self.route_names[route_index]  # Full route name, e.g., "Asprela_2_Campo_Alegre_route_1"

            # Determine the fixed start and end nodes
            if route_name.startswith("Asprela_"):
                start_node = 4523960189
                end_node = 479183608
                origin = "Asprela"
                destination = "Campo Alegre"
            elif route_name.startswith("Campo_Alegre_"):
                start_node = 479183608
                end_node = 4523960189
                origin = "Campo Alegre"
                destination = "Asprela"
            else:
                print(f"Route name {route_name} does not match expected prefixes. Skipping agent {i}.")
                continue

            # Debugging: Print assigned route and nodes
            print(f"Agent {i} - {origin} -> {destination}, Route: {route_name}")

            # Ensure the start and end nodes exist in the main graph
            if start_node not in self.graph.nodes or end_node not in self.graph.nodes:
                print(f"Skipping agent {i} due to invalid start or end node.")
                continue
        
            # Create and place the agent
            agent = TrafficAgent(
                self.next_id(),
                self,
                start_node=start_node,
                end_node=end_node,
                route_graph=route_graph,
                route_name=route_name,  # Pass the full route name
                normalized_route_edges=self.normalized_route_edges[route_index],  # Get normalized route edges
                speed=self.agent_speed,
                step_time=self.step_time,
            )
            self.schedule.add(agent)

            print(f"Agent {agent.unique_id} assigned route: {route_name}")
            for edge in agent.normalized_route_edges:
                print(f"  {edge['start_node']} -> {edge['end_node']}, Length: {edge['length']}")
        
            # Place the agent on the grid
            self.grid.place_agent(agent, start_node)

    def step(self):
        """
        Advance the simulation by one step.
        """
        print(f"--- Step {self.step_count} ---")
        self.step_count += 1
        if self.completed_agents >= self.num_agents:
            print("All agents have completed their journeys. Stopping simulation.")
            self.simulation_finished = True  # Mark the simulation as finished
            return  # Prevent further steps
        self.schedule.step()  # Let all agents take their actions

        # Count completed agents
        print(f"Total completed agents: {self.completed_agents}")

class TrafficAgent:
    def __init__(self, unique_id, model, start_node, end_node, route_graph, route_name, normalized_route_edges, speed=10, step_time=10):
        """
        Initialize the traffic agent.

        Parameters:
            unique_id: Unique identifier for the agent
            model: The simulation model
            start_node: The starting node for the agent
            end_node: The destination node for the agent
            route_graph: The subgraph representing the agent's route
            route_name: The unique name of the route the agent is on
            speed: Speed of the agent in meters/second
        """
        self.unique_id = unique_id
        self.model = model
        self.current_node = start_node
        self.pos = start_node  # Add the position attribute for MESA's NetworkGrid
        self.end_node = end_node
        self.route_graph = route_graph
        self.route_name = route_name  # Store the full route name
        self.speed = speed
        self.step_time = step_time # Step time in seconds (aka one simulation step equals this many seconds)
        self.step_cnt = 0  # Counter for the number of steps taken
        self.distance_travelled = 0.0  # Initialize distance travelled
        self.elapsed_time = 0.0
        self.completed = False
        self.counted = False
        self.route_length = self.model.route_lengths[self.route_name]
        self.route_edges = list(self.route_graph.edges(data=True))
        self.current_edge_index = 0
        self.edge_travelled = 0.0
        self.normalized_route_edges = normalized_route_edges
        
        # Determine origin and destination 
        if start_node == 4523960189:
            self.origin = "Asprela"
            self.destination = "Campo Alegre"
        elif start_node == 479183608:
            self.origin = "Campo Alegre"
            self.destination = "Asprela"
    
    def get_assigned_route_edges(self):
        """
        Returns the list of edges (with data) that define the agent's route.
        """
        return self.route_edges
        
    def get_remaining_nodes_count(self):
        """
        Get the number of nodes left to visit on the route based on current edge index.
        """
        # Remaining nodes are the nodes connected to edges from the current edge index onward
        remaining_edges = self.route_edges[self.current_edge_index:]
        remaining_nodes = set(edge[1] for edge in remaining_edges)  # Target nodes of remaining edges
        remaining_nodes.add(self.end_node)  # Ensure the end node is counted
        return len(remaining_nodes)
    
    def move(self):
        """
        Move the agent along its route based on the distance travelled.
        """
        # Increment step counter and calculate distance travelled
        self.step_cnt += 1
        distance_this_step = self.speed * self.step_time
        self.distance_travelled += distance_this_step
        self.elapsed_time += self.step_time

        # Check if the agent has completed the route
        if self.distance_travelled >= self.route_length:
            if not self.completed:
                self.distance_travelled = self.route_length  # Cap at total route length
                self.completed = True
                self.model.agent_completed(self.unique_id)
                print(f"Agent {self.unique_id} has completed its journey.")
            return

        # Ensure the agent moves along valid edges
        if self.current_edge_index < len(self.route_edges):
            current_edge = self.route_edges[self.current_edge_index]
            edge_length = current_edge[2]["length"]

            # Move along the edge and check if the edge is completed
            self.edge_travelled += distance_this_step
            while self.edge_travelled >= edge_length:
                # Move to the next edge
                self.edge_travelled -= edge_length
                self.current_edge_index += 1

                # Ensure the new edge belongs to the agent's route graph
                if self.current_edge_index < len(self.route_edges):
                    current_edge = self.route_edges[self.current_edge_index]
                    edge_length = current_edge[2]["length"]
                else:
                    break

        # Update the current node based on the current edge
        self.current_node = current_edge[1]  # Target node of the current edge
        self.pos = self.current_node  # Update position in the grid


        # Calculate progress percentage
        progress_percentage = round((self.distance_travelled / self.route_length) * 100, 2)
        remaining_nodes_count = self.get_remaining_nodes_count()
        
        # Debugging output
        print(
            f"Agent {self.unique_id} moving from {self.origin} to {self.destination}. Route: {self.route_name[-1]}. "
            f"Distance travelled: {self.distance_travelled:.2f} meters ({progress_percentage:.2f}% completed). "
            f"Elapsed time: {self.elapsed_time:.2f} seconds. Nodes left: {remaining_nodes_count-1}."
        )



    def step(self):
        """
        Execute one step for the agent.
        """
        self.move()


# Main execution
if __name__ == "__main__":
    # Define parameters
    nodes_and_edges_folder = "nodes_and_edges"
    combined_nodes_file = os.path.join(nodes_and_edges_folder, "all_routes_combined_nodes.csv")
    combined_edges_file = os.path.join(nodes_and_edges_folder, "all_routes_combined_edges.csv")
    num_agents = 1
    agent_speed = 3.9583333     # m/s (assuming an avg speed of 14.25km/h)
    step_time_dimension = 1.0   # s/step aka the "resolution" of the simulation

    # Initialize the model
    model = TrafficModel(
        nodes_and_edges_folder,
        num_agents, 
        agent_speed, 
        step_time_dimension, 
        combined_nodes_file=combined_nodes_file,
        combined_edges_file=combined_edges_file
    )

    # Run the simulation for a few steps
    for i in range(1):
        print(f"--- Step {i + 1} ---")
        model.step()
        
    # Run the simulation until every agent has finished his travel
    # step_count = 0
    # while not model.simulation_finished:
    #     print(f"--- Step {step_count} ---")
    #     model.step()
    #     step_count += 1