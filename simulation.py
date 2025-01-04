import os
import pandas as pd
import networkx as nx
from mesa import Model
from mesa.space import NetworkGrid
from mesa.time import RandomActivation
import Formulas as f
import random

# Automatically change the working directory to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
os.chdir(script_dir)  # Change the working directory

print(f"Working directory changed to: {os.getcwd()}")

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
    def __init__(self, nodes_and_edges_folder, num_agents, step_time=10, episodes=100, alpha=0.1, gamma=0.9, epsilon=0.1, combined_nodes_file=None, combined_edges_file=None):
        """
        Initialize the traffic model.

        Parameters:
            nodes_and_edges_folder: Folder containing route CSVs
            num_agents: Number of agents in the simulation
            step_time: Time per step (seconds)
            combined_nodes_file: Path to the combined subgraph nodes CSV file (optional)
            combined_edges_file: Path to the combined subgraph edges CSV file (optional)
            episodes: Number of episodes for the Q-learning loop
            alpha: Learning rate for Q-learning
            gamma: Discount factor for Q-learning
            epsilon: Exploration probability for Q-learning
        """
        super().__init__()
        
        # Load the combined subgraph
        # self.load_combined_subgraph(combined_nodes_file, combined_edges_file)
        self.schedule = RandomActivation(self)
        self.nodes_and_edges_folder = nodes_and_edges_folder
        self.num_agents = num_agents
        self.custom_agents = set()
        self.routes = []  # List of routes (subgraphs)
        self.route_names = []  # List of route names
        self.route_lengths = extract_route_lengths(nodes_and_edges_folder)  # Extract lengths here
        self.simulation_finished = False
        self.step_time = step_time
        self.step_count = 0
        self.routes_visuals = []
        self.node_positions = []
        
        self.episodes = episodes
        # Define CO2 emission factors
        self.co2_factors = {
            "Bike": 0,  # g/m - According to https://ourworldindata.org/travel-carbon-footprint
            "Car": 0.17,
            "PublicTransport": 0.097,
        }
        self.mode_distributions = []

        self.total_co2_emissions = 0  # Cumulative CO2 emissions across all episodes
        self.co2_emissions_over_time = []
        self.co2_emissions_per_episode = []  # CO2 emissions per episode
        self.current_episode_emissions = 0  # Running total for the current episode
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.current_episode = 0
        self.simulation_finished = False
        
        # Q-table: Nested dictionary for state-action pairs
        self.q_table = {}
        
        # Load routes and agents
        self.load_routes()
        self.validate_routes()
        self.add_agents()
        
        # Initialize environment and agents
        self.reset_environment()

    def reset_environment(self):
        """Reset the simulation environment while retaining Q-table."""
        
        self.completed_agents = 0
        self.simulation_finished = False
        self.step_count = 0
        self.current_episode_emissions = 0
        
        # Count mode distribution at the start of the episode
        start_distribution = {"Bike": 0, "PublicTransport": 0, "Car": 0}
        
        for agent in self.custom_agents:
            start_distribution[agent.last_action] += 1

        # Store the start distribution for analysis
        self.mode_distributions.append(start_distribution)
        print(f"+++ Episode {self.current_episode} Started +++")
        print(f"Mode Distribution: {start_distribution}")

        # Reset agents for the new episode
        for agent in self.custom_agents:
            # Compute Q-learning updates for agents
            current_state = agent.get_state()
            action = agent.last_action
            next_state = agent.get_state()
            reward = self.compute_reward(agent)

            # Update Q-values
            agent.update_q_value(current_state, action, reward, next_state)
            
            agent.reset_for_new_episode()
        
        # # After loading routes
        # print(f"Main graph has {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges.")
        # print(f"Routes loaded: {len(self.routes)}")
        # print(f"First few nodes: {list(self.graph.nodes)[:5]}")
        # # Print the extracted route lengths
        # for route_name, length in self.route_lengths.items():
        #     print(f"Route: {route_name}, Total Length: {length:.2f} meters")
            
        # for route_graph in self.routes:
        #     for node in route_graph.nodes:
        #         if node not in self.graph.nodes:
        #             print(f"Node {node} from subgraph not in main graph!")
        
    def agent_completed(self, agent_id):
        """
        Notify the model that an agent has completed its route.
        This ensures the model's completed_agents counter is updated accurately.
        """
        # print(f"Agent {agent_id} has completed its journey.")
        agent = next(a for a in self.schedule.agents if a.unique_id == agent_id)
        agent.completed = True
        # Set agent's last_action to a no-emission mode (e.g., "None")
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
        
        self.graph = nx.MultiDiGraph()
        self.grid = CongestionNetworkGrid(self.graph)
        
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
                
        print(f"Loaded {len(self.routes)} routes and {len(self.route_names)} route names.")
        
        # Precompute node positions for visualization
        for route_graph_visuals in self.routes_visuals:
            node_positions = {
                node: (data["x"], data["y"])
                for node, data in route_graph_visuals.nodes(data=True)
            }
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
        
        print(f"Loaded {len(self.routes)} routes from {self.nodes_and_edges_folder}")

    def validate_routes(self):
        """Validate the integrity of the loaded routes."""
        if len(self.routes) != len(self.route_names):
            raise ValueError(f"Mismatch between routes ({len(self.routes)}) and route names ({len(self.route_names)})!")


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

    # print(f"Combined subgraph loaded with {len(self.combined_subgraph.nodes)} nodes and {len(self.combined_subgraph.edges)} edges.")
    def get_episode_data(self):
        """Returns the latest data for dashboard updates."""
        return {
            "co2_emissions_over_time": self.co2_emissions_over_time,
            "co2_emissions_per_episode": self.co2_emissions_per_episode,
            "current_episode": self.current_episode,
            "total_co2_emissions": self.total_co2_emissions,
        }   
        
    def add_agents(self):
        """
        Add agents to the model, assigning fixed start and end nodes based on the route.
        """
        for i in range(self.num_agents):
            # Assign a random route to the agent
            route_index = self.random.randint(0, len(self.routes) - 1)
            if route_index >= len(self.route_names):
                raise IndexError(f"Generated route_index {route_index} exceeds route_names size {len(self.route_names)}")

            route_graph = self.routes[route_index]
            route_name = self.route_names[route_index]  # Full route name, e.g., "Asprela_2_Campo_Alegre_route_1"

            # Determine the fixed start and end nodes
            if route_name.startswith("Asprela_"):
                start_node = 4523960189
                end_node = 479183608
                # origin = "Asprela"
                # destination = "Campo Alegre"
            elif route_name.startswith("Campo_Alegre_"):
                start_node = 479183608
                end_node = 4523960189
                # origin = "Campo Alegre"
                # destination = "Asprela"
            else:
                print(f"Route name {route_name} does not match expected prefixes. Skipping agent {i}.")
                continue

            # Ensure the start and end nodes exist in the main graph
            if start_node not in self.graph.nodes or end_node not in self.graph.nodes:
                print(f"Skipping agent {i} due to invalid start or end node.")
                continue
            
            # Check speed of the agent according to chosen route
            if "Bike" in route_name:
                agent_speed = 3.0555556  # m/s (assuming an avg speed of 11km/h)
            elif "Car" or "PublicTransport" in route_name:
                agent_speed = 3.9583333 # m/s (assuming an avg speed of 14.25km/h)
            
            # Create and place the agent
            agent = TrafficAgent(
                self.next_id(),
                self,
                start_node=start_node,
                end_node=end_node,
                route_graph=route_graph,
                route_name=route_name,  # Pass the full route name
                normalized_route_edges=self.normalized_route_edges[route_index],  # Get normalized route edges
                speed=agent_speed,
                step_time=self.step_time,
            )
            self.schedule.add(agent)
            self.custom_agents.add(agent)
            # Place the agent on the grid
            self.grid.place_agent(agent, start_node)
            
    def compute_reward(self, agent):
        """
        Compute the reward for the agent based on its current mode of transport.
        """
        if agent.last_action == "Bike":
            return 1  # Highest reward for biking
        elif agent.last_action == "PublicTransport":
            return 0.5  # Medium reward for public transport
        elif agent.last_action == "Car":
            return -1  # Penalty for car use
        else:
            return 0  # Default reward for unknown modes

    def get_agent_credits(self): #CREDIT SCHEME
            """
            Retrieve the credits of all agents in the simulation.
            Returns:
                dict: A dictionary mapping agent IDs to their credits.
            """
            return {agent.unique_id: agent.credits for agent in self.schedule.agents}

    def update_route_based_on_mode_and_direction(self, agent, selected_mode):
        """
        Update the route and graph based on the selected transport mode and direction (CA -> A or A -> CA).
        
        Parameters:
            agent: The traffic agent to update.
            selected_mode: The selected mode of transport (e.g., "Bike", "Car", "PublicTransport").
        """
        # Define a map for modes and routes, considering both directions
        route_map = {
            ("Asprela", "Campo Alegre", "Bike"): "Asprela_2_Campo_Alegre_Bike",
            ("Asprela", "Campo Alegre", "PublicTransport"): "Asprela_2_Campo_Alegre_PublicTransport",
            ("Asprela", "Campo Alegre", "Car"): "Asprela_2_Campo_Alegre_Car",
            ("Campo Alegre", "Asprela", "Bike"): "Campo_Alegre_2_Asprela_Bike",
            ("Campo Alegre", "Asprela", "PublicTransport"): "Campo_Alegre_2_Asprela_PublicTransport",
            ("Campo Alegre", "Asprela", "Car"): "Campo_Alegre_2_Asprela_Car",
        }

        # Determine the route based on the agent's origin, destination, and mode
        route_key = (agent.origin, agent.destination, selected_mode)
        if route_key not in route_map:
            raise ValueError(f"Invalid route or mode: {route_key}")

        # Update the agent's route and graph
        agent.route_name = route_map[route_key]
        agent.route_graph = self.routes[self.route_names.index(agent.route_name)]
        
    def step(self):
        """
        Advance the simulation by one step.
        """
        if self.simulation_finished:
            return
        
        # print(f"--- Step {self.step_count} in Episode {self.current_episode} ---")
        self.schedule.step()

        # Calculate CO2 emissions for this step
        step_emissions = sum(
            agent.distance_travelled * self.co2_factors[agent.last_action]
            for agent in self.schedule.agents if not agent.completed
        )
        # print(f"Episode {self.current_episode}, Step {self.step_count}, Step Emissions: {step_emissions}")
        # print(f"Agent States: {[agent.last_action for agent in self.schedule.agents]}")
        # print(f"Agent Distances: {[agent.distance_travelled for agent in self.schedule.agents]}")

        self.current_episode_emissions += step_emissions
        self.total_co2_emissions += step_emissions
 
        self.step_count += 1

        if self.completed_agents >= self.num_agents:
            # Ensure emissions are appended only once per episode
            if len(self.co2_emissions_per_episode) <= self.current_episode:
                self.co2_emissions_per_episode.append(self.current_episode_emissions)

            print(f"--- Episode {self.current_episode} Completed ---")
            print(f"CO2 Emissions This Episode: {self.current_episode_emissions}")
            
            
            # Recalculate cumulative emissions
            self.co2_emissions_over_time = [
                sum(self.co2_emissions_per_episode[:i+1])
                for i in range(len(self.co2_emissions_per_episode))
            ]

            # Reset for the next episode
            self.current_episode_emissions = 0
            if self.current_episode < self.episodes - 1:
                self.current_episode += 1
                self.reset_environment()
            elif self.current_episode == self.episodes - 1:
                self.simulation_finished = True
                
        if self.simulation_finished:
                # At the end of the simulation, print the final credits for all agents
                print(f"--- Simulation Completed ---")
                print(f"Final Agent Credits:")
                for agent in self.schedule.agents:
                    print(f"Agent {agent.unique_id}: {agent.credits} credits")

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
        self.start = start_node
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
        self.credits = 0 #CREDIT SCHEME

        #CO2 emission per km for each transport mode (CREDIT SCHEME)
        self.co2_emissions_per_km = {
            "Bike": 0,
            "Car": 170,
            "PublicTransport": 97,
        }
        # HUMAN FACTOR PROPERTIES
        self.human_factor = random.uniform(0.5, 1.0)  # Resistance to change (0.5 - 1.0)
        self.defiance = random.uniform(0, 0.1)  # Small probability of defying logic (0-10%)
        self.last_action = "Car"  # Default transport mode
        
        # Extract initial transport mode from route name
        self.last_action = self.get_mode_from_route(route_name)
        
        # Determine origin and destination 
        if start_node == 4523960189:
            self.origin = "Asprela"
            self.destination = "Campo Alegre"
        elif start_node == 479183608:
            self.origin = "Campo Alegre"
            self.destination = "Asprela"
    
    def calculate_co2_emissions(self): #CREDIT SCHEME
        """
        Calculate the CO2 emissions for the distance travelled in the current step
        """
        distance_km = (self.speed * self.step_time)/1000
        co2_emission = self.co2_emissions_per_km[self.last_action]*distance_km
        return co2_emission
    def update_credits(self, co2_emission): #CREDIT SCHEME
        """
        Update the agent's credit based on CO2 emissions. 
        Penalty for high emissions, credits for low emissions.
        """
        if co2_emission == 0:
            self.credits += 10
        elif co2_emission >= 50:
            self.credits += 5
        elif co2_emission >= 100:
            self.credits -= 10 #penatly for high emissions

    def select_action(self):
        """
        Choose the next transport mode using ε-greedy policy while incorporating human factors and defiance.
        """
        state = self.get_state()
        # Ensure the state is initialized in the Q-table
        if state not in self.model.q_table:
            self.model.q_table[state] = {a: 0 for a in self.get_possible_actions()}
        # Check for defiance (irrational decision-making)
        if random.random() < self.defiance:
            # Defy logic: pick a random action regardless of Q-values
            return random.choice(self.get_possible_actions())
        # ε-greedy policy: explore or exploit
        if random.random() < self.model.epsilon:
            return random.choice(self.get_possible_actions())  # Explore
        # Incorporate human factor into exploitation
        best_action = max(self.get_possible_actions(), 
                        key=lambda action: self.model.q_table[state][action] * (1 - self.human_factor))
        return best_action

    def get_state(self):
        """
        Define the state of the agent based solely on the last chosen mode of transport.
        """
        return self.last_action
    
    def get_possible_actions(self):
        """Define possible actions (mocked for simplicity)."""
        return ["Bike", "Car", "PublicTransport"]
    
    def get_mode_from_route(self, route_name):
        """Extract the mode of transport from the route name."""
        if "Bike" in route_name:
            return "Bike"
        elif "Car" in route_name:
            return "Car"
        elif "PublicTransport" in route_name:
            return "PublicTransport"
        else:
            raise ValueError(f"Unknown transport mode in route name: {route_name}")

    def select_action(self):
        """Choose the next transport mode using ε-greedy policy."""
        state = self.get_state()

        # Ensure the state is initialized in the Q-table
        if state not in self.model.q_table:
            self.model.q_table[state] = {a: 0 for a in self.get_possible_actions()}

        # ε-greedy policy: explore or exploit
        if random.random() < self.model.epsilon:
            return random.choice(self.get_possible_actions())  # Explore
        return max(self.model.q_table[state], key=self.model.q_table[state].get)  # Exploit

    def update_q_value(self, state, action, reward, next_state):
        """
        Update the Q-value for the chosen mode of transport.
        """
        # Ensure the current state is initialized in the Q-table
        if state not in self.model.q_table:
            self.model.q_table[state] = {a: 0 for a in self.get_possible_actions()}

        # Ensure the next state is initialized in the Q-table
        if next_state not in self.model.q_table:
            self.model.q_table[next_state] = {a: 0 for a in self.get_possible_actions()}

        # Q-learning update rule
        current_q = self.model.q_table[state][action]
        max_next_q = max(self.model.q_table[next_state].values(), default=0)
        self.model.q_table[state][action] = current_q + self.model.alpha * (reward + self.model.gamma * max_next_q - current_q)


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

        #Calculate CO2 emissions and update credits (CREDIT SCHEME)
        co2_emission = self.calculate_co2_emissions()
        self.update_credits(co2_emission)

        # Check if the agent has completed the route
        if self.distance_travelled >= self.route_length:
            if not self.completed:
                self.distance_travelled = self.route_length  # Cap at total route length
                self.completed = True
                self.model.agent_completed(self.unique_id)
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
        
        # # Debugging output
        # print(
        #     f"Agent {self.unique_id} moving from {self.origin} to {self.destination}. Travel-Mode: {self.route_name.split('_')[-1]}. "
        #     f"Distance travelled: {self.distance_travelled:.2f} meters ({progress_percentage:.2f}% completed). "
        #     f"Elapsed time: {self.elapsed_time:.2f} seconds. Nodes left: {remaining_nodes_count-1}."
        # )

    def step(self):
        """
        Execute a step in the agent's decision-making process.
        """
        if self.completed:
            return

        # Update Q-value for the chosen action
        current_state = self.get_state()
        reward = self.model.compute_reward(self)  # Compute reward for the current mode
        next_action = self.select_action()  #HUMAN FACTOR
        self.update_q_value(current_state, self.last_action, reward, next_action)  # Next state is the same as selected mode
        
        self.last_action = next_action #HUMAN FACTOR

        # Simulate movement and completion logic
        self.move()
        
    def reset_for_new_episode(self):
        """Reset episode-specific attributes while retaining persistent data."""
        self.current_node = self.start  # Reset to start position
        self.distance_travelled = 0.0
        self.elapsed_time = 0.0
        self.completed = False
        self.step_cnt = 0
        self.edge_travelled = 0.0
        self.current_edge_index = 0
        self.counted = False

        # Select transport mode for the new episode
        self.last_action = self.select_action()


# Main execution
if __name__ == "__main__":
    # Define parameters
    nodes_and_edges_folder = "nodes_and_edges"
    combined_nodes_file = os.path.join(nodes_and_edges_folder, "all_routes_combined_nodes.csv")
    combined_edges_file = os.path.join(nodes_and_edges_folder, "all_routes_combined_edges.csv")
    num_agents = 10
    step_time_dimension = 60.0   # s/step aka the "resolution" of the simulation
    episodes = 10

    # Initialize the model
    model = TrafficModel(
        nodes_and_edges_folder,
        num_agents, 
        step_time_dimension, 
        episodes,
        combined_nodes_file=combined_nodes_file,
        combined_edges_file=combined_edges_file
    )

    # Run the simulation for a few steps
    # for i in range(1):
    #     print(f"--- Step {i + 1} ---")
    #     model.step()
        
    # Run the simulation until every agent has finished his travel
    step_count = 0
    episode_count = 0
    
    while not model.simulation_finished:
        model.step()
        step_count += 1
        if episode_count != model.current_episode:
            episode_count = model.current_episode
            step_count = 0
        