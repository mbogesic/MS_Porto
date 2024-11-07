import osmnx as ox
from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from amc_credit_scheme import AMCCreditScheme
from agent import TransportAgent
from age_groups import AGE_GROUPS
from data_collector import create_data_collector
import random

class TransportModel(Model):
    def __init__(self, num_agents, base_credit, place_name="Porto, Portugal", width=10, height=10):
        super().__init__()
        self.num_agents = num_agents
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(width, height, torus=True)
        self.co2_credit_scheme = AMCCreditScheme(base_credit)
        
        # Example bounding box for a specific area in Porto
        # New bounding box format: (left, bottom, right, top)
        bbox = (-8.64212, 41.14977, -8.59217, 41.18227)  # W, S, E, N
        self.graph = ox.graph_from_bbox(bbox[1], bbox[0], bbox[3], bbox[2], network_type='all')  # (south, west, north, east)
      
        # Define your start point (ASPRELA)
        self.start_lat, self.start_lon = 41.177885, -8.599197
        # Define your endpoint (CAMPO ALEGRE)
        self.end_lat, self.end_lon = 41.152750, -8.637213  

        start_node = ox.distance.nearest_nodes(self.graph, self.start_lon, self.start_lat)
        end_node = ox.distance.nearest_nodes(self.graph, self.end_lon, self.end_lat)

        # Calculate the route
        self.route = ox.shortest_path(self.graph, start_node, end_node)

        # You can now use self.route to inform agent movements or visualizations

                
        # Calculate the bounds of the area from the graph nodes
        node_coords = [(data['x'], data['y']) for _, data in self.graph.nodes(data=True)]
        x_coords, y_coords = zip(*node_coords)
        north = max(y_coords)
        south = min(y_coords)
        east = max(x_coords)
        west = min(x_coords)
        
        # Create DataCollector
        self.datacollector = create_data_collector(self)
        
        # Create agents
        for i in range(self.num_agents):
            age_group = self.assign_age_group()
            transport_mode = self.assign_transport_mode(age_group)
            agent = TransportAgent(i, self, age_group, transport_mode)
            self.schedule.add(agent)

            # Place agent on a random node in the graph
            node = random.choice(list(self.graph.nodes))
            x_geo, y_geo = self.graph.nodes[node]['x'], self.graph.nodes[node]['y']
            
            # Normalize geographic coordinates to grid indices
            x_index = int((x_geo - west) / (east - west) * width)  # Normalize to grid width
            y_index = int((north - y_geo) / (north - south) * height)  # Normalize to grid height
            
            # Make sure indices are within bounds
            x_index = min(max(x_index, 0), width - 1)
            y_index = min(max(y_index, 0), height - 1)

            # Place agent in the grid using the calculated indices
            self.grid.place_agent(agent, (x_index, y_index))
        
    def assign_age_group(self):
        return random.choice(list(AGE_GROUPS.keys()))
    
    def assign_transport_mode(self, age_group):
        transport_modes = {
            "young_adults": ["bike", "public_transport", "car_petrol"],
            "adults": ["car_diesel", "car_electric", "public_transport"],
            "older_adults": ["car_petrol", "public_transport"],
            "seniors": ["public_transport", "bike"]
        }
        return random.choice(transport_modes.get(age_group, ["public_transport"]))

    def step(self):
        for agent in self.schedule.agents:
            self.co2_credit_scheme.apply_credit(agent)
            agent.step()
            
        self.schedule.step()
        self.datacollector.collect(self)  # Collect data after each step
