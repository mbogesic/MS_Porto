from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from model import TransportModel

def agent_portrayal(agent):
    # Default portrayal with shape, color, radius, and layer information
    portrayal = {
        "Shape": "circle",      # Use default circle
        "Color": "blue",        # Default color for agents
        "r": 0.8,               # Radius of the circle for visualization
        "Layer": 1              # Ensure 'Layer' key is present for rendering order
    }

    # Customizing color based on agent's age group
    if agent.age_group == "young_adults":
        portrayal["Color"] = "orange"
    elif agent.age_group == "adults":
        portrayal["Color"] = "blue"
    elif agent.age_group == "older_adults":
        portrayal["Color"] = "purple"
    elif agent.age_group == "seniors":
        portrayal["Color"] = "green"

    # Remove custom shapes and use standard shapes
    if agent.transport_mode == "bike":
        portrayal["Shape"] = "circle"  # Keep the shape as circle
        portrayal["Color"] = "yellow"   # Different color for bike users
    elif agent.transport_mode == "car_petrol":
        portrayal["Shape"] = "circle"    # Keep as circle
        portrayal["Color"] = "red"        # Different color for petrol cars
    elif agent.transport_mode == "car_electric":
        portrayal["Shape"] = "circle"    # Keep as circle
        portrayal["Color"] = "blue"       # Different color for electric cars
    elif agent.transport_mode == "public_transport":
        portrayal["Shape"] = "circle"     # Keep as circle
        portrayal["Color"] = "green"       # Different color for public transport

    return portrayal


# Create the grid and chart modules
grid = CanvasGrid(agent_portrayal, 20, 20, 500, 500)  # Adjust grid size and display size as needed
chart = ChartModule([{"Label": "CO2 Emissions", "Color": "Red"}])  # Add CO2 emissions chart

# Set up the server with the model, visualization elements, and model parameters
server = ModularServer(
    TransportModel,
    [grid, chart],
    "AMC Simulation",
    {"num_agents": 100, "base_credit": 10, "place_name": "Porto, Portugal"}  # Passing the place name for OSMNX
)

server.port = 8521
server.launch()
