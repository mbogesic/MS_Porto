import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import os
from PIL import Image
import base64

# Convert the image to base64 encoding
def encode_image(image_path):
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode()
    return "data:image/png;base64," + encoded_image

# Load the Traffic Model (Ensure your model script is accessible)
from simulation import TrafficModel

# Initialize the Traffic Model and define parameters
nodes_and_edges_folder = "nodes_and_edges"
combined_nodes_file = os.path.join(nodes_and_edges_folder, "all_routes_combined_nodes.csv")
combined_edges_file = os.path.join(nodes_and_edges_folder, "all_routes_combined_edges.csv")
num_agents = 50
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

# Load the background image and get its dimensions
background_image_file = "all_routes_combined_subgraph.png"
background_image = Image.open(background_image_file)
# Encode the image
background_image_encoded = encode_image(background_image_file)
image_width, image_height = background_image.size

# Start the Dash App
app = dash.Dash(__name__)
app.title = "Traffic Simulation Dashboard"

# Layout for Dashboard
app.layout = html.Div([
    html.H1("Traffic Simulation Dashboard"),
    
    # KPIs Section
    html.Div([
        html.Div(id="kpi-co2", style={"width": "30%", "display": "inline-block"}),
        html.Div(id="kpi-completion", style={"width": "30%", "display": "inline-block"}),
    ], style={"margin-bottom": "20px"}),

    # Graph Visualization
    dcc.Graph(id="route-graph"),

    # Interval for periodic updates
    dcc.Interval(
        id="interval-component",
        interval=1000,  # 1000 ms = 1 second
        n_intervals=0
    ),
])

# Callbacks to Update Metrics and Visualizations
@app.callback(
    [
        Output("kpi-co2", "children"),
        Output("kpi-completion", "children"),
        Output("route-graph", "figure"),
    ],
    [
        Input("interval-component", "n_intervals"),
        State("route-graph", "relayoutData"),
    ]
)
def update_dashboard(n, relayout_data):
    # Step the model
    if not model.simulation_finished:
        model.step()

    # KPIs
    co2_emissions = sum(agent.distance_travelled * 0.2 for agent in model.schedule.agents)
    completion_rate = f"{model.completed_agents}/{model.num_agents} agents completed"

    # Create figure with background image
    fig = go.Figure()

    # # Add the PNG as a background image
    # fig.add_layout_image(
    #     dict(
    #         source=background_image_encoded,
    #         x=0,
    #         y=1,
    #         xref="paper",
    #         yref="paper",
    #         sizex=1,
    #         sizey=1,
    #         xanchor="left",
    #         yanchor="top",
    #         layer="below"
    #     )
    # )

    routes_colors = ['green', 'blue', 'red', 'yellow', 'magenta', 'cyan' ]
    # Add edges using scaled positions
    for route, color in zip(model.routes_visuals, routes_colors):
        for edge in route.edges:
            x0, y0 = model.scaled_positions[edge[0]]
            x1, y1 = model.scaled_positions[edge[1]]
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode="lines",
                line=dict(color=color, width=1),
                showlegend=False
            ))
    # Add legend for routes
    route_colors = ['green', 'blue', 'red', 'yellow', 'magenta', 'cyan']
    route_labels = ['Route A', 'Route B', 'Route C', 'Route D', 'Route E', 'Route F']
    for color, label in zip(route_colors, route_labels):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color=color, width=2),
            name=label,  # Add legend entry
        ))
        
    # Add agents dynamically
    for agent in model.schedule.agents:
        if not agent.completed:
            x, y = model.scaled_positions[agent.current_node]
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers",
                marker=dict(size=10, color="beige", symbol="circle"),
                name=f"Agent {agent.unique_id}",
                showlegend=True  # Avoid duplicate legend entries for agents
            ))

    # Use existing layout ranges if available
    x_range = [0, 1]
    y_range = [0, 1]
    if relayout_data:
        x_range = relayout_data.get("xaxis.range", x_range)
        y_range = relayout_data.get("yaxis.range", y_range)

    fig.update_layout(
        title="Traffic Simulation",
        xaxis=dict(visible=False, range=x_range, scaleanchor="y"),
        yaxis=dict(visible=False, range=y_range),
        showlegend=True,
        legend=dict(
            x=1.05,
            y=1,
            title="Legend",
            font=dict(size=12),
            bgcolor="rgba(0,20,40,0.4)",
        ),
        width=1000,
        height=800,
        plot_bgcolor="rgba(0,20,40,0.4)"
    )

    return (
        f"CO2 Emissions: {co2_emissions:.2f} g",
        completion_rate,
        fig,
    )
    
# Run the App
if __name__ == "__main__":
    app.run_server(debug=True)
