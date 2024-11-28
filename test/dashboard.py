import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import os
from PIL import Image
import base64

# Load the Traffic Model (Ensure your model script is accessible)
from simulation import TrafficModel

# Initialize the Traffic Model and define parameters
nodes_and_edges_folder = "nodes_and_edges"
combined_nodes_file = os.path.join(nodes_and_edges_folder, "all_routes_combined_nodes.csv")
combined_edges_file = os.path.join(nodes_and_edges_folder, "all_routes_combined_edges.csv")
num_agents = 5
agent_speed = 3.9583333     # m/s (assuming an avg speed of 14.25km/h)
step_time_dimension = 3.0   # s/step aka the "resolution" of the simulation

# Initialize the model
model = TrafficModel(
        nodes_and_edges_folder,
        num_agents, 
        agent_speed, 
        step_time_dimension, 
        combined_nodes_file=combined_nodes_file,
        combined_edges_file=combined_edges_file
    )

# Start the Dash App
app = dash.Dash(__name__)
app.title = "AMC Simulation Dashboard"

# Layout for Dashboard
app.layout = html.Div([
    html.H1("AMC Simulation Dashboard"),
    
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
        interval=1000,  # 1000 ms = 1 second. Don't go below 330ms (3 steps per second)
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
    ]
)
def update_dashboard(n):
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

    routes_colors = ['rgba(0, 128, 0, 0.8)', 
                     'rgba(0, 0, 255, 0.8)', 
                     'rgba(255, 0, 0, 1)', 
                     'rgba(255, 255, 0, 0.8)', 
                     'rgba(255, 0, 255, 1)', 
                     'rgba(0, 255, 255, 0.8)' ]
    
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
    route_labels = ['A -> CA - Route 1 (5501.72m)', 
                    'A -> CA - Route 2 (6037.01m)', 
                    'A -> CA - Route 3 (7593.82m)', 
                    'CA -> A - Route 1 (5486.20m)', 
                    'CA -> A - Route 2 (6671.41m)', 
                    'CA -> A - Route 3 (6984.30m)']
    for color, label in zip(routes_colors, route_labels):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color=color, width=2),
            name=label,  # Add legend entry
        ))
        
    for agent in model.schedule.agents:
        if not agent.completed:
            # Ensure we are accessing the correct normalized route edges
            if agent.current_edge_index < len(agent.normalized_route_edges):
                edge = agent.normalized_route_edges[agent.current_edge_index]
                x0, y0 = edge["start_pos"]
                x1, y1 = edge["end_pos"]

                # Calculate the interpolation factor
                edge_length = edge["length"]
                interpolation_factor = agent.edge_travelled / edge_length

                # Interpolated position
                x = x0 + interpolation_factor * (x1 - x0)
                y = y0 + interpolation_factor * (y1 - y0)

                # # Debugging: Check if the agent's route matches the visualization
                # print(f"Agent {agent.unique_id} is on edge: {edge['start_node']} -> {edge['end_node']}")
                # print(f"Position: ({x:.2f}, {y:.2f}), Interpolation Factor: {interpolation_factor:.2f}")

                # Plot the agent
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode="markers",
                    marker=dict(size=10, color="beige", symbol="circle"),
                    name=f"Agent {agent.unique_id}",
                    showlegend=True
                ))

        fig.update_layout(
            title="Traffic Simulation",
            xaxis=dict(visible=False, range=[0, 1], scaleanchor="y"),
            yaxis=dict(visible=False, range=[0, 1]),
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
            plot_bgcolor="rgba(0,20,40,0.4)",
            uirevision="constant"
        )
        
    return (
        f"CO2 Emissions: {co2_emissions:.2f} g",
        completion_rate,
        fig,
    )
    
# Run the App
if __name__ == "__main__":
    app.run_server(debug=True)
