import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import os
from PIL import Image
import base64

# Automatically change the working directory to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
os.chdir(script_dir)  # Change the working directory

print(f"Working directory changed to: {os.getcwd()}")
# Load the Traffic Model (Ensure your model script is accessible)
from simulation import TrafficModel

# Initialize the Traffic Model and define parameters
nodes_and_edges_folder = "nodes_and_edges"
combined_nodes_file = os.path.join(nodes_and_edges_folder, "all_routes_combined_nodes.csv")
combined_edges_file = os.path.join(nodes_and_edges_folder, "all_routes_combined_edges.csv")
num_agents = 500
step_time_dimension = 10.0   # s/step aka the "resolution" of the simulation
# Global list to track CO2 emissions over time
co2_emissions_over_time = []

# Initialize the model
model = TrafficModel(
        nodes_and_edges_folder,
        num_agents,  
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
    
html.Div([
    # First column for KPIs and plots
    html.Div([
        html.Div([
            html.H3(id="kpi-co2-total", style={"text-align": "left", "margin-bottom": "10px"}),  
            html.Div(id="kpi-co2-increase", style={"width": "100%", "margin-bottom": "10px", "text-align": "left"}), 
            html.Div(id="kpi-co2-a2c", style={"width": "100%", "margin-bottom": "10px", "text-align": "left"}),  
            html.Div(id="kpi-co2-c2a", style={"width": "100%", "margin-bottom": "10px", "text-align": "left"}),  
            html.Div(id="kpi-completion", style={"width": "100%", "margin-bottom": "10px", "text-align": "left"}),  
        ], style={"margin-bottom": "20px", "display": "flex", "flex-direction": "column", "align-items": "flex-start"}),

        # Optional: Placeholder for additional plots
        dcc.Graph(id="metric-plot", style={"width": "100%", "height": "400px"}),  # Adjust height as needed
    ], style={"flex": "1", "padding": "10px"}),  # First column styling

    # Second column for the graph visualization
    html.Div([
        dcc.Graph(id="route-graph", style={"width": "100%", "height": "600px"}),  # Adjust height as needed
    ], style={"flex": "2", "padding": "10px"}),  # Second column styling
], style={"display": "flex", "flex-direction": "row", "width": "100%"}),


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
        Output("kpi-co2-total", "children"),
        Output("kpi-co2-increase", "children"),
        Output("kpi-co2-a2c", "children"),
        Output("kpi-co2-c2a", "children"),
        Output("kpi-completion", "children"),
        Output("route-graph", "figure"),
        Output("metric-plot", "figure"),
    ],
    [
        Input("interval-component", "n_intervals"),
    ]
)
def update_dashboard(n):
    global co2_emissions_over_time  # Access the global variable
    car_co2_emission_factor = 0.17 # g/m - According to https://ourworldindata.org/travel-carbon-footprint
    pt_co2_emission_factor = 0.097  # g/m 
    # Step the model
    if not model.simulation_finished:
        model.step()

    # Agent information
    num_agents_a2c_bike = len([agent for agent in model.schedule.agents if agent.route_name == "Asprela_2_Campo_Alegre_Bike"])
    num_agents_a2c_car = len([agent for agent in model.schedule.agents if agent.route_name == "Asprela_2_Campo_Alegre_Car"])
    num_agents_a2c_pt = len([agent for agent in model.schedule.agents if agent.route_name == "Asprela_2_Campo_Alegre_PublicTransport"])
    num_agents_c2a_bike = len([agent for agent in model.schedule.agents if agent.route_name == "Campo_Alegre_2_Asprela_Bike"])
    num_agents_c2a_car = len([agent for agent in model.schedule.agents if agent.route_name == "Campo_Alegre_2_Asprela_Car"])
    num_agents_c2a_pt = len([agent for agent in model.schedule.agents if agent.route_name == "Campo_Alegre_2_Asprela_PublicTransport"])

    # KPIs
    a2c_pt_co2_emissions = sum(agent.distance_travelled * pt_co2_emission_factor for agent in model.schedule.agents if agent.route_name == "Asprela_2_Campo_Alegre_PublicTransport")
    a2c_car_co2_emissions = sum(agent.distance_travelled * car_co2_emission_factor for agent in model.schedule.agents if agent.route_name == "Asprela_2_Campo_Alegre_Car")
    c2a_car_co2_emissions = sum(agent.distance_travelled * car_co2_emission_factor for agent in model.schedule.agents if agent.route_name == "Campo_Alegre_2_Asprela_Car")
    c2a_pt_co2_emissions = sum(agent.distance_travelled * pt_co2_emission_factor for agent in model.schedule.agents if agent.route_name == "Campo_Alegre_2_Asprela_PublicTransport")
    total_co2_emissions = a2c_pt_co2_emissions + a2c_car_co2_emissions + c2a_car_co2_emissions + c2a_pt_co2_emissions
    completion_rate = f"{model.completed_agents}/{model.num_agents} agents completed"
    # Calculate CO2 emission rate
    if len(co2_emissions_over_time) > 0:
        co2_increase_rate = (total_co2_emissions - co2_emissions_over_time[-1]) / step_time_dimension
    else:
        co2_increase_rate = 0
    # Append the current total CO2 emissions to the global list
    co2_emissions_over_time.append(total_co2_emissions)
    
    # Time Series Plot for CO2 Emissions
    co2_time_series = go.Figure(
        data=[
            go.Scatter(
                x=list(range(len(co2_emissions_over_time))),  # X-axis: time steps
                y=co2_emissions_over_time,  # Y-axis: CO2 emissions
                mode="lines+markers",
                name="Total CO2 Emissions",
            )
        ],
        layout=go.Layout(
            title="Total CO2 Emissions Over Time",
            xaxis=dict(title=f"Time Steps ({step_time_dimension}s/step)"),
            yaxis=dict(title="CO2 Emissions (g)"),
        ),
    )
    
    # Create figure for Street Network
    fig = go.Figure()

    routes_colors = ['rgba(0, 128, 0, 0.8)', 
                     'rgba(0, 0, 255, 0.8)', 
                     'rgba(255, 0, 0, 1)', 
                     'rgba(255, 255, 0, 0.8)', 
                     'rgba(255, 0, 255, 1)', 
                     'rgba(0, 255, 255, 0.8)' ]
    
    width_normalization_factor = 200
    a2c_car_width = 1 + a2c_car_co2_emissions / (num_agents_a2c_car * width_normalization_factor)
    a2c_pt_width = 1 + a2c_pt_co2_emissions / (num_agents_a2c_pt * width_normalization_factor)
    c2a_car_width = 1 + c2a_car_co2_emissions / (num_agents_c2a_car * width_normalization_factor)
    c2a_pt_width = 1 + c2a_pt_co2_emissions / (num_agents_c2a_pt * width_normalization_factor)
    
    # Add edges using scaled positions
    for route, name, color in zip(model.routes_visuals, model.route_names, routes_colors):
        if name == "Asprela_2_Campo_Alegre_Car":
            width = a2c_car_width
        elif name == "Asprela_2_Campo_Alegre_PublicTransport":
            width = a2c_pt_width
        elif name == "Campo_Alegre_2_Asprela_Car":
            width = c2a_car_width
        elif name == "Campo_Alegre_2_Asprela_PublicTransport":
            width = c2a_pt_width
        else:
            width = 1  # Default width for other routes
        for edge in route.edges:
            x0, y0 = model.scaled_positions[edge[0]]
            x1, y1 = model.scaled_positions[edge[1]]
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode="lines",
                line=dict(color=color, width=width),
                showlegend=False
            ))
    # Add legend for routes
    route_labels = ['A -> CA - Bike (5501.72m)', 
                    'A -> CA - Car (6037.01m)', 
                    'A -> CA - Public Transport (7593.82m)', 
                    'CA -> A - Bike (5486.20m)', 
                    'CA -> A - Car (6671.41m)', 
                    'CA -> A - Public Transport (6984.30m)']
    for color, label in zip(routes_colors, route_labels):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color=color, width=2),
            name=label,  # Add legend entry
        ))
        
        fig.update_layout(
            title="Route Network",
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
        f"Total CO2 Emissions: {total_co2_emissions:.2f} g",
        dcc.Markdown(f"(Increase Rate: {co2_increase_rate:.2f} g/s)"),
        dcc.Markdown(f"CO2 Emissions for Asprela -> Campo Alegre:\n- Bike: {num_agents_a2c_bike} Agents\n- Car: {num_agents_a2c_car} Agents ({a2c_car_co2_emissions:.2f} g)\n- Public Transport: {num_agents_a2c_pt} Agents ({a2c_pt_co2_emissions:.2f} g)"),
        dcc.Markdown(f"CO2 Emissions for Campo Alegre -> Asprela:\n- Bike: {num_agents_c2a_bike} Agents\n- Car: {num_agents_c2a_car} Agents ({c2a_car_co2_emissions:.2f} g)\n- Public Transport: {num_agents_c2a_pt} Agents ({c2a_pt_co2_emissions:.2f} g)"),
        completion_rate,
        fig,
        co2_time_series,
    )
    
# Run the App
if __name__ == "__main__":
    app.run_server(debug=True)