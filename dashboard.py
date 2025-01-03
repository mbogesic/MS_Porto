import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import os
import Formulas as f

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
num_agents = 1000
step_time_dimension = 60.0   # s/step aka the "resolution" of the simulation
episodes = 30

# # Global list to track CO2 emissions over time
# co2_emissions_over_time = []  # Running total of CO2 emissions
# co2_emissions_over_episodes = []  # Total CO2 emissions per episode

# Initialize the model
model = TrafficModel(
        nodes_and_edges_folder,
        num_agents,  
        step_time_dimension, 
        episodes,
        combined_nodes_file=combined_nodes_file,
        combined_edges_file=combined_edges_file
    )

# Start the Dash App
app = dash.Dash(__name__)
app.title = "AMC Simulation Dashboard"

app.layout = html.Div([
    html.H1("AMC Simulation Dashboard"),
    dcc.Graph(id="metric-plot", style={"width": "100%", "height": "400px"}),
    dcc.Graph(id="episode-plot", style={"width": "100%", "height": "400px"}),
    dcc.Interval(
        id="interval-component",
        interval=1000,  # 1000ms = 1 second
        n_intervals=0
    ),
])


@app.callback(
    [
        Output("metric-plot", "figure"),
        Output("episode-plot", "figure"),
    ],
    [Input("interval-component", "n_intervals")]
)
def update_plots(n_intervals):
    # Check if the simulation is finished or a new episode is completed
    if model.simulation_finished or model.current_episode > 0:
        # Fetch latest data
        data = model.get_episode_data()
        # print(f"CO2 Emissions Per Episode: {data['co2_emissions_per_episode']}")
        # Generate cumulative plot
        cumulative_plot = go.Figure(
            data=[
                go.Scatter(
                    x=list(range(1, len(data["co2_emissions_over_time"]) + 1)),
                    y=data["co2_emissions_over_time"],
                    mode="lines+markers",
                    name="Cumulative CO2 Emissions",
                )
            ],
            layout=go.Layout(
                title="Cumulative CO2 Emissions Over Episodes",
                xaxis=dict(title="Episodes"),
                yaxis=dict(title="CO2 Emissions (g)"),
            ),
        )

        # Generate episode-wise CO2 plot
        episode_plot = go.Figure(
            data=[
                go.Bar(
                    x=list(range(1, len(data["co2_emissions_per_episode"]) + 1)),
                    y=data["co2_emissions_per_episode"],
                    name="CO2 Emissions Per Episode",
                )
            ],
            layout=go.Layout(
                title="CO2 Emissions Per Episode",
                xaxis=dict(title="Episodes"),
                yaxis=dict(title="CO2 Emissions (g)"),
            ),
        )

        # Reset the flag to avoid redundant updates
        model.current_episode = 0

        return cumulative_plot, episode_plot

    # Prevent updates if no new data
    raise dash.exceptions.PreventUpdate

    
if __name__ == "__main__":
    while not model.simulation_finished:
        model.step()
    app.run_server(debug=True)
    
    
    ####################################################################
    ## TO BE INTEGRATED AFTER CREDIT SCHEME AND Q-LEARNING IS WORKING ##
    ####################################################################
    
    # app.layout = html.Div([
#     html.H1("AMC Simulation Dashboard"),
    
#     html.Button("Show Results", id="show-results-button", n_clicks=0),

#     # KPIs and CO2 Emissions Plots (First row)
#     html.Div([
#         # First column for KPIs and cumulative CO2 plot
#         html.Div([
#             html.Div([
#                 html.H3(id="kpi-co2-total", style={"text-align": "left", "margin-bottom": "10px"}),  
#                 html.Div(id="kpi-co2-increase", style={"width": "100%", "margin-bottom": "10px", "text-align": "left"}), 
#                 html.Div(id="kpi-co2-a2c", style={"width": "100%", "margin-bottom": "10px", "text-align": "left"}),  
#                 html.Div(id="kpi-co2-c2a", style={"width": "100%", "margin-bottom": "10px", "text-align": "left"}),  
#                 html.Div(id="kpi-completion", style={"width": "100%", "margin-bottom": "10px", "text-align": "left"}),  
#             ], style={"margin-bottom": "20px", "display": "flex", "flex-direction": "column", "align-items": "flex-start"}),

#             # Cumulative CO2 Plot
#             dcc.Graph(id="metric-plot", style={"width": "100%", "height": "400px"}),  # Adjust height as needed
#         ], style={"flex": "1", "padding": "10px"}),  # First column styling

#         # Episode-Wise CO2 Plot
#         html.Div([
#             dcc.Graph(id="episode-plot", style={"width": "100%", "height": "400px"}),
#         ], style={"flex": "1", "padding": "10px"}),  # Second column styling
#     ], style={"display": "flex", "flex-direction": "row", "width": "100%"}),  # First row styling

#     # Route Visualization (Second row)
#     html.Div([
#         dcc.Graph(id="route-graph", style={"width": "100%", "height": "600px"}),  # Adjust height as needed
#     ], style={"padding": "10px"}),  # Second row styling
    
#     dcc.Interval(
#         id="interval-component",
#         interval=1000,  # 1000ms = 1 second
#         n_intervals=0
#     ),
# ])


# # Callbacks to Update Metrics and Visualizations
# @app.callback(
#     [
#         Output("kpi-co2-total", "children"),
#         Output("kpi-co2-increase", "children"),
#         Output("kpi-co2-a2c", "children"),
#         Output("kpi-co2-c2a", "children"),
#         Output("kpi-completion", "children"),
#         Output("route-graph", "figure"),
#         Output("metric-plot", "figure"),
#         Output("episode-plot", "figure"),
#     ],
#     [Input("interval-component", "n_intervals")]
# )
# def update_dashboard(n):
#     global co2_emissions_over_episodes  # Access the global variable
    
#     # Step the model
#     if not model.simulation_finished:
        
#         episode_completed = model.step()
        
#         if episode_completed:
                
#             # Track cumulative CO2 emissions
#             if len(co2_emissions_over_time) == 0 or model.total_co2_emissions != co2_emissions_over_time[-1]:
#                 co2_emissions_over_time.append(model.total_co2_emissions)

#             # Track per-episode CO2 emissions
#             if len(model.co2_emissions_per_episode) > len(co2_emissions_over_episodes):
#                 co2_emissions_over_episodes.append(model.co2_emissions_per_episode[-1])

#             # Agent information
#             num_agents_a2c_bike = len([agent for agent in model.schedule.agents if agent.route_name == "Asprela_2_Campo_Alegre_Bike"])
#             num_agents_a2c_car = len([agent for agent in model.schedule.agents if agent.route_name == "Asprela_2_Campo_Alegre_Car"])
#             num_agents_a2c_pt = len([agent for agent in model.schedule.agents if agent.route_name == "Asprela_2_Campo_Alegre_PublicTransport"])
#             num_agents_c2a_bike = len([agent for agent in model.schedule.agents if agent.route_name == "Campo_Alegre_2_Asprela_Bike"])
#             num_agents_c2a_car = len([agent for agent in model.schedule.agents if agent.route_name == "Campo_Alegre_2_Asprela_Car"])
#             num_agents_c2a_pt = len([agent for agent in model.schedule.agents if agent.route_name == "Campo_Alegre_2_Asprela_PublicTransport"])

#             # KPIs
#             # a2c_pt_co2_emissions = sum(agent.distance_travelled * pt_co2_emission_factor for agent in model.schedule.agents if agent.route_name == "Asprela_2_Campo_Alegre_PublicTransport")
#             # a2c_car_co2_emissions = sum(agent.distance_travelled * car_co2_emission_factor for agent in model.schedule.agents if agent.route_name == "Asprela_2_Campo_Alegre_Car")
#             # a2c_bike_co2_emissions = sum(agent.distance_travelled * bike_co2_emission_factor for agent in model.schedule.agents if agent.route_name == "Asprela_2_Campo_Alegre_Bike")
#             # c2a_car_co2_emissions = sum(agent.distance_travelled * car_co2_emission_factor for agent in model.schedule.agents if agent.route_name == "Campo_Alegre_2_Asprela_Car")
#             # c2a_pt_co2_emissions = sum(agent.distance_travelled * pt_co2_emission_factor for agent in model.schedule.agents if agent.route_name == "Campo_Alegre_2_Asprela_PublicTransport")
#             # c2a_bike_co2_emissions = sum(agent.distance_travelled * bike_co2_emission_factor for agent in model.schedule.agents if agent.route_name == "Campo_Alegre_2_Asprela_Bike")
#             # total_co2_emissions = a2c_pt_co2_emissions + a2c_car_co2_emissions + a2c_bike_co2_emissions + c2a_car_co2_emissions + c2a_pt_co2_emissions + c2a_bike_co2_emissions
            
#             # Calculate emissions
#             emission_data = f.calculate_co2_emissions(model.schedule.agents, model.co2_factors)

#             # Access emissions from the returned dictionary
#             a2c_pt_co2_emissions = emission_data["Asprela_2_Campo_Alegre_PublicTransport"]
#             a2c_car_co2_emissions = emission_data["Asprela_2_Campo_Alegre_Car"]
#             a2c_bike_co2_emissions = emission_data["Asprela_2_Campo_Alegre_Bike"]
#             c2a_pt_co2_emissions = emission_data["Campo_Alegre_2_Asprela_PublicTransport"]
#             c2a_car_co2_emissions = emission_data["Campo_Alegre_2_Asprela_Car"]
#             c2a_bike_co2_emissions = emission_data["Campo_Alegre_2_Asprela_Bike"]
#             total_co2_emissions = emission_data["total"]
#             completion_rate = f"{model.completed_agents}/{model.num_agents} agents completed"
            
#             # Calculate CO2 emission rate
#             if len(co2_emissions_over_episodes) > 0:
#                 co2_increase_rate = (total_co2_emissions - co2_emissions_over_episodes[-1]) / step_time_dimension
#             else:
#                 co2_increase_rate = 0
#             # Append the current total CO2 emissions to the global list
#             co2_emissions_over_episodes.append(total_co2_emissions)
            
#             # Update plots and other metrics
#             cumulative_plot = go.Figure(
#                 data=[
#                     go.Scatter(
#                         x=list(range(len(co2_emissions_over_time))),
#                         y=co2_emissions_over_time,
#                         mode="lines+markers",
#                         name="Total CO2 Emissions",
#                     )
#                 ],
#                 layout=go.Layout(
#                     title="Cumulative CO2 Emissions Over Time",
#                     xaxis=dict(title="Simulation Steps"),
#                     yaxis=dict(title="CO2 Emissions (g)"),
#                 ),
#             )
            
#             episode_plot = go.Figure(
#                 data=[
#                     go.Bar(
#                         x=list(range(len(co2_emissions_over_episodes))),
#                         y=co2_emissions_over_episodes,
#                         name="CO2 Emissions Per Episode",
#                     )
#                 ],
#                 layout=go.Layout(
#                     title="CO2 Emissions Per Episode",
#                     xaxis=dict(title="Episodes"),
#                     yaxis=dict(title="CO2 Emissions (g)"),
#                 ),
#             )
#             # Create figure for Street Network
#             fig = go.Figure()
            
#             width_normalization_factor = 200
#             a2c_car_width = 1 + a2c_car_co2_emissions / (num_agents_a2c_car * width_normalization_factor)
#             a2c_pt_width = 1 + a2c_pt_co2_emissions / (num_agents_a2c_pt * width_normalization_factor)
#             a2c_bike_width = 1 + a2c_bike_co2_emissions / (num_agents_a2c_bike + width_normalization_factor)
#             c2a_car_width = 1 + c2a_car_co2_emissions / (num_agents_c2a_car * width_normalization_factor)
#             c2a_pt_width = 1 + c2a_pt_co2_emissions / (num_agents_c2a_pt * width_normalization_factor)
#             c2a_bike_width = 1 + c2a_bike_co2_emissions / (num_agents_c2a_bike + width_normalization_factor)

#             routes_data = [
#                 ("Asprela_2_Campo_Alegre_Car", a2c_car_width, 'rgba(0, 0, 255, 0.6)'),
#                 ("Asprela_2_Campo_Alegre_PublicTransport", a2c_pt_width, 'rgba(255, 0, 0, 0.6)'),
#                 ("Asprela_2_Campo_Alegre_Bike", a2c_bike_width, 'rgba(0, 128, 0, 1)'),
#                 ("Campo_Alegre_2_Asprela_Car", c2a_car_width, 'rgba(255, 0, 255, 0.6)'),
#                 ("Campo_Alegre_2_Asprela_PublicTransport", c2a_pt_width, 'rgba(0, 255, 255, 0.6)'),
#                 ("Campo_Alegre_2_Asprela_Bike", c2a_bike_width, 'rgba(255, 255, 0, 1)'),
#             ]
            
#             routes_colors = ['rgba(0, 128, 0, 1)',  # A->CA Bike, Green
#                     'rgba(0, 0, 255, 0.5)',         # A->CA Car, Blue
#                     'rgba(255, 0, 0, 0.8)',         # A->CA PT, Red
#                     'rgba(255, 255, 0, 1)',         # CA->A Bike, Yellow
#                     'rgba(255, 0, 255, 0.5)',       # CA->A Car, Magenta
#                     'rgba(0, 255, 255, 0.8)'        # CA->A PT, Cyan
#                     ]

#             # Sort by width to ensure thinner routes are added later
#             routes_data = sorted(routes_data, key=lambda x: x[1], reverse=True)

#             # Add edges using scaled positions
#             for route, name, color in zip(model.routes_visuals, model.route_names, routes_colors):
#                 if name == "Asprela_2_Campo_Alegre_Car":
#                     width = a2c_car_width
#                 elif name == "Asprela_2_Campo_Alegre_PublicTransport":
#                     width = a2c_pt_width
#                 elif name == "Campo_Alegre_2_Asprela_Car":
#                     width = c2a_car_width
#                 elif name == "Campo_Alegre_2_Asprela_PublicTransport":
#                     width = c2a_pt_width
#                 elif name == "Asprela_2_Campo_Alegre_Bike":
#                     width = a2c_bike_width
#                 elif name == "Campo_Alegre_2_Asprela_Bike":
#                     width = c2a_bike_width
#                 else:
#                     width = 1  # Default width for other routes
#                 for edge in route.edges:
#                     x0, y0 = model.scaled_positions[edge[0]]
#                     x1, y1 = model.scaled_positions[edge[1]]
#                     fig.add_trace(go.Scatter(
#                         x=[x0, x1], y=[y0, y1],
#                         mode="lines",
#                         line=dict(color=color, width=width),
#                         showlegend=False
#                     ))

#             # Add legend for routes
#             route_labels = ['A -> CA - Bike (5501.72m)', 
#                             'A -> CA - Car (6037.01m)', 
#                             'A -> CA - Public Transport (7593.82m)', 
#                             'CA -> A - Bike (5486.20m)', 
#                             'CA -> A - Car (6671.41m)', 
#                             'CA -> A - Public Transport (6984.30m)']
#             for color, label in zip(routes_colors, route_labels):
#                 fig.add_trace(go.Scatter(
#                     x=[None], y=[None],
#                     mode="lines",
#                     line=dict(color=color, width=2),
#                     name=label,  # Add legend entry
#                 ))
                
#                 fig.update_layout(
#                     title="Route Network",
#                     xaxis=dict(visible=False, range=[0, 1], scaleanchor="y"),
#                     yaxis=dict(visible=False, range=[0, 1]),
#                     showlegend=True,
#                     legend=dict(
#                         x=1.05,
#                         y=1,
#                         title="Legend",
#                         font=dict(size=12),
#                         bgcolor="rgba(0,20,40,0.4)",
#                     ),
#                     width=1000,
#                     height=800,
#                     plot_bgcolor="rgba(0,20,40,0.4)",
#                     uirevision="constant"
#                 )
                
#             return (
#                 f"Total CO2 Emissions: {total_co2_emissions:.2f} g",
#                 dcc.Markdown(f"(Increase Rate: {co2_increase_rate:.2f} g/s)"),
#                 html.Div([
#                     html.Span("CO2 Emissions for Asprela -> Campo Alegre:", style={"font-weight": "bold"}),
#                     html.Div([
#                         html.Span("Bike:", style={"background-color": "green", "color": "white", "padding": "2px 5px", "border-radius": "3px"}),
#                         f" {num_agents_a2c_bike} Agents",
#                     ], style={"margin-left": "10px", "margin-bottom": "5px"}),
#                     html.Div([
#                         html.Span("Car:", style={"background-color": "blue", "color": "white", "padding": "2px 5px", "border-radius": "3px"}),
#                         f" {num_agents_a2c_car} Agents ({a2c_car_co2_emissions:.2f} g)",
#                     ], style={"margin-left": "10px", "margin-bottom": "5px"}),
#                     html.Div([
#                         html.Span("Public Transport:", style={"background-color": "red", "color": "white", "padding": "2px 5px", "border-radius": "3px"}),
#                         f" {num_agents_a2c_pt} Agents ({a2c_pt_co2_emissions:.2f} g)",
#                     ], style={"margin-left": "10px", "margin-bottom": "5px"}),
#                 ]),
#                 html.Div([
#                     html.Span("CO2 Emissions for Campo Alegre -> Asprela:", style={"font-weight": "bold"}),
#                     html.Div([
#                         html.Span("Bike:", style={"background-color": "yellow", "color": "black", "padding": "2px 5px", "border-radius": "3px"}),
#                         f" {num_agents_c2a_bike} Agents",
#                     ], style={"margin-left": "10px", "margin-bottom": "5px"}),
#                     html.Div([
#                         html.Span("Car:", style={"background-color": "magenta", "color": "white", "padding": "2px 5px", "border-radius": "3px"}),
#                         f" {num_agents_c2a_car} Agents ({c2a_car_co2_emissions:.2f} g)",
#                     ], style={"margin-left": "10px", "margin-bottom": "5px"}),
#                     html.Div([
#                         html.Span("Public Transport:", style={"background-color": "cyan", "color": "black", "padding": "2px 5px", "border-radius": "3px"}),
#                         f" {num_agents_c2a_pt} Agents ({c2a_pt_co2_emissions:.2f} g)",
#                     ], style={"margin-left": "10px", "margin-bottom": "5px"}),
#                 ]),
#                 completion_rate,
#                 fig,
#                 cumulative_plot,
#                 episode_plot,
#                 -1  # Stop updates
#             )


#         else:
#             # If finished, return the same plots and metrics, but stop updates
#             return (
#                 dash.no_update,  # kpi-co2-total
#                 dash.no_update,  # kpi-co2-increase
#                 dash.no_update,  # kpi-co2-a2c
#                 dash.no_update,  # kpi-co2-c2a
#                 dash.no_update,  # kpi-completion
#                 dash.no_update,  # route-graph
#                 dash.no_update,  # metric-plot
#                 dash.no_update,  
#                 -1  # Stop updates
#             )
# # Run the App
# if __name__ == "__main__":
#     app.run_server(debug=True)