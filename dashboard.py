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
step_time_dimension = 10.0   # s/step aka the "resolution" of the simulation, DON'T TOUCH


### TWEAK No. Agents AND No. Episodes PARAMETERS TO YOUR LIKING ###
### (1 Episode = 1 Day) ###
num_agents = 1000
episodes = 60

# Initialize the model
model = TrafficModel(
    nodes_and_edges_folder,
    num_agents,  
    step_time_dimension, 
    episodes,
    combined_nodes_file=combined_nodes_file,
    combined_edges_file=combined_edges_file
)

# # Global list to track CO2 emissions over time
# co2_emissions_over_time = []  # Running total of CO2 emissions
# co2_emissions_over_episodes = []  # Total CO2 emissions per episode

# Start the Dash App
app = dash.Dash(__name__)
app.title = "AMC Simulation Dashboard"

app.layout = html.Div([
    html.H1("AMC Simulation Dashboard"),
    dcc.Dropdown(
        id='episode-dropdown',
        options=[{'label': f"Episode {ep_id}", 'value': ep_id} for ep_id in sorted(model.episode_history.keys())],
        placeholder="Select an Episode",
        style={'width': '50%'}
    ),
    html.Div(id='episode-details', style={'marginTop': '20px'}),  # Mode Distribution Display

    dcc.Dropdown(
        id='agent-dropdown',
        placeholder="Select an Agent",
        style={'width': '50%', 'marginTop': '20px'}
    ),
    html.Div(id='agent-details', style={'marginTop': '20px'}),  # Agent Details Display

    dcc.Graph(id="metric-plot", style={"width": "100%", "height": "400px"}),
    dcc.Graph(id="episode-plot", style={"width": "100%", "height": "400px"}),
    dcc.Graph(id="mode-distribution-plot", style={"width": "100%", "height": "400px"}),
    # dcc.Graph(id="traffic-volume-reduction-plot", style={"width": "100%", "height": "400px"}),
    dcc.Graph(id="cumulative-credits-plot", style={"width": "100%", "height": "400px"}),
    dcc.Graph(id="credits-plot", style={"width": "100%", "height": "400px"}),
    dcc.Interval(
        id="interval-component",
        interval=1000,
        n_intervals=0,
        disabled=model.simulation_finished
    ),
])

@app.callback(
    Output('episode-details', 'children'),
    Input('episode-dropdown', 'value')
)
def display_episode_details(selected_episode):
    if selected_episode is None:
        return html.Div("Select an episode to see its mode distribution.",
                        style={'color': 'blue', 'fontSize': 16, 'textAlign': 'center'})

    episode_data = model.episode_history.get(selected_episode, {})
    mode_distribution = episode_data.get("mode_distribution", {})

    mode_summary = [
        html.P(f"{mode}: {count}") for mode, count in mode_distribution.items()
    ]
    return html.Div(mode_summary, style={'textAlign': 'left', 'margin': '10px'})

@app.callback(
    Output('episode-dropdown', 'options'),
    Input('interval-component', 'n_intervals')
)
def update_dropdown_options(n_intervals):
    if not model.simulation_finished:
        return []

    # Dynamically populate dropdown options with "WARMUP" for episodes below 30
    dropdown_options = [
        {
            'label': f"Episode {ep_id} {'(WARMUP)' if ep_id < 30 else ''}",
            'value': ep_id
        }
        for ep_id in model.episode_history.keys()
    ]
    print("Dropdown Options Updated:", dropdown_options)  # Debugging
    return dropdown_options

@app.callback(
    Output('agent-dropdown', 'options'),
    Input('episode-dropdown', 'value')
)
def update_agent_dropdown(selected_episode):
    if selected_episode is None:
        return []

    episode_data = model.episode_history.get(selected_episode, {})
    sorted_agents = sorted(episode_data.get("agents", {}).keys())
    agent_options = [{'label': f"Agent {agent_id}", 'value': agent_id} for agent_id in sorted_agents]
    return agent_options

@app.callback(
    Output('agent-details', 'children'),
    Input('agent-dropdown', 'value'),
    State('episode-dropdown', 'value')
)
def display_agent_details(selected_agent, selected_episode):
    if selected_agent is None or selected_episode is None:
        return html.Div("Select an episode and an agent to view details.")

    episode_data = model.episode_history.get(selected_episode, {})
    agent_data = episode_data["agents"].get(selected_agent, {})

    agent_summary = [
        html.P(f"{key}: {value}") for key, value in agent_data.items()
    ]
    return html.Div(agent_summary, style={'textAlign': 'left', 'margin': '10px'})


@app.callback(
    [Output("metric-plot", "figure"),
     Output("episode-plot", "figure"),
     Output("mode-distribution-plot", "figure"),
    #  Output("traffic-volume-reduction-plot", "figure"),
     Output("cumulative-credits-plot", "figure"),
     Output("credits-plot", "figure"),
     Output("interval-component", "disabled")],  # Stop interval when simulation is done
    [Input("interval-component", "n_intervals")]
)
def update_plots(n_intervals):
    if model.simulation_finished:
        print(f"Update Plots Called: Simulation Finished = {model.simulation_finished}")
        data = model.get_unfiltered_episode_data()
        # data = model.get_filtered_episode_data()
        # print("Data Retrieved for Plots:", data)  # Debugging
        # Adjust episode numbers to start at 30
        start_episode = 30
        if len(data["co2_emissions_per_episode"]) > 30:
            episode_numbers = list(range(len(data["co2_emissions_per_episode"])))
        else:
            episode_numbers = list(range(start_episode, start_episode + len(data["co2_emissions_per_episode"])))

        # Define bar colors: "lightgrey" for episodes < 30, "blue" for others
        bar_colors = ["lightgrey" if ep < 30 else "blue" for ep in episode_numbers]

        # Cumulative CO2 Plot
        cumulative_co2 = data["co2_emissions_per_episode"]
        mean_co2 = [sum(cumulative_co2[:i+1]) / (i+1) for i in range(len(cumulative_co2))]
        overall_mean_co2 = sum(cumulative_co2) / len(cumulative_co2) if cumulative_co2 else 0
        mean_co2_per_episode = [overall_mean_co2] * len(cumulative_co2)
        
        # Cumulative CO2 Plot
        cumulative_co2 = data["co2_emissions_over_time"]
        if len(cumulative_co2) > 30:
            # Separate into pre-reset and post-reset data
            pre_reset_co2 = cumulative_co2[:30]
            post_reset_co2 = [
                sum(data["co2_emissions_per_episode"][30:i + 1]) for i in range(30, len(data["co2_emissions_per_episode"]))
            ]
        else:
            pre_reset_co2 = cumulative_co2
            post_reset_co2 = []

        cumulative_plot = go.Figure(
            data=[
                # Pre-reset cumulative CO2 trajectory
                go.Scatter(
                    x=list(range(len(pre_reset_co2))),
                    y=pre_reset_co2,
                    mode="lines+markers",
                    name="Warmup CO2 (0-29)",
                    line=dict(color="gray")
                ),
                # Post-reset cumulative CO2 trajectory starting at zero
                go.Scatter(
                    x=list(range(len(post_reset_co2))),
                    y=post_reset_co2,
                    mode="lines+markers",
                    name="Post-Warmup CO2 (30+)",
                    line=dict(color="green")
                )
            ],
            layout=go.Layout(
                title="Cumulative CO2 Emissions Over Episodes",
                xaxis=dict(title="Episodes"),
                yaxis=dict(title="CO2 Emissions (g)"),
            )
        )


        episode_plot = go.Figure(
            data=[go.Bar(x=episode_numbers, y=data["co2_emissions_per_episode"], marker_color=bar_colors, name="CO2 Emissions Per Episode"),
                    go.Scatter(x=list(range(len(mean_co2))), y=mean_co2_per_episode, mode="lines", name="Mean CO2", line=dict(dash="dash", color="red"))],
            layout=go.Layout(
                title="CO2 Emissions Per Episode",
                xaxis=dict(title="Episodes"),
                yaxis=dict(title="CO2 Emissions (g)"),
            )
        )

        # Mode Distribution Plot
        mode_distributions = [
            model.episode_history[ep]["mode_distribution"] for ep in model.episode_history.keys()
        ]
        modes = list(mode_distributions[0].keys())
        layers = {
            mode: [distribution.get(mode, 0) for distribution in mode_distributions] for mode in modes
        }

        # Define color sets for saturation adjustment
        color_palette = {
            "Bike": "rgba(0, 128, 0, 1)",  # Green
            "PublicTransport": "rgba(0, 0, 255, 1)",  # Blue
            "Car": "rgba(255, 0, 0, 1)",  # Red
        }
        faded_palette = {
            "Bike": "rgba(0, 128, 0, 0.3)",  # Faded Green
            "PublicTransport": "rgba(0, 0, 255, 0.3)",  # Faded Blue
            "Car": "rgba(255, 0, 0, 0.3)",  # Faded Red
        }

        mode_plot = go.Figure(
            data=[
                go.Bar(
                    name=mode,
                    x=list(range(len(mode_distributions))),
                    y=counts,
                    marker_color=[
                        faded_palette[mode] if ep < 30 else color_palette[mode] for ep in range(len(mode_distributions))
                    ]
                ) for mode, counts in layers.items()
            ],
            layout=go.Layout(
                title="Mode Distribution Per Episode",
                barmode="stack",
                xaxis=dict(title="Episodes"),
                yaxis=dict(title="Count"),
            )
        )
        
        # traffic_volume_plot = go.Figure(
        #     layout=go.Layout(
        #         title="Traffic Volume Reduction Over Episodes (No Data)",
        #         xaxis=dict(title="Episodes"),
        #         yaxis=dict(title="Reduction Percentage (%)"),
        #     )
        # )
        # if "traffic_volume_per_episode" in data and data["traffic_volume_per_episode"]:
        #     traffic_volume_initial = data["traffic_volume_per_episode"][0]
        #     traffic_volume_reduction = [
        #         (traffic_volume_initial - tv) / traffic_volume_initial * 100 for tv in data["traffic_volume_per_episode"]
        #     ]
        #     traffic_volume_plot = go.Figure(
        #         data=[go.Scatter(x=episode_numbers, y=traffic_volume_reduction, mode="lines+markers", name="Traffic Volume Reduction (%)")],
        #         layout=go.Layout(
        #             title="Traffic Volume Reduction Over Episodes",
        #             xaxis=dict(title="Episodes"),
        #             yaxis=dict(title="Reduction Percentage (%)"),
        #         )
        #     )
        
        # Cumulative Credits Plot
        credits_per_episode = []
        for episode_id in range(len(model.episode_history)):
            episode_credits = sum(agent_data['credits'] for agent_data in model.episode_history[episode_id]['agents'].values())
            credits_per_episode.append(episode_credits)

        credits_plot = go.Figure(
            data=[go.Bar(x=list(range(len(credits_per_episode))), y=credits_per_episode, marker_color=bar_colors, name="Credits Per Episode")],
            layout=go.Layout(
                title="Cumulative Credits Per Episode",
                xaxis=dict(title="Episodes"),
                yaxis=dict(title="Total Credits"),
            )
        )

        agent_credits = model.get_agent_credits()
        sorted_agent_credits = sorted(agent_credits.items(), key=lambda x: x[1], reverse=True)
        sorted_agent_ids = [item[0] for item in sorted_agent_credits]
        sorted_credits = [item[1] for item in sorted_agent_credits]

        agents_plot = go.Figure(
            data=[go.Bar(x=sorted_agent_ids, y=sorted_credits, name="Agent Credits")],
            layout=go.Layout(
                title="Agent Credits",
                xaxis=dict(title="Agent IDs", categoryorder="total descending"),
                yaxis=dict(title="Credits"),
            )
        )
        mean_credit = sum(sorted_credits) / len(sorted_credits) if sorted_credits else 0
        agents_plot.add_trace(
            go.Scatter(
                x=sorted_agent_ids, 
                y=[mean_credit] * len(sorted_agent_ids), 
                mode="lines", 
                name="Mean Credit", 
                line=dict(dash="dash", color="red")
            )
        )


        return cumulative_plot, episode_plot, mode_plot, credits_plot, agents_plot, True  # Disable interval updates
    else:
        # Continue with normal updates
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, False

if __name__ == "__main__":
   
    while not model.simulation_finished:
        model.step()
    app.run_server(debug=False)
    

    
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