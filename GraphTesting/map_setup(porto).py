import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

def get_routes(city_name, start_coords, end_coords):
    # Download the road network from OSMnx
    G = ox.graph_from_place(city_name, network_type='drive')
    
    # Find the nearest nodes to the start and end coordinates
    start_nodes = [ox.nearest_nodes(G, X=lon, Y=lat) for lat, lon in start_coords]
    end_nodes = [ox.nearest_nodes(G, X=lon, Y=lat) for lat, lon in end_coords]
    
    # Calculate the shortest paths for each group
    routes = [nx.shortest_path(G, start, end, weight='length') for start, end in zip(start_nodes, end_nodes)]
    
    return G, routes

# Coordinates for Porto, Portugal (Estádio Universitário and Faculty of Engineering)
start_coords = [(41.197287, -8.611410),  # starting point 1
                (41.15735042637268, -8.628023586352771), # starting point 2
                (41.1513442550581, -8.642034923720752)]  # starting point 3

end_coords = [(41.178507, -8.598544),    # ending point 1
              (41.17535748435213, -8.600186030994282),    # endpoint 2
              (41.14900009260539, -8.586508588298637)]    # endpoint 3

city_name = "Porto, Portugal"
G, routes = get_routes(city_name, start_coords, end_coords)

# Save the graph to the file (if needed)
save_path = r'C:/path_to_save/porto_graph.graphml'
ox.save_graphml(G, save_path)

# Print routes for debugging
for i, route in enumerate(routes):
    print(f"Route {i+1}: {route}")

# Step 1: Plot the road network of Porto using OSMnx
fig, ax = plt.subplots(figsize=(12, 12))

# Plot the road network
ox.plot_graph(G, ax=ax, node_size=0, edge_linewidth=0.5, show=False)

# Step 2: Define the colors for the three routes
route_colors = ['red', 'blue', 'green']  # Different colors for each route

# Step 3: Plot the routes on top of the road network
route_lines = []
for i, route in enumerate(routes):
    # Extract the nodes from the route and get their positions
    route_edges = list(zip(route[:-1], route[1:]))  # Edge list from nodes
    
    # Get the coordinates (lat, lon) of the nodes along the route
    route_line = []
    for u, v in route_edges:
        u_lat, u_lon = G.nodes[u]['y'], G.nodes[u]['x']
        v_lat, v_lon = G.nodes[v]['y'], G.nodes[v]['x']
        route_line.append([(u_lon, u_lat), (v_lon, v_lat)])

    # Plot the route as a line on the graph
    for line in route_line:
        x, y = zip(*line)
        ax.plot(x, y, color=route_colors[i], linewidth=2)

# Step 4: Zoom out to show Estádio Universitário and Faculty of Engineering area
# Calculate the bounding box of all routes
lats = []
lons = []
for route in routes:
    for node in route:
        lat, lon = G.nodes[node]['y'], G.nodes[node]['x']
        lats.append(lat)
        lons.append(lon)

# Set the limits for the plot to zoom out more
lat_min, lat_max = min(lats) - 0.03, max(lats) + 0.002  # Increased padding for zooming out
lon_min, lon_max = min(lons) - 0.005, max(lons) + 0.002  # Increased padding for zooming out

ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)

# Show the final plot
plt.title("Zoomed-out Routes in Porto with Different Colors")
plt.show()
