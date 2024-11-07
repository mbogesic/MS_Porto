import osmnx as ox

def fetch_transportation_networks(place_name):
    # Fetch different types of networks
    networks = {
        "walk": ox.graph_from_place(place_name, network_type="walk"),
        "bike": ox.graph_from_place(place_name, network_type="bike"),
        "drive": ox.graph_from_place(place_name, network_type="drive"),
    }
    return networks
