from mesa.datacollection import DataCollector

def create_data_collector(model):
    """Creates a DataCollector for the TransportModel."""
    data_collector = DataCollector(
        agent_reporters={"CO2 Emissions": "co2_emissions"},  # Adjust based on your agent's attributes
    )
    return data_collector
