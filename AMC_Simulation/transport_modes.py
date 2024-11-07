# Dictionary to define the attributes of each transport mode
TRANSPORT_MODES = {
    "car_petrol": {
        "fuel_type": "petrol",
        "energy_consumption_per_km": 0.2,  # kWh/km
        "average_speed": 50,               # km/h
        "co2_emissions_per_km": 0.15       # kg/km
    },
    "car_electric": {
        "fuel_type": "electric",
        "energy_consumption_per_km": 0.1,
        "average_speed": 50,
        "co2_emissions_per_km": 0.05
    },
    "bike": {
        "fuel_type": "human_power",
        "energy_consumption_per_km": 0,
        "average_speed": 15,
        "co2_emissions_per_km": 0
    },
    "public_transport": {
        "fuel_type": "diesel",
        "energy_consumption_per_km": 0.15,
        "average_speed": 30,
        "co2_emissions_per_km": 0.07
    }
    # Additional modes can be added here
}

# Helper function to get transport mode parameters
def get_transport_mode_params(mode):
    return TRANSPORT_MODES.get(mode, {})
