from mesa import Agent

class TransportAgent(Agent):
    def __init__(self, unique_id, model, age_group, transport_info):
        super().__init__(unique_id, model)
        self.age_group = age_group
        self.transport_mode = transport_info[0]  # Extract the transport mode
        self.network = transport_info[1]  # Extract the network corresponding to the mode
        self.fuel_type = self.set_fuel_type()
        self.energy_consumption_per_km = self.set_energy_consumption()
        self.average_speed = self.set_average_speed()
        self.co2_emissions_per_km = self.set_co2_emissions()
    
    def set_fuel_type(self):
        # Set fuel type based on age group preferences or external model inputs
        pass
    
    def set_energy_consumption(self):
        # Energy consumption per km
        pass

    def set_average_speed(self):
        # Average speed depending on transport mode
        transport_speeds = {
            "walk": 5,  # km/h
            "bike": 15,  # km/h
            "car_diesel": 60,  # km/h
            "car_petrol": 60,
            "car_electric": 60,
            "public_transport": 30,  # km/h
        }
        return transport_speeds.get(self.transport_mode, 0)

    def set_co2_emissions(self):
        # CO₂ emissions per km based on fuel type and transport mode
        emissions = {
            "car_petrol": 120,  # g CO2/km
            "car_diesel": 100,
            "car_electric": 0,  # Assuming zero emissions for electric
            "bike": 0,
            "public_transport": 30,  # Assuming a value for public transport
            "walk": 0,
        }
        return emissions.get(self.transport_mode, 0)

    def step(self):
        # Define agent behavior per step, such as choosing routes or responding to CO₂ credits
        # Use self.network to determine routes or other agent behavior
        pass
