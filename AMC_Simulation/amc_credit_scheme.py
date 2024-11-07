class AMCCreditScheme:
    def __init__(self, base_credit):
        self.base_credit = base_credit
    
    def apply_credit(self, agent):
        # Incentivize lower CO₂ options by adjusting agent preferences
        if agent.transport_mode in ["bike", "public_transport"]:
            agent.credit_score += self.base_credit
        elif agent.transport_mode in ["car_petrol", "car_diesel"]:
            agent.credit_score -= self.base_credit * 2
