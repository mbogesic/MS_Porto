from model import TransportModel

# Set up and run the model
model = TransportModel(num_agents=100, base_credit=10)
for i in range(100):  # Run for 100 steps
    model.step()

# Optional: save or print output data
