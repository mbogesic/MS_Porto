def calculate_transport_increase( simulation_data: dict, previous_percentage: float = None) -> dict:
    # Extract data from the simulation results
    bike_users = simulation_data.get("bike_users", 0)
    scooter_users = simulation_data.get("scooter_users", 0)
    public_transport_users = simulation_data.get("public_transport_users", 0)
    total_commuters = simulation_data.get("total_commuters", 0)   
    if total_commuters <= 0:
        raise ValueError("Total commuters must be greater than zero.")   
    # Calculate current percentage
    current_percentage = ((bike_users + scooter_users + public_transport_users) / total_commuters) * 100  
    result = {"current_percentage": current_percentage}   
    # Calculate change in percentage if previous_percentage is provided
    if previous_percentage is not None:
        change = current_percentage - previous_percentage
        result["percentage_change"] = change   
    return result
# Example usage after a simulation
#simulation_results = {
#    "bike_users": 200,
#    "scooter_users": 150,
#    "public_transport_users": 400,
#    "total_commuters": 1000 }
#data_now = calculate_transport_increase(simulation_data=simulation_results, previous_percentage=65.0)
#print(data_now)

def calculate_traffic_and_emissions( baseline_traffic: int, current_traffic: int, baseline_co2: float, current_co2: float) -> dict:
    # Calculate traffic volume reduction percentage
    if baseline_traffic <= 0:
        raise ValueError("Baseline traffic volume must be greater than zero.")   
    traffic_reduction = ((baseline_traffic - current_traffic) / baseline_traffic) * 100
    # Calculate CO2 emissions change
    if current_co2 < 0:
        # Negative current CO2 indicates an increase in emissions
        co2_change = baseline_co2 - (-current_co2)
    else:
        # Positive or zero current CO2 indicates a decrease or no change
        co2_change = baseline_co2 - current_co2
    # Return results
    return {
        "traffic_reduction_percentage": traffic_reduction,
        "net_co2_change": co2_change
    }
# Example usage
#results = calculate_traffic_and_emissions(
#    baseline_traffic=10000,
#    current_traffic=8000,
#    baseline_co2=150.0,
#    current_co2=120.0
#)
#print(results)

def calculate_subsidy_per_user(total_subsidies: float, public_transport_users: int) -> float:
    if public_transport_users <= 0:
        raise ValueError("Number of public transport users must be greater than zero.")   
    # Calculate average subsidy per user
    average_subsidy = total_subsidies / public_transport_users
    return average_subsidy
# Example usage
#average_subsidy = calculate_subsidy_per_user(
#    total_subsidies=500000.0,
#    public_transport_users=25000
#)
#print(f"Average subsidy per user: ${average_subsidy:.2f}")

def calculate_traffic_change(baseline_foot_traffic: int, current_foot_traffic: int, baseline_bike_traffic: int, current_bike_traffic: int) -> dict:
    # Validate inputs
    if baseline_foot_traffic <= 0:
        raise ValueError("Baseline foot traffic must be greater than zero.")
    if baseline_bike_traffic <= 0:
        raise ValueError("Baseline bike traffic must be greater than zero.")   
    # Calculate percentage changes
    foot_traffic_change = ((current_foot_traffic - baseline_foot_traffic) / baseline_foot_traffic) * 100
    bike_traffic_change = ((current_bike_traffic - baseline_bike_traffic) / baseline_bike_traffic) * 100
    return {
        "foot_traffic_change_percentage": foot_traffic_change,
        "bike_traffic_change_percentage": bike_traffic_change
    }
# Example usage
#traffic_changes = calculate_traffic_change(
#    baseline_foot_traffic=1000,
#    current_foot_traffic=1200,
#    baseline_bike_traffic=500,
#    current_bike_traffic=700
#)
#print(traffic_changes)

def calculate_parking_reduction(baseline_parked_cars: int, current_parked_cars: int) -> float:  
    if baseline_parked_cars <= 0:
        raise ValueError("Baseline parked cars must be greater than zero.")
    # Calculate percentage reduction in parked cars
    reduction_percentage = ((baseline_parked_cars - current_parked_cars) / baseline_parked_cars) * 100 
    return reduction_percentage
# Example usage
#parking_reduction = calculate_parking_reduction(
#    baseline_parked_cars=2000,
#    current_parked_cars=1500)
#print(f"Reduction in parked cars: {parking_reduction:.2f}%")

def calculate_car_trip_reduction(
    baseline_car_trips: int,
    current_car_trips: int
) -> float:
    if baseline_car_trips <= 0:
        raise ValueError("Baseline car trips must be greater than zero.") 
    # Calculate percentage reduction in car trips
    reduction_percentage = ((baseline_car_trips - current_car_trips) / baseline_car_trips) * 100
    return reduction_percentage

# Example usage
#car_trip_reduction = calculate_car_trip_reduction(
#    baseline_car_trips=10000,
#    current_car_trips=8500)
#print(f"Reduction in car trips to the city center: {car_trip_reduction:.2f}%")

def calculate_peak_parking_reduction(baseline_peak_parking: int, current_peak_parking: int) -> float:
    if baseline_peak_parking <= 0:
        raise ValueError("Baseline peak parking demand must be greater than zero.")  
    # Calculate percentage reduction in peak parking demand
    reduction_percentage = ((baseline_peak_parking - current_peak_parking) / baseline_peak_parking) * 100   
    return reduction_percentage

# Example usage
#peak_parking_reduction = calculate_peak_parking_reduction(
#    baseline_peak_parking=500,
#    current_peak_parking=400)
#print(f"Reduction in peak parking demand: {peak_parking_reduction:.2f}%")

def calculate_revenue_growth(baseline_revenue: float, current_revenue: float) -> float:
    if baseline_revenue <= 0:
        raise ValueError("Baseline revenue must be greater than zero.")  
    # Calculate revenue growth percentage
    revenue_growth_percentage = ((current_revenue - baseline_revenue) / baseline_revenue) * 100   
    return revenue_growth_percentage
# Example usage
#revenue_growth = calculate_revenue_growth(
#    baseline_revenue=100000.0,
#    current_revenue=120000.0)
#print(f"Growth in revenue from dynamic pricing: {revenue_growth:.2f}%")
