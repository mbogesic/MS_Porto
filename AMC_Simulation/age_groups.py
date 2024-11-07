# Dictionary to define the attributes of each age group
AGE_GROUPS = { 
    "young_adults": {
        "min_age": 18,
        "max_age": 25
    },
    "adults": {
        "min_age": 26,
        "max_age": 35
    },
    "older_adults": {
        "min_age": 36,
        "max_age": 76
    },
    "seniors": {
        "min_age": 77,
        "max_age": 100
    }
}

# Helper function to get transport mode parameters
def get_age_group_params(mode):
    return AGE_GROUPS.get(mode, {})