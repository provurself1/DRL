import numpy as np
import random
from itertools import permutations

# Defining hyperparameters
m = 5  # number of cities
t = 24  # number of hours
d = 7  # number of days
C = 5  # Per hour fuel and other costs
R = 9  # base revenue from a passenger
surge_factor = 2  # Surge pricing factor for peak hours

class CabDriver:
    def __init__(self, driver_id):
        """Initialize state and define action space and state space"""
        self.driver_id = driver_id
        self.action_space = list(permutations([i for i in range(m)], 2)) + [(0, 0)]  # All permutations + no action
        self.state_space = [[x, y, z] for x in range(m) for y in range(t) for z in range(d)]
        self.state_init = random.choice(self.state_space)
        self.reset()

    def reset(self):
        """Reset the driver's state"""
        self.current_state = self.state_init
        return self.current_state

    def dynamic_reward(self, hour):
        """Calculate dynamic reward based on the hour of the day"""
        if 7 <= hour < 10 or 17 <= hour < 20:  # Peak hours
            return R * surge_factor
        else:
            return R

    def step(self, action):
        """Take an action and return the new state and reward"""
        current_city, current_hour, current_day = self.current_state

        # Action is a tuple (pickup_city, dropoff_city)
        pickup_city, dropoff_city = action

        # Reward is based on whether the agent makes a successful pickup and dropoff
        reward = 0
        if pickup_city != dropoff_city:  # Successful pickup/dropoff
            reward = self.dynamic_reward(current_hour)

        # Simulate state transition:
        # Assume the agent can travel to the dropoff city and update the state
        new_hour = (current_hour + random.randint(1, 2)) % t  # Randomly increments hour (travel time)
        new_day = (current_day + (current_hour + 1) // t) % d  # Increment day if needed
        new_state = (dropoff_city, new_hour, new_day)

        # Update the current state to the new state for the next step
        self.current_state = new_state
        return new_state, reward
