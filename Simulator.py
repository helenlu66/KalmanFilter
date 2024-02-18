import numpy as np
from pprint import pprint

class Simulator:
    def __init__(self, wind_mean=0, wind_variance=1.0) -> None:
        self.wind_mean=wind_mean
        self.wind_var=wind_variance
    
    def simulate(self, init_state=np.array([[0.0], [0.0], [0.0]]), state_transition=np.array([[1.0, 1.0, 0.5],[0.0, 1.0, 1.0],[0.0, 0.0, 0.0]]), num_timestep=1):
        """generate a sequence of simulated states after the initial state"""
        state = init_state
        accelerations = np.random.normal(self.wind_mean, np.sqrt(self.wind_var), num_timestep)
        simulated_states = []
        for i in range(num_timestep):
            acceleration_ti = accelerations[i]
            state = state_transition @ state
            state[2, 0] = acceleration_ti
            simulated_states.append(state)
        
        return simulated_states

if __name__=="__main__":
    simulator = Simulator()
    simmulated_states = simulator.simulate(num_timestep=10)
    pprint(simmulated_states)
