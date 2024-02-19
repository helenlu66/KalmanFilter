import numpy as np
from Simulator import Simulator
class Sensor:
    def __init__(self, variance) -> None:
        self.variance = variance
        self.simulator = Simulator()
    
    def sense(self, true_position, fail_prob=0.0, sample_size=1):
        std = np.sqrt(self.variance)
        sample = []
        for _ in range(sample_size):
            if np.random.rand() > fail_prob:
                sample.append(np.random.normal(true_position, std, 1)[0])
            else:
                sample.append(None)
        return sample

    def calculate_error(self, num_timestep=5):
        """calculate the error of the sensor as a percentage of the true position averaged across N simulations"""
        simulated_states = self.simulator.simulate(num_timestep=num_timestep)
        ground_truth_final_state = simulated_states[-1]
        ground_truth_final_pos = ground_truth_final_state[0]
        measurement = self.sense(true_position=ground_truth_final_pos)
        with open('simulation_measurement.txt', 'w') as file:
            for i in range(num_timestep):
                file.write(f"true position at t{i + 1}: {simulated_states[i][0]}\n")
            file.write(f"sensor measurement at t{i + 1}: {measurement}\n")
            file.write(f"error: {abs((ground_truth_final_pos - measurement) / (ground_truth_final_pos + 0.001))}")

if __name__=="__main__":
    sensor = Sensor(variance=8)
    sensor.calculate_error()