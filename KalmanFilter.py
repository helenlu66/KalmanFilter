import numpy as np
from plot import *
from Constants import *
from Simulator import Simulator
from Sensor import Sensor
class KalmanFilter:
    def __init__(self, A, B, R, C=None, Q=None) -> None:
        self.A:np.array = A
        self.B:np.array = B
        self.C:np.array = C
        self.R:np.array = R
        self.Q:np.array = Q
    
    def mu_prediction(self, mu_prev:np.array, u_curr:np.array) -> np.array:
        return self.A@mu_prev + self.B@u_curr

    def calculate_covariance(self, sigma_prev:np.array):
        return self.A@sigma_prev@self.A.T + self.R
    
    def kalman_gain(self, sigma_prior:np.array):
        tmp:np.array = self.C@sigma_prior@self.C.T+self.Q
        if np.shape(tmp) == (1, ):
            tmp_inv = 1.0 / tmp
        else:
            tmp_inv = np.linalg.inv(tmp)
        if np.shape(tmp_inv) == (1,):
            K = sigma_prior@self.C.T*tmp_inv
            return K
        else:
            K= sigma_prior@self.C.T@tmp_inv
            return K
    
    def measurement_update(self, K, mu_prior, sigma_prior, obs):
        diff = obs - self.C@mu_prior
        gain = K @ diff
        mu_post = mu_prior + gain
        tmp = K @ self.C
        sigma_post = (np.identity(n=np.shape(tmp)[0]) - tmp)@sigma_prior
        return mu_post, sigma_post
    
    def run(self, mu_prev, sigma_prev, u_curr, z_curr=None):
        mu_curr_prior = self.mu_prediction(mu_prev=mu_prev, u_curr=u_curr)
        sigma_curr_prior = self.calculate_covariance(sigma_prev=sigma_prev)
        K = self.kalman_gain(sigma_prior=sigma_curr_prior)
        if z_curr:#if there's measurement, return the updated prediction
            mu_curr_posterior, sigma_curr_posterior = self.measurement_update(K=K, mu_prior=mu_curr_prior, sigma_prior=sigma_curr_prior, obs=z_curr)
            return mu_curr_posterior, sigma_curr_posterior
        else:#if there's no measurement, return the prediction prior to measurement
            return mu_curr_prior, sigma_curr_prior

    

if __name__ == "__main__":
    filter = KalmanFilter(A=A, B=B, R=R, C=C, Q=Q)
    results = []
    mu_prev = mu_t0
    sigma_prev = sigma_t0
    with open('state_distributions.txt', 'w') as file:
        for i in range(1, 6):
            means = filter.mu_prediction(mu_prev=mu_prev, u_curr=u)
            cov = filter.calculate_covariance(sigma_prev=sigma_prev)
            file.write(f"covariance_t{i}:\n{cov}")
            file.write('\n')
            file.write(f"mu_x_t{i}: {means[0]}")
            file.write('\n')
            file.write(f"variance_x_t{i}: {cov[0, 0]}")
            file.write('\n')
            file.write(f"mu_dotx_t{i}: {means[1]}")
            file.write('\n')
            file.write(f"variance_dotx_t{i}: {cov[1, 1]}")
            
            plot_ellipse(mean=means, cov=cov, filename=f"uncertainty_ellipse_t{i}")
            mu_prev = means
            sigma_prev = cov
            file.write('\n\n')
    print(f"sigma prior ", cov)
    print(f"mu prior ", means)
    
    with open('measurement_update.txt', 'w') as file:
        
        K = filter.kalman_gain(sigma_prior=cov)
        mu_post, sigma_post = filter.measurement_update(K=K, mu_prior=means, sigma_prior=cov, obs=np.array([10]))
        file.write(f"sigma prior t{i}:\n{cov}")
        file.write("\n\n")
        file.write(f"mu prior t{i}:\n{means}")
        file.write("\n\n")
        file.write(f"sigma posterior t{i}:\n{sigma_post}")
        file.write("\n\n")
        file.write(f"mu posterior t{i}:\n{mu_post}")
    
    simulator = Simulator()
    sensor = Sensor(variance=8)
    sensor_fail_prob_errors = []
    with open('faulty_sensor_filter_error.txt', 'w') as file:
        for fail_prob in [0.1, 0.5, 0.9]:
            file.write(f"==================sensor fail probability {fail_prob}: ================ \n")
            errors = []
            for i in range(1, N + 1):
                file.write(f"------------simulation {i} -------------\n")
                mu_prev, sigma_prev, u_curr = mu_t0, sigma_t0, u
                simulated_states = simulator.simulate(num_timestep=20)
                for i in range(20):
                    file.write(f"timestep {i + 1}: ----------- \n")
                    simulated_pos = simulated_states[i][0]
                    sensed_pos = sensor.sense(true_position=simulated_pos, fail_prob=fail_prob)[0]
                    mu_curr, sigma_curr = filter.run(mu_prev=mu_prev, sigma_prev=sigma_prev, u_curr=u_curr, z_curr=sensed_pos)
                    mu_prev, sigma_prev = mu_curr, sigma_curr
                    file.write(f"sensed position: {sensed_pos}\n")
                    file.write(f"true position at t{i + 1}: {simulated_pos}\n")
                    file.write(f"final predicted mu at t{i + 1}: \n{mu_curr}\n")
                    error = abs(mu_curr[0][0] - simulated_pos[0])
                    file.write(f"error distance from true position: {error}\n")
                errors.append(error)
            avg_error = np.average(errors)
            sensor_fail_prob_errors.append(avg_error)
            file.write(f"average error with sensor fail probability {fail_prob} at t{i + 1}: {avg_error}")
    plot_errors(errors=sensor_fail_prob_errors, categories=["0.1", "0.5", "0.9"])