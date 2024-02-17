import numpy as np
from plot import *

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
    
    def measurement_update_mu(self, K, mu_prior, sigma_prior, obs):
        diff = obs - self.C@mu_prior
        gain = K @ diff
        mu_post = mu_prior + gain
        tmp = K @ self.C
        sigma_post = (np.identity(n=np.shape(tmp)[0]) - tmp)@sigma_prior
        return mu_post, sigma_post
    
    def run(self, mu_prev, sigma_prev, u_curr, z_curr):
        pass


    

if __name__ == "__main__":
    B = np.array(
        [(0.0, 0.0),
        (0.0, 0.0),
        ]
    )
    u = np.array([[0.0], 
                  [0.0]])


    delta_t = 1.0
    variance_ddotx_t0 = 1.0
    variance_x_t0 = 0
    variance_dotx_t0 = 0
    sigma_t0 = np.array([
        (0.0, 0.0),
        (0.0, 0.0)
    ])
    A = np.array(
        [(1.0, delta_t),
        (0.0, 1.0),
        ]
    )
    R = np.array([
        (delta_t**4/4.0, delta_t**3/2.0),
        (delta_t**3/2.0, delta_t**2) 
    ])
    C = np.array([[1.0, 0.0]])
    Q = np.array([8.0])
    mu_t0 = np.array([[0],
                    [0]])
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
        mu_post, sigma_post = filter.measurement_update_mu(K=K, mu_prior=means, sigma_prior=cov, obs=np.array([10]))
        file.write(f"sigma prior t{i}:\n{cov}")
        file.write("\n\n")
        file.write(f"mu prior t{i}:\n{means}")
        file.write("\n\n")
        file.write(f"sigma posterior t{i}:\n{sigma_post}")
        file.write("\n\n")
        file.write(f"mu posterior t{i}:\n{mu_post}")
            