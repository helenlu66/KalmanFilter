import numpy as np

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
B = np.array(
        [(0.0, 0.0),
        (0.0, 0.0),
        ]
    )
B_motor_commands = np.array(
    [
        [0.5 * pow(delta_t, 2), 0.5 * pow(delta_t, 2)],
        [delta_t, delta_t]
    ]
)
u = np.array([[0.0], 
                [0.0]])
u_motor_commands = np.array(
    [[1.0],
     [np.random.normal(0, 1.0)]]
)
R = np.array([
    (delta_t**4/4.0, delta_t**3/2.0),
    (delta_t**3/2.0, delta_t**2) 
])
C = np.array([[1.0, 0.0]])
Q = np.array([8.0])
mu_t0 = np.array([[0],
                [0]])
mu_t_minus_1 = np.array(
    [[5.0],
     [1.0]]
)
N = 30