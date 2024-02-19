# Kalman Filter
Install requirements:
```
pip3 install -r requirements.txt
```
running the following
```
python3 KalmanFilter.py
```
should produce a `state_distributions.txt`, some `uncertainty_ellipse_t{timestep}.png` files, and a `measurement_update.txt`, `error_distance_bar_plot.png`, `faulty_sensor_filter_error.txt` and `state_prediction.txt`.
## state distributions
The `state_distributions.txt` file contains the covariance matrix and the mean at each timestep for timestep=1 to timestep=5. 

## the uncertainty ellipses
The .png files titled `uncertainty_ellipse_t{timestep}.png` for timestp=1 to timestep=5 illustrate what the uncertainty ellipse is at each time step

## the measurement update
The `measurement_update.txt` contains the parameters(mu, sigma) at t=5 prior to the measurement update and the parameters after the measurement update.

## Sampling the true position
```
python3 Sensor.py
```
running the above will call the `calculate_error` method of class `Sensor` in `Sensor.py` and generate a `simulation_measurement.txt`. This method calls the `simulate` method of the `Simulator` class to simulate the true position, velocity, and acceleration of the drone for t=1 to t=5 and sample the true position at t=5 with a variance equal to the variance of the gps sensor. The `calculate_error` method then compares the sampled sensor measurement wit the true measurement to calculate a percent error.

## Error from simulated true position
The `error_distance_bar_plot.txt` illustrates what the expected distance from the simulated true position is for each sensor fail probability at timestep t=20. The predicted position, simulated true position and the error distance for each timestep and each sensor fail probability are documented in `faulty_sensor_filter_error.txt`.

## Mean estimate for state at time t given motor command
`state_prediction.txt` shows what mu_t is given the motor command and random acceleration due to wind in the previous timestep.
