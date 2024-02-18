# Kalman Filter
running the following
```
python3 KalmanFilter.py
```
should produce a `state_distributions.txt`, some `uncertainty_ellipse_t{timestep}.png` files, and a `measurement_update.txt`.
## state distributions
The `state_distributions.txt` file contains the covariance matrix and the mean at each timestep for timestep=1 to timestep=5. 

## the uncertainty ellipses
The .png files titled `uncertainty_ellipse_t{timestep}.png` for timestp=1 to timestep=5 illustrate what the uncertainty ellipse is at each time step

## the measurement update
The `measurement_update.txt` contains the parameters(mu, sigma) at t=5 prior to the measurement update and the parameters after the measurement update.

## Sampling the true position
the `sense` method of class `Sensor` in `Sensor.py` takes the true position and the number of samples and returns samples drawn from a normal distributino with mean = the true positino and standard deviation = square root of the sensor's noise