# Getting state distributions
running the following
```
python3 KalmanFilter.py
```
should produce a `state_distributions.txt` file that contains the covariance matrix and the mean at each timestep for timestep=1 to timestep=5. 

# Getting the uncertainty ellipses
```
python3 KalmanFilter.py
```
should also produce 5 .png files titled `uncertainty_ellipse_t{timestep}.png` for timestp=1 to timestep=5