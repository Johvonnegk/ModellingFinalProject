import numpy as np
from numba import njit

# Time setup
dt = 0.001
t_max = 3
time = np.arange(0, t_max, dt)

# Constants
L = 1.0  # effective spring length
gravity_options = [1.62, 3.71, 9.81, 24.79]  # Gravity for Moon, Mars, Earth, Jupiter

# Function responsible for stop detection
@njit
def has_stopped(displacement, velocity, threshold=0.05, window=0.05, dt=0.001):
    samples = int(window / dt)
    for i in range(len(displacement) - samples):
        if np.all(np.abs(displacement[i:i+samples]) < threshold) and \
           np.all(np.abs(velocity[i:i+samples]) < threshold):
            return i * dt
    return -1.0  # Numba-safe fallback (instead of None)

# Function responsible for calculating the derivatives of the system
@njit
def calculate_derivatives(t, y, m, c, k, g):
    y1, y2 = y
    dy1dt = y2
    dy2dt = -(c/m) * y2 - (k/m) * y1 + g
    return np.array([dy1dt, dy2dt])

# Function responsible for solving RK4 (Runge Kutta 4th Order) that accepts full dynamics parameters
@njit
def runge_kutta_4(y0, t, m, c, k, g):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        k1 = calculate_derivatives(t[i-1], y[i-1], m, c, k, g)
        k2 = calculate_derivatives(t[i-1] + dt/2, y[i-1] + dt/2 * k1, m, c, k, g)
        k3 = calculate_derivatives(t[i-1] + dt/2, y[i-1] + dt/2 * k2, m, c, k, g)
        k4 = calculate_derivatives(t[i-1] + dt, y[i-1] + dt * k3, m, c, k, g)
        y[i] = y[i-1] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y

# Runs the simulation in its entirety for a given set of parameters and returns the time it takes to stop
@njit
def simulate(m, c, k, g, angle_deg):
    theta = np.radians(angle_deg)
    y0 = L * theta
    v0 = 0.0
    y_init = np.array([y0, v0])
    result = runge_kutta_4(y_init, time, m, c, k, g)
    disp, vel = result[:, 0], result[:, 1]
    t_stop = has_stopped(disp, vel, dt=dt)
    return t_stop if t_stop > 0 else None
