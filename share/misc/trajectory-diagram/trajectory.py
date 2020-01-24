import sys
import numpy as np
from scipy.integrate import ode
from pbpl import compton
from pbpl.common.units import *

def calc_electron_trajectory(n0, x0, KE, E, B, dx, cutoff):
    m0 = me
    q0 = -eplus
    rest_energy = m0*c_light**2
    total_energy = rest_energy + KE
    gamma0 = total_energy/rest_energy
    beta0 = np.sqrt(1-1/gamma0**2)
    p0 = gamma0 * m0 * c_light * beta0
    dt = dx / (c_light * beta0)
    y0 = np.concatenate((x0, n0*p0))

    def yprime(t, y):
        x = y[:3]
        p = y[3:]
        p2 = np.dot(p, p)
        energy = np.sqrt(m0**2*c_light**4 + p2*c_light**2)
        E_x = B(x)
        B_x = B(x)
        dx = p*c_light**2/energy
        dp = q0 * np.cross(dx, B_x) + q0 * E_x
        return np.concatenate((dx, dp))

    solver = ode(yprime)
    solver.set_integrator('dopri5', max_step=1e-4)
    solver.set_initial_value(y0)
    trajectory = []
    times = []
    curr_t = 0.0
    while 1:
        trajectory.append(solver.integrate(curr_t))
        times.append(curr_t)
        if cutoff(trajectory[-1], curr_t):
            break
        curr_t += dt
    trajectory = np.array(trajectory).T
    times = np.array(times)
    return trajectory
