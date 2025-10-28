
import numpy as np
from ..backend import CPUArray, FCPUArray

def _fourier_at_omega(signal: FCPUArray, dt: float, omega: float) -> np.complexfloating:
    N = len(signal)
    n = np.arange(N)
    t = n * dt
    phase = np.exp(-1j * omega * t)
    return np.sum(signal * phase) * dt

def ohc(orbital_current_values: FCPUArray, E_amplitude_values: FCPUArray, dt: float, omega: float, t_eq: float=0.0) -> np.complexfloating:
    """Compute the orbital Hall conductivity from the ratio of the Fourier components at the driving frequency omega.
    Parameters:
        orbital_current_values: Array of measured orbital currents.
        E_amplitude_values: Electric field amplitude values at the same time steps.
        dt: Time step size.
        omega: Driving frequency at which the conductivity is measured.
        t_eq: Equilibration time to skip initial transient data, data points with time.
    Returns:
        The orbital Hall conductivity as a complex number in units (e/2pi).
    """
    return 2 * np.pi * _fourier_at_omega(orbital_current_values, dt, omega) / _fourier_at_omega(E_amplitude_values, dt, omega)