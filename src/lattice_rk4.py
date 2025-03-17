import warnings
import numpy as np
from scipy import sparse
from typing import Callable, List, Optional, Union
from tqdm import trange


def time_evolution_derivative(
    t: float,
    density_matrix: np.ndarray,
    H_hop: np.ndarray,
    H_onsite: np.ndarray,
    field_amplitude: Callable[[float], float],
    initial_density: Optional[np.ndarray] = None,
    decay_time: float = float("inf"),
) -> np.ndarray:
    """
    Calculate the time derivative of the density matrix.

    Parameters:
    t (float): Time.
    density_matrix (np.ndarray): Density matrix at time t.
    H_hop (np.ndarray): Hopping Hamiltonian.
    H_onsite (np.ndarray): On-site potential Hamiltonian.
    field_amplitude (Union[float, Callable]): Field amplitude (scalar or function of time).
    initial_density (np.ndarray, optional): Initial density matrix for decay term.
    decay_time (float): Decay time constant. Default is infinity (no decay).

    Returns:
    np.ndarray: The time derivative of the density matrix. ∂ρ/∂t = -i[H, ρ] + (ρ₀ - ρ) / τ
    """

    # Calculate the time-dependent Hamiltonian
    H = H_hop + field_amplitude(t) * H_onsite

    # Calculate the commutator of H_t and D
    commutator = H @ density_matrix - density_matrix @ H

    # Calculate the decay term
    if decay_time != float("inf") and initial_density is not None:
        decay_term = (initial_density - density_matrix) / decay_time
        return -1j * commutator + decay_term

    return -1j * commutator


def rk4_step(
    t: float,
    density_matrix: np.ndarray,
    H_hop: np.ndarray,
    H_onsite: np.ndarray,
    field_amplitude: Union[float, Callable[[float], float]],
    dt: float,
    initial_density: Optional[np.ndarray] = None,
    decay_time: float = float("inf"),
) -> np.ndarray:
    """
    Perform a single Runge-Kutta 4th order step to evolve the density matrix.

    Parameters:
    t (float): Current time.
    density_matrix (np.ndarray): Density matrix at time t.
    H_hop (np.ndarray): Hopping Hamiltonian.
    H_onsite (np.ndarray): On-site potential Hamiltonian.
    field_amplitude (Union[float, Callable]): Field amplitude (scalar or function of time).
    dt (float): Time step.
    initial_density (np.ndarray, optional): Initial density matrix for decay term.
    decay_time (float): Decay time constant. Default is infinity (no decay).

    Returns:
    np.ndarray: The density matrix after one RK4 step.
    """
    k1 = dt * time_evolution_derivative(t, density_matrix, H_hop, H_onsite, field_amplitude, initial_density, decay_time)

    k2 = dt * time_evolution_derivative(t + 0.5 * dt, density_matrix + 0.5 * k1, H_hop, H_onsite, field_amplitude, initial_density, decay_time)

    k3 = dt * time_evolution_derivative(t + 0.5 * dt, density_matrix + 0.5 * k2, H_hop, H_onsite, field_amplitude, initial_density, decay_time)

    k4 = dt * time_evolution_derivative(t + dt, density_matrix + k3, H_hop, H_onsite, field_amplitude, initial_density, decay_time)

    D_next = density_matrix + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Ensure hermiticity (numerical error can cause slight deviations)
    D_next = (D_next + D_next.T.conj()) / 2.0

    return D_next


def evolve_density_matrix_rk4(
    H_hop: np.ndarray,
    H_onsite: np.ndarray,
    initial_density: np.ndarray,
    field_amplitude: Union[float, Callable[[float], float]],
    dt: float,
    total_time: float,
    decay_time: float = float("inf"),
    sample_every: int = 1,
    use_sparse: bool = True,
) -> List[np.ndarray]:
    """
    Evolve the density matrix using RK4 method.

    Parameters:
    H_hop (np.ndarray): Hopping Hamiltonian.
    H_onsite (np.ndarray): On-site potential Hamiltonian.
    initial_density (np.ndarray): Initial density matrix.
    field_amplitude (Union[float, Callable]): Electric field amplitude or function of time.
    dt (float): Time step.
    total_time (float): Total simulation time.
    decay_time (float): Decay time constant. Default is infinity (no decay).
    sample_every (int): Store every n-th step. Default is 1 (store all steps). 0 will store only the final state.
    use_sparse (bool): Use sparse matrices for computation. Default is True.

    Returns:
    List[np.ndarray]: List of density matrices at each stored time step.
    """
    n_steps = int(total_time / dt)
    result = []

    if use_sparse:
        H_hop = sparse.csr_matrix(H_hop)
        H_onsite = sparse.csr_matrix(H_onsite)

    D_t = initial_density.copy()

    for step in trange(n_steps):
        if sample_every and (step % sample_every == 0):
            result.append(D_t.copy())

        t = step * dt
        D_t = rk4_step(t, D_t, H_hop, H_onsite, field_amplitude, dt, initial_density, decay_time)

    result.append(D_t.copy())
    return result
