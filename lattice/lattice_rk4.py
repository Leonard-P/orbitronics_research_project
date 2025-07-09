from typing import Callable, List, Optional, Union
import numpy as np
from scipy import sparse
from tqdm import trange

Matrix = Union[np.ndarray, sparse.spmatrix]


def time_evolution_derivative(
    t: float,
    density_matrix: np.ndarray,
    H_hop: Matrix,
    H_onsite: Matrix,
    field_amplitude: Callable[[float], float],
    initial_density: Optional[np.ndarray] = None,
    decay_time: float = float("inf"),
) -> np.ndarray:
    """
    Calculate the time derivative of the density matrix.

    ∂ρ/∂t = -i[H, ρ] + (ρ₀ - ρ) / τ

    Parameters:
      t: Current time.
      density_matrix: Density matrix ρ(t) (dense).
      H_hop: Hopping Hamiltonian (dense or sparse).
      H_onsite: On-site potential Hamiltonian (dense or sparse).
      field_amplitude: Field amplitude (function of time).
      initial_density: Initial density matrix ρ₀ (dense), used for decay term.
      decay_time: Decay time constant τ. Default is infinity (no decay).

    Returns:
      The time derivative of the density matrix (dense ndarray).
    """

    # Calculate the time-dependent Hamiltonian
    H = H_hop + field_amplitude(t) * H_onsite

    # Calculate [H, ρ]
    if sparse.issparse(H):
        commutator = H.dot(density_matrix) - sparse.csr_matrix.dot(density_matrix, H)
    else:
        commutator = H @ density_matrix - density_matrix @ H

    if decay_time != float("inf") and initial_density is not None:
        decay_term = (initial_density - density_matrix) / decay_time
        return -1j * commutator + decay_term

    return -1j * commutator


def rk4_step(
    t: float,
    density_matrix: np.ndarray,
    H_hop: np.ndarray,
    H_onsite: np.ndarray,
    field_amplitude: Callable[[float], float],
    dt: float,
    initial_density: Optional[np.ndarray] = None,
    decay_time: float = float("inf"),
) -> np.ndarray:
    """
    Perform a single Runge-Kutta 4th order step to evolve the density matrix.

    Parameters:
      t: Current time.
      density_matrix: Density matrix ρ(t) (dense).
      H_hop: Hopping Hamiltonian (dense or sparse).
      H_onsite: On-site potential Hamiltonian (dense or sparse).
      field_amplitude: Field amplitude as a callable.
      dt: Time step.
      initial_density: Initial density matrix for the decay term.
      decay_time: Decay time constant τ.

    Returns:
      The density matrix after one RK4 step (dense ndarray).
    """
    k1 = dt * time_evolution_derivative(t, density_matrix, H_hop, H_onsite, field_amplitude, initial_density, decay_time)
    k2 = dt * time_evolution_derivative(t + 0.5 * dt, density_matrix + 0.5 * k1, H_hop, H_onsite, field_amplitude, initial_density, decay_time)
    k3 = dt * time_evolution_derivative(t + 0.5 * dt, density_matrix + 0.5 * k2, H_hop, H_onsite, field_amplitude, initial_density, decay_time)
    k4 = dt * time_evolution_derivative(t + dt, density_matrix + k3, H_hop, H_onsite, field_amplitude, initial_density, decay_time)

    D_next = density_matrix + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Ensure hermiticity
    D_next = (D_next + D_next.T.conj()) / 2.0

    return D_next


def evolve_density_matrix_rk4(
    H_hop: np.ndarray,
    H_onsite: np.ndarray,
    initial_density: np.ndarray,
    field_amplitude: Callable[[float], float],
    dt: float,
    total_time: float,
    decay_time: float = float("inf"),
    sample_every: int = 1,
    first_snapshot_step: int = 0,
    use_sparse: bool = True,
) -> List[np.ndarray]:
    """
    Evolve the density matrix using the RK4 method.

    Parameters:
      H_hop: Hopping Hamiltonian (dense ndarray). Will be converted to sparse if use_sparse is True.
      H_onsite: On-site potential Hamiltonian (dense ndarray). Converted similarly.
      initial_density: Initial density matrix ρ₀ (dense ndarray).
      field_amplitude: Field amplitude as a callable.
      dt: Time step.
      total_time: Total simulation time.
      decay_time: Decay time constant τ.
      sample_every: Store every n-th step (if 0, only the final state is stored).
      use_sparse: If True, convert Hamiltonians to sparse matrices for multiplication efficiency.

    Returns:
      A list of density matrices (dense ndarrays) at each stored time step.
    """
    n_steps = int(total_time / dt)
    result: List[np.ndarray] = []

    if use_sparse:
        H_hop = sparse.csr_matrix(H_hop)
        H_onsite = sparse.csr_matrix(H_onsite)

    D_t = initial_density.copy()

    for step in trange(n_steps):
        if sample_every and (step >= first_snapshot_step) and (step % sample_every == 0):
            result.append(D_t.copy())

        t = step * dt
        D_t = rk4_step(t, D_t, H_hop, H_onsite, field_amplitude, dt, initial_density, decay_time)

    result.append(D_t.copy())
    return result
