from typing import Callable, List, Optional, Union
from tqdm import trange

import cupy as cp
import cupy.sparse as sparse  # Use cupy.sparse


Matrix = Union[cp.ndarray, sparse.spmatrix]

DEFAULT_PRECISION = "double"  # Options: 'single', 'double'


def get_cupy_dtypes(precision: str = DEFAULT_PRECISION):
    """Gets the CuPy float and complex dtypes based on precision string."""
    if precision.lower() == "single":
        return cp.float32, cp.complex64
    elif precision.lower() == "double":
        return cp.float64, cp.complex128
    else:
        raise ValueError("precision must be 'single' or 'double'")


def time_evolution_derivative(
    t: float,
    density_matrix: cp.ndarray,
    H_hop: Matrix,
    H_onsite: Matrix,
    field_amplitude: Callable[[float], float],
    initial_density: Optional[cp.ndarray] = None,
    decay_time: float = float("inf"),
    float_dtype=cp.float64,
    complex_dtype=cp.complex128,
) -> cp.ndarray:
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
    H = H_hop + float_dtype(field_amplitude(t)) * H_onsite

    # Calculate [H, ρ]
    if sparse.issparse(H):
        commutator = H.dot(density_matrix) - sparse.csr_matrix.dot(density_matrix, H)
    else:
        # "explicitly" use np.matmul for dense arrays
        commutator = H @ density_matrix - density_matrix @ H

    if decay_time != float("inf") and initial_density is not None:
        decay_term = (initial_density - density_matrix) / float_dtype(decay_time)
        return -1j * commutator + decay_term  # Possible TODO: ensure decay_term is complex with .astype(complex_dtype, copy=False)

    return -1j * commutator


def rk4_step(
    t: float,
    density_matrix: cp.ndarray,
    H_hop: cp.ndarray,
    H_onsite: cp.ndarray,
    field_amplitude: Union[float, Callable[[float], float]],
    dt: float,
    initial_density: Optional[cp.ndarray] = None,
    decay_time: float = float("inf"),
    float_dtype=cp.float64,
    complex_dtype=cp.complex128,
) -> cp.ndarray:
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
    dt = float_dtype(dt)  # Ensure dt is compatible float type

    k1 = dt * time_evolution_derivative(t, density_matrix, H_hop, H_onsite, field_amplitude, initial_density, decay_time, float_dtype, complex_dtype)
    k2 = dt * time_evolution_derivative(t + 0.5 * dt, density_matrix + 0.5 * k1, H_hop, H_onsite, field_amplitude, initial_density, decay_time, float_dtype, complex_dtype)
    k3 = dt * time_evolution_derivative(t + 0.5 * dt, density_matrix + 0.5 * k2, H_hop, H_onsite, field_amplitude, initial_density, decay_time, float_dtype, complex_dtype)
    k4 = dt * time_evolution_derivative(t + dt, density_matrix + k3, H_hop, H_onsite, field_amplitude, initial_density, decay_time, float_dtype, complex_dtype)

    D_next = density_matrix + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Ensure hermiticity (numerical error can cause slight deviations)
    D_next = (D_next + D_next.T.conj()) / 2.0

    return D_next # Possible TODO: ensure D_next is complex dtype with .astype(complex_dtype, copy=False)


def evolve_density_matrix_rk4_gpu(
    H_hop: cp.ndarray,
    H_onsite: cp.ndarray,
    initial_density: cp.ndarray,
    field_amplitude: Union[float, Callable[[float], float]],
    dt: float,
    total_time: float,
    decay_time: float = float("inf"),
    sample_every: int = 1,
    use_sparse: bool = True,
    precision: str = DEFAULT_PRECISION,
) -> List[cp.ndarray]:
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
    float_dtype, complex_dtype = get_cupy_dtypes(precision)
    print(f"Using GPU with precision: float={float_dtype}, complex={complex_dtype}")

    # Move data to GPU
    initial_density_gpu = cp.asarray(initial_density, dtype=complex_dtype)
    H_hop_gpu = cp.asarray(H_hop, dtype=float_dtype)
    H_onsite_gpu = cp.asarray(H_onsite, dtype=float_dtype)

    if use_sparse:
        # Use cupy.sparse to create sparse matrices on GPU
        H_hop_gpu = sparse.csr_matrix(H_hop_gpu, dtype=float_dtype)
        H_onsite_gpu = sparse.csr_matrix(H_onsite_gpu, dtype=float_dtype)

    D_t_gpu = initial_density_gpu.copy()
    result_gpu: List[cp.ndarray] = []  # Store results on GPU initially

    for step in trange(n_steps):
        if sample_every and (step % sample_every == 0):
            result_gpu.append(D_t_gpu.copy())

        t = step * dt
        # Pass GPU arrays to rk4_step
        D_t_gpu = rk4_step(t, D_t_gpu, H_hop_gpu, H_onsite_gpu, field_amplitude, dt, initial_density_gpu, decay_time, float_dtype, complex_dtype)

    result_gpu.append(D_t_gpu.copy())

    # Move results back to CPU
    result_cpu = [cp.asnumpy(res) for res in result_gpu]
    return result_cpu