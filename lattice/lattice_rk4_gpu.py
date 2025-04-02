from typing import Callable, List, Optional, Union
from tqdm import trange

import cupy as cp
import cupy.sparse as sparse  # Use cupy.sparse
import numpy as np
from numpy.typing import NDArray

InputMatrix = np.ndarray
GpuMatrix = Union[cp.ndarray, sparse.spmatrix]


DEFAULT_PRECISION = "double"  # Options: 'single', 'double'


def get_cupy_dtypes(precision: str = DEFAULT_PRECISION):
    """Gets the CuPy float and complex dtypes based on precision string."""
    if precision.lower() == "single":
        return cp.float32, cp.complex64, np.float32
    elif precision.lower() == "double":
        return cp.float64, cp.complex128, np.float64
    else:
        raise ValueError("precision must be 'single' or 'double'")


def time_evolution_derivative(
    density_matrix: cp.ndarray,
    H_hop: GpuMatrix,
    H_onsite: GpuMatrix,
    field_amplitude_value: cp.float_,
    initial_density: Optional[cp.ndarray] = None,
    decay_rate: Optional[cp.float_] = None,  # Precomputed 1/decay_time or None
) -> cp.ndarray:
    """
    Calculate the time derivative of the density matrix.

    ∂ρ/∂t = -i[H, ρ] + (ρ₀ - ρ) * decay_rate

    Parameters:
      density_matrix: Density matrix ρ(t) (dense GPU array).
      H_hop: Hopping Hamiltonian (dense or sparse GPU matrix).
      H_onsite: On-site potential Hamiltonian (dense or sparse GPU matrix).
      amplitude_value: Precomputed field amplitude value for this specific time step.
      initial_density_gpu: Initial density matrix ρ₀ (dense GPU array), used for decay term.
      decay_rate: Precomputed decay rate (1 / τ) as GPU float, or None if no decay.

    Returns:
      The time derivative of the density matrix (dense GPU ndarray).
    """

    # Calculate the time-dependent Hamiltonian
    H = H_hop + field_amplitude_value * H_onsite

    # Calculate -i[H, ρ]
    commutator = H @ density_matrix - density_matrix @ H
    d_rho_dt = -1j * commutator

    if decay_rate is not None and initial_density is not None:
        # Ensure decay_term calculation maintains correct type
        decay_term = (initial_density - density_matrix) * decay_rate
        d_rho_dt += decay_term  # Adding complex + complex

    return d_rho_dt


def rk4_step(
    density_matrix: cp.ndarray,
    H_hop: GpuMatrix,
    H_onsite: GpuMatrix,
    amp_t: cp.float_,
    amp_mid: cp.float_,  # Value at t + dt/2 (used twice)
    amp_next: cp.float_,  # Value at t + dt
    dt: cp.float_,  # dt already cast to float_dtype
    initial_density: Optional[cp.ndarray] = None,
    decay_rate: Optional[cp.float_] = None,  # Precomputed 1/decay_time
    float_dtype=cp.float64,
) -> cp.ndarray:
    """
    Perform a single Runge-Kutta 4th order step to evolve the density matrix.

    Parameters:
      density_matrix: Density matrix ρ(t) (dense GPU array).
      H_hop: Hopping Hamiltonian (dense or sparse GPU matrix).
      H_onsite: On-site potential Hamiltonian (dense or sparse GPU matrix).
      amp_t: Field amplitude value at time t.
      amp_mid: Field amplitude value at time t + dt/2.
      amp_next: Field amplitude value at time t + dt.
      dt: Time step (CuPy float scalar).
      initial_density_gpu: Initial density matrix ρ₀ (dense GPU array).
      decay_rate: Precomputed decay rate (1/τ) as GPU float, or None.

    Returns:
      The density matrix after one RK4 step (dense GPU ndarray).
    """
    # RK4 Coefficients
    one_sixth = float_dtype(1.0 / 6.0)
    one_half = float_dtype(0.5)
    two = float_dtype(2.0)

    k1 = time_evolution_derivative(density_matrix, H_hop, H_onsite, amp_t, initial_density, decay_rate)
    rho_k1 = density_matrix + (one_half * dt) * k1

    k2 = time_evolution_derivative(rho_k1, H_hop, H_onsite, amp_mid, initial_density, decay_rate)
    rho_k2 = density_matrix + (one_half * dt) * k2

    k3 = time_evolution_derivative(rho_k2, H_hop, H_onsite, amp_mid, initial_density, decay_rate)
    rho_k3 = density_matrix + dt * k3

    k4 = time_evolution_derivative(rho_k3, H_hop, H_onsite, amp_next, initial_density, decay_rate)

    D_next = density_matrix + one_sixth * dt * (k1 + two * k2 + two * k3 + k4)

    # Ensure hermiticity (numerical error can cause slight deviations)
    D_next = (D_next + D_next.T.conj()) * one_half

    return D_next


def evolve_density_matrix_rk4_gpu(
    H_hop: InputMatrix,
    H_onsite: InputMatrix,
    initial_density: InputMatrix,
    field_amplitude_vectorized: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    dt: float,
    total_time: float,
    decay_time: float = float("inf"),
    sample_every: int = 1,
    first_snapshot_step: int = 0,
    use_sparse: bool = True,
    precision: str = DEFAULT_PRECISION,
) -> List[cp.ndarray]:
    """
    Evolve the density matrix using the RK4 method.

    Parameters:
      H_hop: Hopping Hamiltonian (dense ndarray). Will be converted to sparse if use_sparse is True.
      H_onsite: On-site potential Hamiltonian (dense ndarray). Converted similarly.
      initial_density: Initial density matrix ρ₀ (dense ndarray).
      field_amplitude_vectorized: Field amplitude as a callable.
      dt: Time step.
      total_time: Total simulation time.
      decay_time: Decay time constant τ.
      sample_every: Store every n-th step (if 0, only the final state is stored).
      use_sparse: If True, convert Hamiltonians to sparse matrices for multiplication efficiency.
      precision: 'single' or 'double'.

    Returns:
      A list of density matrices (dense ndarrays) at each stored time step.
    """
    n_steps = int(round(total_time / dt))
    float_dtype, complex_dtype, np_float_dtype = get_cupy_dtypes(precision)
    print(f"Using GPU with precision: float={float_dtype}, complex={complex_dtype}")

    times = np.arange(n_steps + 1, dtype=np_float_dtype) * dt
    times_mid = times_t + np_float_dtype(0.5 * dt)

    amplitudes_t_np = field_amplitude_vectorized(times)
    amplitudes_mid_np = field_amplitude_vectorized(times_mid)

    # 3. Ensure correct dtype and move precomputed amplitudes to GPU
    # Use copy=False if the vectorized function already returns the correct dtype
    amplitudes_t_gpu = cp.asarray(amplitudes_t_np, dtype=float_dtype)
    amplitudes_mid_gpu = cp.asarray(amplitudes_mid_np, dtype=float_dtype)

    del amplitudes_t_np, amplitudes_mid_np, times_t, times_mid
    print("Amplitude precomputation complete.")

    # Move data to GPU
    initial_density_gpu = cp.asarray(initial_density, dtype=complex_dtype)
    H_hop_gpu = cp.asarray(H_hop, dtype=float_dtype)
    H_onsite_gpu = cp.asarray(H_onsite, dtype=float_dtype)

    if use_sparse:
        # Use cupy.sparse to create sparse matrices on GPU
        H_hop_gpu = sparse.csr_matrix(H_hop_gpu, dtype=float_dtype)
        H_onsite_gpu = sparse.csr_matrix(H_onsite_gpu, dtype=float_dtype)

    decay_rate_gpu: Optional[cp.float_] = None
    if decay_time != float("inf") and decay_time > 0:
        decay_rate_gpu = float_dtype(1.0 / decay_time)

    dt_gpu = float_dtype(dt)

    D_t_gpu = initial_density_gpu.copy()
    result_gpu: List[cp.ndarray] = []  # Store results on GPU initially

    for step in trange(n_steps):
        if sample_every and (step >= first_snapshot_step) and (step % sample_every == 0):
            result_gpu.append(D_t_gpu.copy())

        amp_t = amplitudes_t_gpu[step]
        amp_mid = amplitudes_mid_gpu[step]
        amp_next = amplitudes_t_gpu[step + 1]

        D_t_gpu = rk4_step(
            D_t_gpu,
            H_hop_gpu,
            H_onsite_gpu,
            amp_t,
            amp_mid,
            amp_next,
            dt_gpu,
            initial_density_gpu, 
            decay_rate_gpu, 
            float_dtype,
        )

    result_gpu.append(D_t_gpu.copy())

    # Move results back to CPU
    result_cpu = [cp.asnumpy(res) for res in result_gpu]
    return result_cpu
