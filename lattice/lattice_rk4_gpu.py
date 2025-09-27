from typing import Callable, List, Optional, Union
from tqdm import trange

import cupy as cp
try:  # CuPy changed sparse namespace across versions
    import cupy.sparse as sparse  # type: ignore
except Exception:  # pragma: no cover - fallback for older/newer CuPy
    from cupyx.scipy import sparse  # type: ignore
import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod

# Import CPU Observable type for snapshot (host-side) observers without creating heavy deps
try:
    from .lattice_rk4 import Observable  # type: ignore
except Exception:  # pragma: no cover - optional typing only
    class Observable:  # fallback stub if import context differs
        def measure(self, density_matrix: np.ndarray, step_index: int) -> float:  # noqa: D401
            pass

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


class GPUObservable(ABC):
        """GPU-side observable interface.

        Contract:
        - setup_gpu(...) is called once before the time loop to bind GPU data, allocate buffers
            and cache dtypes. Avoid any host synchronization in setup beyond initial data upload.
        - measure_gpu(density_gpu, step) is called at every integration step (or as configured by
            the solver). It must not perform cp.asnumpy or other host syncs. All state should remain
            on device; accumulate into device buffers.
        - finalize() is called after the loop and may transfer small aggregated results back to host.
        """

        def setup_gpu(
                self,
                H_hop: GpuMatrix,
                H_onsite: GpuMatrix,
                initial_density: cp.ndarray,
                dt: float,
                n_steps: int,
                float_dtype=cp.float64,
                complex_dtype=cp.complex128,
        ) -> None:
                del H_hop, H_onsite, initial_density, dt, n_steps, float_dtype, complex_dtype

        @abstractmethod
        def measure_gpu(self, density_gpu: cp.ndarray, step: int) -> None:
                ...

        def finalize(self):  # -> Any
                return None


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
      amplitude_value: field amplitude value for this specific time step.
      initial_density_gpu: Initial density matrix ρ₀ (dense GPU array), used for decay term.
      decay_rate: decay rate (1 / τ) as GPU float, or None if no decay.

    Returns:
      The time derivative of the density matrix (dense GPU ndarray).
    """

    # Calculate the time-dependent Hamiltonian
    H = H_hop + field_amplitude_value * H_onsite

    # Calculate -i[H, ρ]
    commutator = H @ density_matrix - density_matrix @ H
    d_rho_dt = -1j * commutator

    if decay_rate is not None and initial_density is not None:
        decay_term = (initial_density - density_matrix) * decay_rate
        d_rho_dt += decay_term

    return d_rho_dt


def rk4_step(
    density_matrix: cp.ndarray,
    H_hop: GpuMatrix,
    H_onsite: GpuMatrix,
    amp_t: cp.float_,
    amp_mid: cp.float_,  # Value at t + dt/2
    amp_next: cp.float_,  # Value at t + dt
    dt: cp.float_,
    initial_density: Optional[cp.ndarray] = None,
    decay_rate: Optional[cp.float_] = None,  # 1/decay_time
    float_dtype=cp.float64,
) -> cp.ndarray:
    """
    Perform a single RK4 step to evolve the density matrix.

    Parameters:
      density_matrix: Density matrix ρ(t) (dense GPU array).
      H_hop: Hopping Hamiltonian (dense or sparse GPU matrix).
      H_onsite: On-site potential Hamiltonian (dense or sparse GPU matrix).
      amp_t: Field amplitude value at time t.
      amp_mid: Field amplitude value at time t + dt/2.
      amp_next: Field amplitude value at time t + dt.
      dt: Time step (CuPy float scalar).
      initial_density_gpu: Initial density matrix ρ₀ (dense GPU array).
      decay_rate: decay rate (1/τ) as GPU float, or None.

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

    # Ensure hermiticity
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
    observables_gpu: Optional[List[GPUObservable]] = None,
    snapshot_observables_cpu: Optional[List[Observable]] = None,
) -> List[np.ndarray]:
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
    times_mid = times + np_float_dtype(0.5 * dt)

    amplitudes_t_np = field_amplitude_vectorized(times)
    amplitudes_mid_np = field_amplitude_vectorized(times_mid)

    # Move to GPU
    amplitudes_t_gpu = cp.asarray(amplitudes_t_np, dtype=float_dtype)
    amplitudes_mid_gpu = cp.asarray(amplitudes_mid_np, dtype=float_dtype)

    del amplitudes_t_np, amplitudes_mid_np, times, times_mid
    print("Amplitude precomputation complete.")

    # Move data to GPU
    initial_density_gpu = cp.asarray(initial_density, dtype=complex_dtype)
    H_hop_gpu = cp.asarray(H_hop, dtype=float_dtype)
    H_onsite_gpu = cp.asarray(H_onsite, dtype=float_dtype)

    if use_sparse:
        H_hop_gpu = sparse.csr_matrix(H_hop_gpu, dtype=float_dtype)
        H_onsite_gpu = sparse.csr_matrix(H_onsite_gpu, dtype=float_dtype)

    decay_rate_gpu: Optional[cp.float_] = None
    if decay_time != float("inf") and decay_time > 0:
        decay_rate_gpu = float_dtype(1.0 / decay_time)

    dt_gpu = float_dtype(dt)

    D_t_gpu = initial_density_gpu.copy()
    result_gpu: List[cp.ndarray] = []
    snapshot_steps: List[int] = []

    # Setup GPU observables once (no host sync)
    if observables_gpu:
        for obs in observables_gpu:
            obs.setup_gpu(
                D_t_gpu,
                dt,
                n_steps,
                float_dtype=float_dtype,
                complex_dtype=complex_dtype,
            )
    for step in trange(n_steps):
        if sample_every and (step >= first_snapshot_step) and (step % sample_every == 0):
            result_gpu.append(D_t_gpu.copy())
            snapshot_steps.append(step)

        # Measure GPU observables per step without host sync (state at time t)
        if observables_gpu:
            for obs in observables_gpu:
                obs.measure_gpu(D_t_gpu, step)

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

    # next loop iteration uses updated D_t_gpu

    result_gpu.append(D_t_gpu.copy())
    snapshot_steps.append(n_steps)

    # Convert stored snapshots to host once at the end
    result_cpu = [cp.asnumpy(res) for res in result_gpu]

    # Run CPU snapshot observables post-loop to avoid per-step transfers
    if snapshot_observables_cpu:
        for snap, step in zip(result_cpu, snapshot_steps):
            for obs in snapshot_observables_cpu:
                obs.measure(snap, step)

    # Finalize GPU observables (optional small host transfers)
    if observables_gpu:
        for obs in observables_gpu:
            obs.finalize()
    return result_cpu


# -----------------------
# Example GPU Observables
# -----------------------

class IdempotencyGPU(GPUObservable):
    """Tracks Tr[rho^2] efficiently on GPU without dense matmul.

    For Hermitian rho, Tr[rho^2] = sum_{ij} |rho_{ij}|^2 = ||rho||_F^2.
    """

    def setup_gpu(
        self,
        initial_density: cp.ndarray,
        dt: float,
        n_steps: int,
        float_dtype=cp.float64,
        complex_dtype=cp.complex128,
    ) -> None:
        del initial_density, dt, complex_dtype
        self._values = cp.empty(n_steps, dtype=float_dtype)
        self._index = 0

    def measure_gpu(self, density_gpu: cp.ndarray, step: int) -> None:
        # Frobenius norm squared (real scalar)
        self._values[self._index] = cp.sum(cp.abs(density_gpu) ** 2)
        self._index += 1

    def finalize(self):
        # Return numpy copy for used entries
        self.values = cp.asnumpy(self._values[: self._index])
        return self.values


class PolarisationGPU(GPUObservable):
    """Computes electronic polarisation P = (1/N) sum_i (r_i - origin) rho_ii on GPU.

    Parameters
    ----------
    site_positions: np.ndarray shape (N, 2)
        Cartesian coordinates per lattice site. Will be moved to GPU in setup.
    origin: np.ndarray shape (2,)
        Reference origin for polarisation.
    norm_N: Optional[float]
        Normalisation factor. Defaults to number of sites N.
    sample_every: Optional[int]
        If set, only records every k-th call; otherwise records each call.
    """

    def __init__(
        self,
        site_positions: NDArray[np.float64],
        origin: NDArray[np.float64],
        norm_N: Optional[float] = None,
        sample_every: Optional[int] = None,
    ) -> None:
        self._pos_np = np.asarray(site_positions, dtype=np.float64)
        self._origin_np = np.asarray(origin, dtype=np.float64)
        self._norm_N = norm_N
        self._sample_every = sample_every

    def setup_gpu(
        self,
        initial_density: cp.ndarray,
        dt: float,
        n_steps: int,
        float_dtype=cp.float64,
        complex_dtype=cp.complex128,
    ) -> None:
        del dt, complex_dtype
        N = initial_density.shape[0]
        self._N = N if self._norm_N is None else float(self._norm_N)
        self._pos = cp.asarray(self._pos_np, dtype=float_dtype)
        self._origin = cp.asarray(self._origin_np, dtype=float_dtype)
        # Allocate for per-step sampling; will slice in finalize if sampling is sparser
        self._values = cp.empty((n_steps, 2), dtype=float_dtype)
        self._index = 0

    def measure_gpu(self, density_gpu: cp.ndarray, step: int) -> None:
        if self._sample_every and step % self._sample_every != 0:
            return
        occ = cp.real(density_gpu.diagonal())  # (N,)
        centered = self._pos - self._origin  # (N,2)
        # Weighted sum over sites -> (2,)
        P_vec = (centered.T @ occ) / self._N
        self._values[self._index, :] = P_vec
        self._index += 1

    def finalize(self):
        # Trim unused rows if sampling was applied
        used = self._values[: self._index]
        self.values = cp.asnumpy(used)
        return self.values




class OrbitalPolarisationWithShapeGPU(GPUObservable):
    """GPU observable computing orbital polarisation using loop currents around cells.

    This avoids building the full current matrix J by directly sampling the bonds
    that form each unit-cell loop (provided by geometry). All heavy lifting stays
    on the GPU; only a compact P_orb time series is transferred at finalize().

    Parameters
    ----------
    site_positions : np.ndarray, shape (N, 2)
        Cartesian coordinates for each site index.
    curl_origin : np.ndarray, shape (2,)
        Reference origin used in polarisation evaluation.
    cell_anchor_sites : np.ndarray, shape (n_cells,)
        Anchor site index per unit cell (e.g., geometry.get_curl_sites()).
    cell_path_offsets : np.ndarray, shape (L, 2)
        For each edge in the oriented loop, offsets (di, dj) such that the bond current
        is taken from pair (i=row, j=col) => (row = s + di, col = s + dj) for anchor s.
        Orientation determines the sign according to the path.
    H_hop_cpu : np.ndarray, shape (N, N)
        CPU dense hopping matrix to pre-gather bond amplitudes along the loop edges.
        Using CPU here is OK since we only gather a small subset of entries once.
    norm_N : Optional[float]
        Normalisation factor; defaults to number of sites N.
    sample_every : Optional[int]
        If set, only records every k-th call; otherwise records each call.
    """

    def __init__(
        self,
        site_positions: NDArray[np.float64],
        curl_origin: NDArray[np.float64],
        cell_anchor_sites: NDArray[np.int64],
        cell_path_offsets: NDArray[np.int64],  # (L, 2) with (di,dj)
        t_hop: float = 1.0,
        m: float = 1.0,
        a_nn: float = 1.0,
        Area: float = 1.0,
        sample_every: Optional[int] = None,
    ) -> None:
        self._pos_np = np.asarray(site_positions, dtype=np.float64)
        self._curl_origin_np = np.asarray(curl_origin, dtype=np.float64)
        self._anchors_np = np.asarray(cell_anchor_sites, dtype=np.int64)
        self._path_np = np.asarray(cell_path_offsets, dtype=np.int64)
        self._sample_every = sample_every
        self.t_hop = float(t_hop)
        self._m = float(m)
        self._a_nn = float(a_nn)
        self._A = float(Area)

        # Precompute edge index pairs and corresponding H values on CPU (small)
        n_cells = self._anchors_np.shape[0]
        L = self._path_np.shape[0]
        rows = np.empty((n_cells, L), dtype=np.int64)
        cols = np.empty((n_cells, L), dtype=np.int64)
        for k, (di, dj) in enumerate(self._path_np):
            rows[:, k] = self._anchors_np + di
            cols[:, k] = self._anchors_np + dj
        self._rows_np = rows
        self._cols_np = cols
        # Edge vectors (CPU, will be moved to GPU in setup): r_k - r_l per (cell, edge)
        pos_cpu = self._pos_np
        self._edge_vecs_np = pos_cpu[self._rows_np] - pos_cpu[self._cols_np]  # (n_cells, L, 2)

    def setup_gpu(
        self,
        initial_density: cp.ndarray,
        dt: float,
        n_steps: int,
        float_dtype=cp.float64,
        complex_dtype=cp.complex128,
    ) -> None:
        del dt, complex_dtype
        # Normalisation by number of sites (optional)
        self._N = initial_density.shape[0]
        # Move geometry and pre-gathered data to GPU
        self._pos = cp.asarray(self._pos_np, dtype=float_dtype)
        self._curl_origin = cp.asarray(self._curl_origin_np, dtype=float_dtype)
        self._rows = cp.asarray(self._rows_np, dtype=cp.int64)
        self._cols = cp.asarray(self._cols_np, dtype=cp.int64)
        # Edge vectors and their 90° rotation R @ (r_k - r_l)
        self._edge_vecs = cp.asarray(self._edge_vecs_np, dtype=float_dtype)  # (n_cells, L, 2)
        # R @ v = [-v_y, v_x]
        self._rot_edge_vecs = cp.stack(
            (-self._edge_vecs[..., 1], self._edge_vecs[..., 0]), axis=-1
        )  # (n_cells, L, 2)
        # Per-cell centers using anchor site positions
        self._anchors = cp.asarray(self._anchors_np, dtype=cp.int64)
        self._cell_pos = self._pos[self._anchors]
        # Storage for results
        self._values = cp.empty((n_steps, 2), dtype=float_dtype)

    def measure_gpu(self, density_gpu: cp.ndarray, step: int) -> None:
        if self._sample_every and step % self._sample_every != 0:
            return
        # Bond currents along the oriented loop edges for each cell
        # I_ij = 2 * t_hop * Im(rho_ij)
        rho_ij = density_gpu[self._rows, self._cols]  # shape (n_cells, L)
        I_edges = 2.0 * self.t_hop * cp.imag(rho_ij)
        # Term 1: (9/4) * R_i * sum_edges I_edge per cell
        curl_per_cell = cp.sum(I_edges, axis=1)  # (n_cells,)
        centered = self._cell_pos - self._curl_origin  # (n_cells, 2)
        term1_vec = centered.T @ curl_per_cell  # (2,)
        # Term 2: -(5*sqrt(3)/16) * sum_edges [ R @ (r_k - r_l) * I_edge ] over cells
        weighted_rot_edges = (I_edges[..., None] * self._rot_edge_vecs).sum(axis=1)  # (n_cells, 2)
        term2_vec = cp.sum(weighted_rot_edges, axis=0)  # (2,)
        # Combine with overall factor -m * a_nn^4
        coeff = -self._m * (self._a_nn ** 2)
        c1 = coeff * (cp.sqrt(3.0) / 2.0)
        c2 = coeff * (-5.0 / 24.0)
        P_vec = c1 * term1_vec + c2 * term2_vec
        # Optional normalisation by Area to keep scale comparable with other observables
        P_vec = P_vec / self._A
        self._values[step, :] = P_vec

    def finalize(self):
        self.values = cp.asnumpy(self._values)
        return self.values


class DynamicsFrameRecorderGPU(GPUObservable):
    """Records frames of densities (diag rho), bond currents, and orbital charge components.

    Captures a window of steps [start_index, start_index + steps) at cadence sample_every:
    - densities: real(diag(rho)) per frame, shape (F, N)
    - bond currents: J_edges = 2 t_hop * Im(rho[row_nn, col_nn]) per frame, shape (F, E)
    - orbital charges (per cell):
        * curl_per_cell = sum_edges I_edge, shape (F, n_cells)
        * shape_sum_per_cell = sum_edges (R @ (r_k - r_l)) I_edge, shape (F, n_cells, 2)

    All computations are done on GPU; only final arrays are transferred on finalize().
    """

    def __init__(
        self,
        # Nearest-neighbour edge list (unique directed edges preferred)
        nn_rows: NDArray[np.int64],
        nn_cols: NDArray[np.int64],
        # Per-cell loop edges specified via anchors and path offsets (as in orbital observable)
        site_positions: NDArray[np.float64],
        cell_anchor_sites: NDArray[np.int64],
        cell_path_offsets: NDArray[np.int64],  # (L, 2)
        # Sampling window
        start_index: int,
        steps: int,
        sample_every: int = 1,
        # Model params
        t_hop: float = 1.0,
        m: float = 1.0,
        a_nn: float = 1.0,
    ) -> None:
        # NN edges (CPU)
        self._nn_rows_np = np.asarray(nn_rows, dtype=np.int64).ravel()
        self._nn_cols_np = np.asarray(nn_cols, dtype=np.int64).ravel()
        # Loop geometry (CPU)
        self._pos_np = np.asarray(site_positions, dtype=np.float64)
        self._anchors_np = np.asarray(cell_anchor_sites, dtype=np.int64)
        self._path_np = np.asarray(cell_path_offsets, dtype=np.int64)
        # Sampling controls
        self._start = int(start_index)
        self._end = int(start_index + steps)
        self._every = max(1, int(sample_every))
        self._t_hop = float(t_hop)
        self._m = float(m)
        self._a_nn = float(a_nn)

        # Precompute per-cell loop edge index arrays on CPU
        n_cells = self._anchors_np.shape[0]
        L = self._path_np.shape[0]
        rows = np.empty((n_cells, L), dtype=np.int64)
        cols = np.empty((n_cells, L), dtype=np.int64)
        for k, (di, dj) in enumerate(self._path_np):
            rows[:, k] = self._anchors_np + di
            cols[:, k] = self._anchors_np + dj
        self._rows_np = rows
        self._cols_np = cols
        # Edge vectors r_k - r_l (CPU)
        self._edge_vecs_np = self._pos_np[self._rows_np] - self._pos_np[self._cols_np]  # (n_cells, L, 2)

    def setup_gpu(
        self,
        initial_density: cp.ndarray,
        dt: float,
        n_steps: int,
        float_dtype=cp.float64,
        complex_dtype=cp.complex128,
    ) -> None:
        del dt, complex_dtype
        self._float_dtype = float_dtype
        N = initial_density.shape[0]
        self._N = int(N)

        # Move indices to GPU
        self._nn_rows = cp.asarray(self._nn_rows_np, dtype=cp.int64)
        self._nn_cols = cp.asarray(self._nn_cols_np, dtype=cp.int64)
        self._rows = cp.asarray(self._rows_np, dtype=cp.int64)
        self._cols = cp.asarray(self._cols_np, dtype=cp.int64)

        # Geometry on GPU
        self._edge_vecs = cp.asarray(self._edge_vecs_np, dtype=float_dtype)
        self._rot_edge_vecs = cp.stack((-self._edge_vecs[..., 1], self._edge_vecs[..., 0]), axis=-1)

        # Frame capacity
        total = max(0, self._end - self._start)
        frames = (total + self._every - 1) // self._every  # ceil division
        self._capacity = int(frames)

        # Allocate device buffers
        E = int(self._nn_rows_np.shape[0])
        n_cells = int(self._rows_np.shape[0])
        self._densities = cp.empty((frames, N), dtype=float_dtype)
        self._bond_currents = cp.empty((frames, E), dtype=float_dtype)
        self._curl = cp.empty((frames, n_cells), dtype=float_dtype)
        self._frame_index = 0

    def _should_record(self, step: int) -> bool:
        return (step >= self._start) and (step < self._end) and ((step - self._start) % self._every == 0)

    def measure_gpu(self, density_gpu: cp.ndarray, step: int) -> None:
        if not self._should_record(step):
            return
        fi = self._frame_index
        # 1) Onsite densities (real(diag rho))
        self._densities[fi, :] = cp.real(density_gpu.diagonal()).astype(self._float_dtype)
        # 2) Bond currents on NN edges
        rho_edges = density_gpu[self._nn_rows, self._nn_cols]
        J_edges = (2.0 * self._t_hop) * cp.imag(rho_edges).astype(self._float_dtype)
        self._bond_currents[fi, :] = J_edges
        # 3) Orbital charge components per cell (loop sums)
        coeff = -self._m * (self._a_nn ** 2)
        c1 = coeff * (cp.sqrt(3.0) / 2.0)
        rho_ij = density_gpu[self._rows, self._cols]  # (n_cells, L)
        I_edges = (2.0 * self._t_hop) * cp.imag(rho_ij).astype(self._float_dtype)
        curl_per_cell = cp.sum(I_edges, axis=1)  # (n_cells,)
        self._curl[fi, :] = c1 * curl_per_cell
        self._frame_index += 1

    def finalize(self):
        # Trim to number of recorded frames and move to host
        F = int(self._frame_index)
        self.densities = cp.asnumpy(self._densities[:F])
        self.bond_currents = cp.asnumpy(self._bond_currents[:F])
        self.orbital_curl = cp.asnumpy(self._curl[:F])
        return {
            "densities": self.densities,
            "bond_currents": self.bond_currents,
            "orbital_curl": self.orbital_curl,
        }
