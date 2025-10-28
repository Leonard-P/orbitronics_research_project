from ..observable import Observable
from .honeycomb_geometry import HoneycombLatticeGeometry
from ..backend import xp, Array, DTYPE, FDTYPE, USE_GPU
import numpy as np

class OrbitalPolarizationHoneycomb(Observable):
    """Measures the orbital polarization using loop currents around cells as
    :math:`\expval{P_{orb}} = -i\frac{m_e}{A_\mathrm{tot}} \sum_\alpha\sum_{(k,l)\in\circlearrowleft_{\vec R_\alpha}} (\sqrt 3\,\vec R_\alpha +\frac{5}{12}\begin{pmatrix}0&-1\\1&0\end{pmatrix} (\vec r_l - \vec r_k)) \Im \rho_{kl}`

    Parameters:
    """

    def __init__(self, geometry: HoneycombLatticeGeometry, electron_mass: float=0.741, measurement_start_time: float=0.0, measurement_end_time: float=float("inf"), measurement_stride: int=1):
        self._plaquette_anchors_cpu = geometry.bravais_site_indices
        self._stride = measurement_stride
        self._m = electron_mass
        self._path_offsets_cpu = geometry.plaquette_path_offsets_ccw
        self._origin = xp().array(geometry.origin)
        self.c1 = -self._m * (xp().sqrt(3.0) / 2.0)
        self.c2 = -self._m * (5.0 / 24.0)

        self.start = measurement_start_time
        self.end = measurement_end_time

        n_cells = self._plaquette_anchors_cpu.shape[0]
        L = self._path_offsets_cpu.shape[0]
        # compute rows/cols on CPU (NumPy) to build a boolean mask reliably for mypy
        rows_cpu = np.empty((n_cells, L), dtype=np.int64)
        cols_cpu = np.empty((n_cells, L), dtype=np.int64)
        for k, (di, dj) in enumerate(self._path_offsets_cpu):
            rows_cpu[:, k] = self._plaquette_anchors_cpu + di
            cols_cpu[:, k] = self._plaquette_anchors_cpu + dj

        # Filter out plaquettes whose path leaves the lattice (out-of-bounds or row wrap)
        Lx = geometry.Lx
        Ly = geometry.Ly
        N = Lx * Ly

        in_bounds = (rows_cpu >= 0) & (rows_cpu < N) & (cols_cpu >= 0) & (cols_cpu < N)

        ax = self._plaquette_anchors_cpu % Lx
        ay = self._plaquette_anchors_cpu // Lx
        rx = rows_cpu % Lx
        ry = rows_cpu // Lx
        cx = cols_cpu % Lx
        cy = cols_cpu // Lx

        dx_r = rx - ax[:, None]
        dy_r = ry - ay[:, None]
        dx_c = cx - ax[:, None]
        dy_c = cy - ay[:, None]

        # ensure no wrap-around relative to anchor; use integer NumPy array to satisfy mypy
        flat = self._path_offsets_cpu.astype(np.int64).ravel()
        w = int((flat % Lx).max())
        h = int((flat // Lx).max())

        valid_rows = (dx_r >= 0) & (dy_r >= 0) & (dx_r <= w) & (dy_r <= h)
        valid_cols = (dx_c >= 0) & (dy_c >= 0) & (dx_c <= w) & (dy_c <= h)

        valid_edge = in_bounds & valid_rows & valid_cols

        # only keep plaquettes whose edges all lie within the system
        cell_mask = valid_edge.all(axis=1)
        self._plaquette_anchors_cpu = self._plaquette_anchors_cpu[cell_mask]
        n_cells = self._plaquette_anchors_cpu.shape[0]
        self._A = n_cells * geometry.plaquette_area

        rows_cpu = rows_cpu[cell_mask]
        cols_cpu = cols_cpu[cell_mask]

        # move rows/cols to active backend for later advanced indexing
        self._rows = xp().array(rows_cpu, dtype=xp().int64)
        self._cols = xp().array(cols_cpu, dtype=xp().int64)

        site_positions = xp().array([
            geometry.index_to_position(i) for i in range(N)
        ])

        # Edge vectors r_l - r_k per (cell, edge)
        self._edge_vecs = site_positions[self._cols] - site_positions[self._rows]  # (n_cells, L, 2)
        # Their 90° rotation R @ (r_l - r_k) with R @ v = [-v_y, v_x]
        self._rot_edge_vecs = xp().stack(
            (-self._edge_vecs[..., 1], self._edge_vecs[..., 0]), axis=-1
        )  # (n_cells, L, 2)

        # per-plaquette positions using anchor site positions; index with backend array
        self._plaquette_positions = site_positions[xp().array(self._plaquette_anchors_cpu, dtype=np.int64)]

    def setup(self, dt: float, total_steps: int) -> None:
        # storage for results
        self.end = min(self.end, dt * total_steps)
        n_measurements = int((self.end - self.start) / (dt * self._stride)) + 1
        self._values = xp().empty((n_measurements, 2), dtype=FDTYPE)
        self._measurement_times = xp().empty((n_measurements,), dtype=FDTYPE)
        self._index = 0

    def measure(self, rho: Array, t: float, step_index: int) -> None:
        if t < self.start or t > self.end: return
        if step_index % self._stride != 0: return

        # Bond currents along the oriented loop edges for each cell
        # I_ij = 2 * (t_hop = 1) * Im(rho_ij)
        I_edges = 2.0 * xp().imag(rho[self._rows, self._cols])

        # R_alpha * sum_edges I_edge per cell
        curl_per_cell = xp().sum(I_edges, axis=1)  # (n_cells,)
        centered = self._plaquette_positions - self._origin  # (n_cells, 2)
        term1_vec = centered.T @ curl_per_cell  # (2,)

        # Term 2: sum_edges [ R @ (r_l - r_k) * I_edge ] over cells
        weighted_rot_edges = (I_edges[..., None] * self._rot_edge_vecs).sum(axis=1)  # (n_cells, 2)
        term2_vec = xp().sum(weighted_rot_edges, axis=0)  # (2,)

        P_vec = (self.c1 * term1_vec + self.c2 * term2_vec) / self._A
        self._values[self._index, :] = P_vec
        self._measurement_times[self._index] = t
        self._index += 1

    def finalize(self) -> None:
        """Some measurement windows leave the last _values entry unfilled."""
        if USE_GPU:
            # if backend was set to GPU, _values is a cupy array, so move to CPU
            self.values = self._values[:self._index].asnumpy()
            self.measurement_times = self._measurement_times[:self._index].asnumpy()
        else:
            self.values = self._values[:self._index]
            self.measurement_times = self._measurement_times[:self._index]