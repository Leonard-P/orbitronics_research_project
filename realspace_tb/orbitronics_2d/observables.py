from ..observable import Observable
from .honeycomb_geometry import HoneycombLatticeGeometry
from .. import backend as B
from typing import cast
import numpy as np


class PlaquetteOAMObservable(Observable):
    r"""Measures the plaquette orbital angular momentum using loop currents around cells as
    :math:`\expval{\int_{A_\circlearrowleft} d^2s \ L_z} = -\frac{\sqrt{3} m_e}{2} \sum_{(k,l)\in\circlearrowleft_{\vec R}} J_{kl}`
    """

    def __init__(
        self,
        geometry: HoneycombLatticeGeometry,
        electron_mass: float = 0.741,
        measurement_start_time: float = 0.0,
        measurement_end_time: float = float("inf"),
        measurement_stride: int = 1,
    ):
        super().__init__(
            measurement_start_time, measurement_end_time, measurement_stride
        )

        self._plaquette_anchors_cpu = geometry.bravais_site_indices
        self._c = -electron_mass * geometry.plaquette_area / 3
        self._path_offsets_cpu = geometry.plaquette_path_offsets_ccw

        n_cells = self._plaquette_anchors_cpu.shape[0]
        L = self._path_offsets_cpu.shape[0]

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
        self._n_cells = self._plaquette_anchors_cpu.shape[0]

        rows_cpu = rows_cpu[cell_mask]
        cols_cpu = cols_cpu[cell_mask]

        # move rows/cols to active backend for later advanced indexing
        self._rows = B.xp().array(rows_cpu, dtype=B.xp().int64)
        self._cols = B.xp().array(cols_cpu, dtype=B.xp().int64)

    def setup(self, dt: float, total_steps: int) -> int:
        n_measurements = super().setup(dt, total_steps)
        self._values = B.xp().empty((n_measurements, self._n_cells), dtype=B.FDTYPE)
        self._measurement_times = B.xp().empty((n_measurements,), dtype=B.FDTYPE)
        self._index = 0

        return n_measurements

    def measure(self, rho: B.Array, t: float, step_index: int) -> None:
        if not self._should_measure(t, step_index):
            return

        # Bond currents along the oriented loop edges for each cell
        I_edges = 2.0 * B.xp().imag(rho[self._rows, self._cols])
        self._values[self._index, :] = self._c * B.xp().sum(I_edges, axis=1)  # (n_cells,)
        self._measurement_times[self._index] = t
        self._index += 1

    def finalize(self) -> None:
        """Some measurement windows leave the last _values entry unfilled."""
        if B.USE_GPU:
            # if backend was set to GPU, _values is a cupy array, so move to CPU
            self.values = self._values[: self._index].get()
            self.measurement_times = self._measurement_times[: self._index].get()
        else:
            self.values = self._values[: self._index]
            self.measurement_times = self._measurement_times[: self._index]


class OrbitalPolarizationObservable(PlaquetteOAMObservable):
    r"""Measures the orbital polarization using loop currents around cells as
    :math:`\expval{P_{orb}} = -i\frac{m_e}{A_\mathrm{tot}} \sum_\alpha\sum_{(k,l)\in\circlearrowleft_{\vec R_\alpha}} (\sqrt 3\,\vec R_\alpha +\frac{5}{12}\begin{pmatrix}0&-1\\1&0\end{pmatrix} (\vec r_l - \vec r_k)) \Im \rho_{kl}`
    """

    def __init__(
        self,
        geometry: HoneycombLatticeGeometry,
        electron_mass: float = 0.741,
        measurement_start_time: float = 0.0,
        measurement_end_time: float = float("inf"),
        measurement_stride: int = 1,
    ):
        super().__init__(
            geometry,
            electron_mass,
            measurement_start_time,
            measurement_end_time,
            measurement_stride,
        )

        self._origin = B.xp().array(geometry.origin)
        self._m = electron_mass
        self._c1 = -self._m * (B.xp().sqrt(3.0) / 2.0)
        self._c2 = -self._m * (5.0 / 24.0)
        self._A = self._n_cells * geometry.plaquette_area

        site_positions = B.xp().array(
            [geometry.index_to_position(i) for i in range(geometry.Lx * geometry.Ly)],
            dtype=B.FDTYPE,
        )

        # Edge vectors r_l - r_k per (cell, edge)
        self._edge_vecs = (
            site_positions[self._cols] - site_positions[self._rows]
        )  # (n_cells, L, 2)
        # Their 90° rotation R @ (r_l - r_k) with R @ v = [-v_y, v_x]
        self._rot_edge_vecs = B.xp().stack(
            (-self._edge_vecs[..., 1], self._edge_vecs[..., 0]), axis=-1
        )  # (n_cells, L, 2)

        # per-plaquette positions using anchor site positions; index with backend array
        self._plaquette_positions = site_positions[
            B.xp().array(self._plaquette_anchors_cpu, dtype=np.int64)
        ]

    def setup(self, dt: float, total_steps: int) -> int:
        n_measurements = super().setup(dt, total_steps)
        self._values = B.xp().empty((n_measurements, 2), dtype=B.FDTYPE)
        self._measurement_times = B.xp().empty((n_measurements,), dtype=B.FDTYPE)
        self._index = 0

        return n_measurements

    def measure(self, rho: B.Array, t: float, step_index: int) -> None:
        if not self._should_measure(t, step_index):
            return

        # Bond currents along the oriented loop edges for each cell
        # I_ij = 2 * (t_hop = 1) * Im(rho_ij)
        I_edges = 2.0 * B.xp().imag(rho[self._rows, self._cols])

        # R_alpha * sum_edges I_edge per cell
        curl_per_cell = B.xp().sum(I_edges, axis=1)  # (n_cells,)
        centered = self._plaquette_positions - self._origin  # (n_cells, 2)
        term1_vec = centered.T @ curl_per_cell  # (2,)

        # Term 2: sum_edges [ R @ (r_l - r_k) * I_edge ] over cells
        weighted_rot_edges = (I_edges[..., None] * self._rot_edge_vecs).sum(
            axis=1
        )  # (n_cells, 2)
        term2_vec = B.xp().sum(weighted_rot_edges, axis=0)  # (2,)

        P_vec = (self._c1 * term1_vec + self._c2 * term2_vec) / self._A
        self._values[self._index, :] = P_vec
        self._measurement_times[self._index] = t
        self._index += 1

    def finalize(self) -> None:
        """Some measurement windows leave the last _values entry unfilled."""
        if B.USE_GPU:
            # if backend was set to GPU, _values is a cupy array, so move to CPU
            self.values = self._values[: self._index].get()
            self.measurement_times = self._measurement_times[: self._index].get()
        else:
            self.values = self._values[: self._index]
            self.measurement_times = self._measurement_times[: self._index]


class SiteDensityObservable(Observable):
    """Measures the site-resolved electron density :math:`n_i = \rho_{ii}`."""

    def __init__(
        self,
        geometry: HoneycombLatticeGeometry,
        measurement_start_time: float = 0.0,
        measurement_end_time: float = float("inf"),
        measurement_stride: int = 1,
    ):
        super().__init__(
            measurement_start_time, measurement_end_time, measurement_stride
        )
        self._N = geometry.Lx * geometry.Ly

    def setup(self, dt: float, total_steps: int) -> int:
        n_measurements = super().setup(dt, total_steps)
        self._values = B.xp().empty((n_measurements, self._N), dtype=B.FDTYPE)
        self._measurement_times = B.xp().empty((n_measurements,), dtype=B.FDTYPE)
        self._index = 0

        return n_measurements

    def measure(self, rho: B.Array, t: float, step_index: int) -> None:
        if not self._should_measure(t, step_index):
            return

        density = B.xp().real(B.xp().diag(rho))
        self._values[self._index, :] = density
        self._measurement_times[self._index] = t
        self._index += 1

    def finalize(self) -> None:
        if B.USE_GPU:
            self.values = self._values[: self._index].get()
            self.measurement_times = self._measurement_times[: self._index].get()
        else:
            self.values = self._values[: self._index]
            self.measurement_times = self._measurement_times[: self._index]


class BondCurrentObservable(Observable):
    """Measures the bond currents :math:`I_{ij} = 2 t_{hop} \Im \rho_{ij}`."""

    def __init__(
        self,
        geometry: HoneycombLatticeGeometry,
        measurement_start_time: float = 0.0,
        measurement_end_time: float = float("inf"),
        measurement_stride: int = 1,
    ):
        super().__init__(
            measurement_start_time, measurement_end_time, measurement_stride
        )
        self._nearest_neighbors = B.xp().array(
            geometry.nearest_neighbors, dtype=B.xp().int64
        )

    def setup(self, dt: float, total_steps: int) -> int:
        n_measurements = super().setup(dt, total_steps)

        self._nn_rows = B.xp().array(self._nearest_neighbors[:, 0], dtype=B.xp().int64)
        self._nn_cols = B.xp().array(self._nearest_neighbors[:, 1], dtype=B.xp().int64)

        E = int(self._nn_rows.shape[0])
        self._values = B.xp().empty((n_measurements, E), dtype=B.FDTYPE)
        self._measurement_times = B.xp().empty((n_measurements,), dtype=B.FDTYPE)
        self._index = 0

        return n_measurements

    def measure(self, rho: B.Array, t: float, step_index: int) -> None:
        if not self._should_measure(t, step_index):
            return

        # Vectorized gather of nearest-neighbor bond currents
        currents = 2.0 * B.xp().imag(rho[self._nn_rows, self._nn_cols])  # (E,)
        self._values[self._index, :] = currents
        self._measurement_times[self._index] = t
        self._index += 1

    def finalize(self) -> None:
        if B.USE_GPU:
            self.values = self._values[: self._index].get()
            self.measurement_times = self._measurement_times[: self._index].get()
        else:
            self.values = self._values[: self._index]
            self.measurement_times = self._measurement_times[: self._index]


class LatticeFrameObservable(Observable):
    """Measures the full density matrix at given time steps for debugging and visualization."""

    def __init__(
        self,
        geometry: HoneycombLatticeGeometry,
        electron_mass: float = 0.741,
        measurement_start_time: float = 0.0,
        measurement_end_time: float = float("inf"),
        measurement_stride: int = 1,
    ) -> None:
        super().__init__(
            measurement_start_time, measurement_end_time, measurement_stride
        )

        self.geometry = geometry

        self.density_obs = SiteDensityObservable(
            geometry, measurement_start_time, measurement_end_time, measurement_stride
        )

        self.current_obs = BondCurrentObservable(
            geometry, measurement_start_time, measurement_end_time, measurement_stride
        )

        self.plaquette_oam_obs = PlaquetteOAMObservable(
            geometry,
            electron_mass,
            measurement_start_time,
            measurement_end_time,
            measurement_stride,
        )

        self.plaquette_anchor_indices = self.plaquette_oam_obs._plaquette_anchors_cpu

    def setup(self, dt: float, total_steps: int) -> int:
        self.current_obs.setup(dt, total_steps)
        self.density_obs.setup(dt, total_steps)
        self.plaquette_oam_obs.setup(dt, total_steps)

        return super().setup(dt, total_steps)

    def measure(self, rho: B.Array, t: float, step_index: int) -> None:
        self.density_obs.measure(rho, t, step_index)
        self.current_obs.measure(rho, t, step_index)
        self.plaquette_oam_obs.measure(rho, t, step_index)

    def finalize(self) -> None:
        self.density_obs.finalize()
        self.current_obs.finalize()
        self.plaquette_oam_obs.finalize()

        index = self.density_obs._index

        self.values = {
            "densities": cast(B.FCPUArray, self.density_obs.values),
            "currents": cast(B.FCPUArray, self.current_obs.values),
            "plaquette_oam": cast(B.FCPUArray, self.plaquette_oam_obs.values),
        }

        self.measurement_times = self.density_obs.measurement_times[:index]
