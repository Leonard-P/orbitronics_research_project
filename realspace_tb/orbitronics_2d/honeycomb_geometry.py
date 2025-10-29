from .lattice_2d_geometry import Lattice2DGeometry
from ..backend import FCPUArray
import numpy as np

class HoneycombLatticeGeometry(Lattice2DGeometry):
    def __init__(self, Lx: int, Ly: int):
        super().__init__()
        
        self.Lx = Lx
        self.Ly = Ly

        self.plaquette_path_offsets_ccw = np.array([
            (0, 1), (1, 2), (2, Lx + 2), (Lx + 2, Lx + 1), (Lx + 1, Lx), (Lx, 0)
        ])

        self._row_height = 1.5
        self._col_width = np.sqrt(3) / 2

        self.plaquette_area = np.sqrt(3) * 3 / 2

        self._origin = np.array([(self.Lx - 1) * self._col_width, (self.Ly - 1) * self._row_height]) / 2
        self._nearest_neighbors: "FCPUArray | None" = None
        self._bravais_site_indices: "FCPUArray | None" = None

    @property
    def nearest_neighbors(self) -> FCPUArray:
        """Array of nearest neighbor indices [[i, j], ...] = <i, j>"""
        if self._nearest_neighbors is not None:
            return self._nearest_neighbors
        
        neighbors = []
        for index in range(self.Lx * self.Ly):
            row = index // self.Lx
            col = index % self.Lx

            if (row + col) % 2 == 0:
                continue  # add each (A, B) pair only once

            deltas = [(-1, 0), (0, -1), (0, 1)]

            for dr, dc in deltas:
                neighbor_row = row + dr
                neighbor_col = col + dc
                if 0 <= neighbor_row < self.Ly and 0 <= neighbor_col < self.Lx:
                    neighbor_index = neighbor_row * self.Lx + neighbor_col
                    neighbors.append([index, neighbor_index])

        self._nearest_neighbors = np.array(neighbors, dtype=int)
        return self._nearest_neighbors

    @property
    def bravais_site_indices(self) -> FCPUArray:
        """List of all indices that form the Bravais lattice."""
        if self._bravais_site_indices is not None:
            return self._bravais_site_indices

        # Return indices where (i + j) % 2 == 0 (A sublattice)
        self._bravais_site_indices = np.array([i for i in range(self.Lx * self.Ly) if sum(divmod(i, self.Lx)) % 2 == 0])
        return self._bravais_site_indices

    @property
    def origin(self) -> FCPUArray:
        """Origin of the lattice as real space vector."""
        return self._origin

    def index_to_position(self, index: int) -> FCPUArray:
        row = index // self.Lx
        col = index % self.Lx

        y_offset = 0.25 * (-1) ** ((col + row) % 2)

        x = self._col_width * (index % self.Lx)
        y = self._row_height * row + y_offset

        return np.array([x, y])