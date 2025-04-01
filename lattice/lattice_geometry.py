from typing import List, Tuple
from abc import ABC, abstractmethod
import numpy as np


class LatticeGeometry(ABC):
    """Abstract base class for lattice geometries"""

    def __init__(self, dimensions: Tuple[int, int], cell_path: List[Tuple[int, int]] = None):
        self.dimensions = dimensions
        self.Lx, self.Ly = dimensions
        self.origin = (np.array([self.Lx, self.Ly]) - 1) / 2

        self.cell_path = cell_path
        flat = np.array(self.cell_path).flatten()
        self.cell_width = (flat % self.Lx).max()
        self.cell_height = (flat // self.Lx).max()

    def site_to_position(self, site_index: int) -> Tuple[int, int]:
        """Convert site index to (x, y) position"""
        return site_index % self.Lx, site_index // self.Lx

    def position_to_site(self, x: int, y: int) -> int:
        """Convert (x, y) position to site index"""
        return y * self.Lx + x

    @abstractmethod
    def get_curl_sites(self) -> List[int]:
        """Starting sites for curl calculation"""

    @abstractmethod
    def get_hopping_matrix(self) -> np.ndarray:
        """Matrix defining hopping energies between sites"""

    @abstractmethod
    def cell_field_gradient(self, f: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        """Calculate the gradient of a scalar field evaluated at each unit cell"""


class RectangularLatticeGeometry(LatticeGeometry):
    """Rectangular lattice geometry with connections between adjacent sites"""

    def __init__(self, dimensions: Tuple[int, int]):
        Lx, _ = dimensions
        path = [(1, 0), (Lx + 1, 1), (Lx, Lx + 1), (0, Lx)]
        super().__init__(dimensions, path)

    def get_hopping_matrix(self) -> np.ndarray:
        x_hop = np.tile([1] * (self.Lx - 1) + [0], self.Ly)[:-1]
        y_hop = np.array([1] * self.Lx * (self.Ly - 1))

        H = np.diag(x_hop, 1) + np.diag(y_hop, self.Lx)
        return H + H.conj().T

    def get_curl_sites(self) -> List[int]:
        return [i * self.Lx + j for i in range(self.Ly - self.cell_height) for j in range(self.Lx - self.cell_width)]

    def cell_field_gradient(self, f: dict[int, float]) -> dict[int, Tuple[float, float]]:
        field_array = np.array([[f.get(self.position_to_site(x, y), 0) for x in range((self.Lx - 1))] for y in range(self.Ly - 1)])
        grad = np.gradient(field_array)
        return {self.position_to_site(x, y): (grad[1][y, x], grad[0][y, x]) for y in range(self.Ly - 1) for x in range(self.Lx - 1)}


class SquareLatticeGeometry(RectangularLatticeGeometry):
    """RectangularLatticeGeometry with Lx = Ly"""

    def __init__(self, dimension: int):
        super().__init__((dimension, dimension))


class BrickwallLatticeGeometry(RectangularLatticeGeometry):
    """Brickwall lattice geometry with 2x1 unit cells, alternating vertical connections"""

    def __init__(self, dimensions: Tuple[int, int]):
        Lx, _ = dimensions
        path = np.array([(1, 0), (2, 1), (Lx + 2, 2), (Lx + 1, Lx + 2), (Lx, Lx + 1), (0, Lx)])
        super(RectangularLatticeGeometry, self).__init__(dimensions, path)

    def get_hopping_matrix(self) -> np.ndarray:
        if self.Lx % 2 == 0:
            y_hop_row = np.tile([0, 1], self.Lx // 2)
            y_hop = np.concatenate([y_hop_row, 1 - y_hop_row] * (self.Ly // 2 - 1) + [y_hop_row] + [1 - y_hop_row] * (self.Ly % 2))
        else:
            y_hop_row = np.tile([0, 1], (self.Ly - 1) * self.Lx // 2)
            y_hop = np.concatenate([y_hop_row, [0] * (1 - self.Ly % 2)])

        y_mask = 1 - (np.diag(y_hop, self.Lx) + np.diag(y_hop, -self.Lx))

        return super().get_hopping_matrix() * y_mask

    def get_curl_sites(self) -> List[int]:
        return [
            i * self.Lx + j
            for i in range(0, self.Ly - self.cell_height, self.cell_height)
            for j in range(i % 2, self.Lx - self.cell_width, self.cell_width)
        ]

    def cell_field_gradient(self, f: dict[int, float]) -> dict[int, np.ndarray]:
        grad: dict[int, np.ndarray] = {}

        for site in self.get_curl_sites():
            x, y = self.site_to_position(site)

            f_site = f.get(site, None)

            # get nearest neighbors
            f_lr = f.get(self.position_to_site(x + 1, y - 1), None)
            f_ul = f.get(self.position_to_site(x - 1, y + 1), None)
            f_ur = f.get(self.position_to_site(x + 1, y + 1), None)
            f_ll = f.get(self.position_to_site(x - 1, y - 1), None)

            # Compute central differences
            if f_lr is not None and f_ul is not None:
                df_da1 = (f_ul - f_lr) / (2 * np.sqrt(2))
            else:
                if f_lr is not None:
                    df_da1 = (f_site - f_lr) / np.sqrt(2)
                elif f_ul is not None:
                    df_da1 = (f_ul - f_site) / np.sqrt(2)
                else:
                    df_da1 = 0

            if f_ur is not None and f_ll is not None:
                df_da2 = (f_ur - f_ll) / (2 * np.sqrt(2))
            else:
                if f_ur is not None:
                    df_da2 = (f_ur - f_site) / np.sqrt(2)
                elif f_ll is not None:
                    df_da2 = (f_site - f_ll) / np.sqrt(2)
                else:
                    df_da2 = 0

            # Convert to Cartesian coordinates with Jacobian. Should be equivalent of rotating by 45 degrees and normalizing by sqrt(2)
            # TODO: check if above normalization is correct or if extra factors are present
            J = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
            grad[site] = J @ np.array([df_da2, df_da1])

        return grad


class HexagonalLatticeGeometry(BrickwallLatticeGeometry):
    def __init__(self, dimensions):
        super().__init__(dimensions)
        
        self.row_height = 1.5
        self.col_width = np.sqrt(3) / 2
        self.origin = np.array([(self.Lx-1) * self.col_width, (self.Ly-1) * self.row_height]) / 2

    def site_to_position(self, site_index: int) -> Tuple[float, float]:
        row = site_index // self.Lx
        col = site_index % self.Lx

        y_offset = 0.25 * (-1) ** ((col + row) % 2)

        x = self.col_width * (site_index % self.Lx)
        y = self.row_height * row + y_offset

        return x, y

    def cell_field_gradient(self, f: dict[int, float]) -> dict[int, np.ndarray]:
        # TODO
        raise NotImplementedError