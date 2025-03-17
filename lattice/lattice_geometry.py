from typing import List, Tuple
from abc import ABC, abstractmethod
import numpy as np


class LatticeGeometry(ABC):
    """Abstract base class for lattice geometries"""

    def __init__(self, dimensions: Tuple[int, int], cell_path: List[Tuple[int, int]] = None):
        self.dimensions = dimensions
        self.Lx, self.Ly = dimensions

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
    def gradient(self, f: np.ndarray) -> np.ndarray:
        """Calculate the gradient of a scalar field"""


class RectangularLatticeGeometry(LatticeGeometry):
    """Rectangular lattice geometry with connections between adjacent sites"""

    def __init__(self, dimensions: Tuple[int, int]):
        Lx, _ = dimensions
        path = [(0, 1), (1, Lx + 1), (Lx + 1, Lx), (Lx, 0)]
        super().__init__(dimensions, path)

    def get_hopping_matrix(self) -> np.ndarray:
        x_hop = np.tile([1] * (self.Lx - 1) + [0], self.Ly)[:-1]
        y_hop = np.array([1] * self.Lx * (self.Ly - 1))

        H = np.diag(x_hop, 1) + np.diag(y_hop, self.Lx)
        return H + H.conj().T

    def get_curl_sites(self) -> List[int]:
        return [i * self.Lx + j for i in range(self.Ly - self.cell_height) for j in range(self.Lx - self.cell_width)]

    def gradient(self, f: np.ndarray, axis: int) -> np.ndarray:

class SquareLatticeGeometry(RectangularLatticeGeometry):
    """RectangularLatticeGeometry with Lx = Ly"""

    def __init__(self, dimension: int):
        super().__init__((dimension, dimension))


class BrickwallLatticeGeometry(RectangularLatticeGeometry):
    """Brickwall lattice geometry with 2x1 unit cells, alternating vertical connections"""

    def __init__(self, dimensions: Tuple[int, int]):
        Lx, _ = dimensions
        path = np.array([(0, 1), (1, 2), (2, Lx + 2), (Lx + 2, Lx + 1), (Lx + 1, Lx), (Lx, 0)])
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
