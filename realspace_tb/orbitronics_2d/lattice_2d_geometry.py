from abc import ABC, abstractmethod
from ..backend import FCPUArray
import numpy as np
from numpy.typing import NDArray

class Lattice2DGeometry(ABC):
    def __init__(self) -> None:
        self._site_positions: "FCPUArray | None" = None

    @property
    def site_positions(self) -> FCPUArray:
        if self._site_positions is None:
            self._site_positions = np.array([
                self.index_to_position(i) for i in range(self.Lx * self.Ly)
            ])
        return self._site_positions

    @abstractmethod
    def index_to_position(self, index: int) -> FCPUArray:
        """Convert site index to real space position"""
        ...

    @property
    @abstractmethod
    def nearest_neighbors(self) -> FCPUArray:
        """Array of nearest neighbor indices [[i, j], ...] = <i, j>"""
        ...

    @property
    @abstractmethod
    def bravais_site_indices(self) -> FCPUArray:
        """List of all indices that form the Bravais lattice."""
        ...

    @property
    def origin(self) -> FCPUArray:
        """Origin of the lattice as real space vector."""
        return np.array([0.0, 0.0])

    Lx: int
    Ly: int

    # [[i, j], ...] the integer offsets of the plaquette that need to be added to the bravais lattice index to traverse the ring of bonds i->j around the plaquette counter-clockwise (looking against z)
    plaquette_path_offsets_ccw: NDArray[np.int_]

    # real space area of a single plaquette, often the unit cell area
    plaquette_area: float

