from ..hamiltonian import Hamiltonian
from .lattice_2d_geometry import Lattice2DGeometry
from ..backend import Array, SparseArray, xp, xp_sparse, FCPUArray
from abc import ABC, abstractmethod
import numpy as np

class HomogeneousFieldAmplitude(ABC):
    """Abstract base class for homogeneous electric field with only scalar time dependence."""

    @abstractmethod
    def amplitude_at_time(self, time: "float | Array") -> "float | Array":
        """Return the electric field amplitude at a given time."""
        ...

    direction: FCPUArray = np.zeros(2)



class RampedACFieldAmplitude(HomogeneousFieldAmplitude):
    """
    Electric field amplitude ramping over time:
    E(t) = E0 * sin^2(pi * t / 2 * T_ramp) * sin(ω t), capped at E0.
    """

    def __init__(self, E0: float, omega: float, T_ramp: float, direction: FCPUArray):
        self.E0 = E0
        self.omega = omega
        self.ramp_time = T_ramp
        self.direction = direction

    def amplitude_at_time(self, t: "float | Array") -> "float | Array":
        # TODO maybe make it CPU only and move backend transfer to Hamiltonian class
        # -> avoid confusion and minimize code scope of GPU backend
        ramp = xp().where(
            xp().array(t) < self.ramp_time,
            xp().sin(xp().pi * t / (2 * self.ramp_time))**2, 
            xp().ones_like(t)
        )
        return self.E0 * ramp * xp().sin(self.omega * t)


class LinearFieldHamiltonian(Hamiltonian):
    def __init__(self, geometry: Lattice2DGeometry, field: HomogeneousFieldAmplitude):
        super().__init__()
        
        self.geometry = geometry
        self.field = field

        # construct H_0 such that all nearest neighbor hoppings are -1
        size = geometry.Lx * geometry.Ly
        row_indices = []
        col_indices = []
        data = []
        for i, j in geometry.nearest_neighbors:
            row_indices.append(i)
            col_indices.append(j)
            data.append(-1.0)
            # add h.c.
            row_indices.append(j)
            col_indices.append(i)
            data.append(-1.0)
        self.H_0 = xp_sparse().coo_matrix((data, (row_indices, col_indices)), shape=(size, size)).tocsr()

        # make a sparse matrix diag(r_i . E) for position operator along field direction
        position_shifts = xp().array([
            np.dot(geometry.index_to_position(index), field.direction) for index in range(size)
        ])
        position_shifts -= xp().mean(position_shifts)  # center around zero

        self.position_operator = xp_sparse().diags(position_shifts, format="csr")

    def at_time(self, t: float) -> SparseArray:
        # Implementation of the Hamiltonian at time t
        return self.H_0 + self.field.amplitude_at_time(t) * self.position_operator