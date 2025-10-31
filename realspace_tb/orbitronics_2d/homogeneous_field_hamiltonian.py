from ..hamiltonian import Hamiltonian
from .lattice_2d_geometry import Lattice2DGeometry
from .. import backend as B
from abc import ABC, abstractmethod
import numpy as np


class HomogeneousFieldAmplitude(ABC):
    """Abstract base class for homogeneous electric field with only scalar time dependence."""

    @abstractmethod
    def at_time(self, time: "float | B.Array") -> "float | B.Array":
        """Return the electric field amplitude at a given time."""
        ...

    direction: B.FCPUArray = np.zeros(2)


class RampedACFieldAmplitude(HomogeneousFieldAmplitude):
    """
    Electric field amplitude ramping over time:
    E(t) = E0 * sin^2(pi * t / 2 * T_ramp) * sin(ω t), capped at E0.
    """

    def __init__(self, E0: float, omega: float, T_ramp: float, direction: B.FCPUArray):
        self.E0 = B.FDTYPE(E0)
        self.omega = B.FDTYPE(omega)
        self.ramp_time = B.FDTYPE(T_ramp)
        self.direction = B.FDTYPE(direction)

    def at_time(self, t: "float | B.Array") -> "float | B.Array":
        # TODO maybe make it CPU only and move backend transfer to Hamiltonian class
        # -> avoid confusion and minimize code scope of GPU backend
        xp = B.xp()
        if xp.isscalar(t):
            if t < self.ramp_time:
                ramp = xp.sin(np.pi * t / (2 * self.ramp_time)) ** 2
            else:
                ramp = 1.0
            return self.E0 * ramp * xp.sin(self.omega * t)

        ramp = xp.where(
            t < self.ramp_time,
            xp.sin(xp.pi * t / (2 * self.ramp_time)) ** 2,
            xp.ones_like(t, dtype=B.FDTYPE),
        )
        return self.E0 * ramp * xp.sin(self.omega * t)


class LinearFieldHamiltonian(Hamiltonian):
    def __init__(
        self, geometry: Lattice2DGeometry, field_amplitude: HomogeneousFieldAmplitude
    ):
        super().__init__()

        self.geometry = geometry
        self.field_amplitude = field_amplitude

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
        self.H_0 = (
            B.xp_sparse()
            .coo_matrix(
                (
                    B.xp().array(data),
                    (B.xp().array(row_indices), B.xp().array(col_indices)),
                ),
                shape=(size, size),
                dtype=B.FDTYPE,
            )
            .tocsr()
        )

        # make a sparse matrix diag(r_i . E) for position operator along field direction
        position_shifts = B.xp().array(
            [
                np.dot(geometry.index_to_position(index), field_amplitude.direction)
                for index in range(size)
            ],
            dtype=B.FDTYPE,
        )
        position_shifts -= B.xp().mean(position_shifts)  # center around zero

        self.position_operator = B.xp_sparse().diags(
            position_shifts, format="csr", dtype=B.FDTYPE
        )

    def at_time(self, t: float) -> B.SparseArray:
        # Implementation of the Hamiltonian at time t
        return self.H_0 + self.field_amplitude.at_time(t) * self.position_operator
