from abc import ABC, abstractmethod
from .backend import Array, SparseArray, xp, DTYPE

class Hamiltonian(ABC):
    def __init__(self) -> None:
        self._eigenvalues: "Array | None" = None
        self._eigenstates: "Array | None" = None

    @abstractmethod
    def at_time(self, t: float) -> SparseArray:
        """Return the Hamiltonian at time t."""
        ...

    @property
    def eigenvalues(self) -> Array:
        """Return the eigenvalues of the Hamiltonian at t=0."""
        if self._eigenvalues is None:
            print("Calculating eigenvalues at t=0...")
            self._eigenvalues, self._eigenstates = xp().linalg.eigh(self.at_time(0.0).toarray())
        return self._eigenvalues

    @property
    def eigenstates(self) -> Array:
        """Return the eigenstates of the Hamiltonian at t=0."""
        if self._eigenstates is None:
            print("Calculating eigenstates at t=0...")
            self._eigenvalues, self._eigenstates = xp().linalg.eigh(self.at_time(0.0).toarray())
        return self._eigenstates

    def ground_state_density_matrix(self, fermi_level: float=0.0) -> Array:
        """Return the ground state density matrix. Calculates eigenstates from H_0 (assumed as H at t=0 here).
        Returns \sum_n |psi_n><psi_n| for all eigenstates with energy E_n <= fermi_level.
        """
        rho_energy_basis = xp().diag([1 if self.eigenvalues[i] <= fermi_level else 0 for i in range(self.eigenvalues.shape[0])])
        return (self.eigenstates @ rho_energy_basis @ self.eigenstates.T.conj()).astype(DTYPE, copy=False)