from abc import ABC, abstractmethod
from . import backend as B

class Hamiltonian(ABC):
    def __init__(self) -> None:
        self._eigenvalues: "B.Array | None" = None
        self._eigenstates: "B.Array | None" = None

    @abstractmethod
    def at_time(self, t: float) -> B.SparseArray:
        """Return the Hamiltonian at time t."""
        ...

    @property
    def eigenvalues(self) -> B.Array:
        """Return the eigenvalues of the Hamiltonian at t=0."""
        if self._eigenvalues is None:
            print("Calculating eigenvalues at t=0...")
            self._eigenvalues, self._eigenstates = B.xp().linalg.eigh(self.at_time(0.0).toarray())
        return self._eigenvalues

    @property
    def eigenstates(self) -> B.Array:
        """Return the eigenstates of the Hamiltonian at t=0."""
        if self._eigenstates is None:
            print("Calculating eigenstates at t=0...")
            self._eigenvalues, self._eigenstates = B.xp().linalg.eigh(self.at_time(0.0).toarray())
        return self._eigenstates

    def ground_state_density_matrix(self, fermi_level: float=0.0) -> B.Array:
        """Return the ground state density matrix. Calculates eigenstates from H_0 (assumed as H at t=0 here).
        Returns \sum_n |psi_n><psi_n| for all eigenstates with energy E_n <= fermi_level.
        """
        rho_energy_basis = B.xp().diag([1 if self.eigenvalues[i] <= fermi_level else 0 for i in range(self.eigenvalues.shape[0])])
        return (self.eigenstates @ rho_energy_basis @ self.eigenstates.T.conj()).astype(B.DTYPE, copy=False)