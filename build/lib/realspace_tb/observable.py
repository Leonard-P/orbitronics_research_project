from abc import ABC, abstractmethod
from .backend import Array, SparseArray, CPUArray

class Observable(ABC):
    @abstractmethod
    def setup(self, dt: float, total_steps: int) -> None:
        """Setup the observable for measurement. Called by the numeric solver before integrating. 
        Is needed since the resource allocation can depend on the number of measurements which depends on dt,
        which might differ between solver runs.
        """
        ...

    @abstractmethod
    def measure(self, rho: Array, t: float, step_index: int) -> None:
        """Measure the observable given the density matrix."""
        ...

    def finalize(self) -> None:
        """Finalize measurement (e.g. post-process stored data or move from GPU to CPU). 
        Called by the numeric solver after integrating.
        """
        ...

    values: "CPUArray | dict[str, CPUArray]"
    measurement_times: CPUArray