from abc import ABC, abstractmethod
from .backend import Array, SparseArray, CPUArray


class Observable(ABC):
    def __init__(
        self,
        measurement_start_time: float = 0.0,
        measurement_end_time: float = float("inf"),
        measurement_stride: int = 1,
    ) -> None:
        """Base class for observables measured during time evolution. Default setup uses start time,
        end time and stride to determine when measurements are taken."""
        self._start_time = measurement_start_time
        self._end_time = measurement_end_time
        self._stride = measurement_stride


    def setup(self, dt: float, total_steps: int) -> int:
        """Setup the observable for measurement. Called by the numeric solver before integrating.
        Is needed since the resource allocation can depend on the number of measurements which depends on dt,
        which might differ between solver runs.
        Default implementation adjusts the end time and computes the number of measurements.
        
        Parameters:
            dt: time step of the numeric solver
            total_steps: total number of time steps of the numeric solver
            
        Returns:
            number of measurements to be performed, possibly off by one.
        """
        self._end_time = min(self._end_time, dt * total_steps)
        return int((self._end_time - self._start_time) / (dt * self._stride)) + 1
    
    def _should_measure(self, t: float, step_index: int) -> bool:
        """Check whether a measurement should be performed at the given time and step index."""
        return self._start_time <= t <= self._end_time and step_index % self._stride == 0

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
