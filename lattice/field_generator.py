import numpy as np
from numpy.typing import NDArray
from typing import Callable

VectorizedFieldAmplitude = Callable[[NDArray[np.float64]], NDArray[np.float64]]


class FieldAmplitudeGenerator:
    """
    Provides static methods to generate common vectorized field amplitude callables.
    """

    @staticmethod
    def _smooth_sine_squared_vectorized(t_array: NDArray[np.float64], width: float, dtype=np.float64) -> NDArray[np.float64]:
        """
        Internal helper: Vectorized smooth step ramp-up using sine squared.
        Returns 0 for t < 0, sin^2(pi*t/(2*width)) for 0 <= t <= width, and 1 for t > width.
        Handles width=0 case (becomes a step function at t=0).
        """
        t_array = np.asarray(t_array, dtype=dtype)

        if width <= 0:
            # Step function at t=0
            return np.where(t_array > 0, dtype(1.0), dtype(0.0)).astype(dtype)

        # Conditions
        in_ramp = (t_array >= 0) & (t_array <= width)
        past_ramp = t_array > width

        sine_arg = np.zeros_like(t_array, dtype=dtype)
        sine_arg[in_ramp] = (np.pi * t_array[in_ramp]) / (2.0 * width)
        ramp_values = np.sin(sine_arg) ** 2

        # Select output based on conditions
        conditions = [past_ramp, in_ramp]
        choices = [dtype(1.0), ramp_values]
        result = np.select(conditions, choices, default=dtype(0.0))

        return result.astype(dtype)  # Ensure final dtype

    @staticmethod
    def constant(amplitude: float) -> VectorizedFieldAmplitude:
        """
        Generates a callable for a constant field amplitude.

        Args:
            amplitude: The constant amplitude value.

        Returns:
            A vectorized callable: f(t_array) -> constant amplitude array.
        """
        amp_val = float(amplitude)

        def constant_field(t_array: NDArray[np.float64]) -> NDArray[np.float64]:
            # Return an array of the same shape as t_array filled with the value
            return np.full_like(t_array, fill_value=amp_val, dtype=t_array.dtype)

        return constant_field

    @staticmethod
    def oscillation(amplitude: float, omega: float) -> VectorizedFieldAmplitude:
        """
        Generates a callable for a simple sinusoidal oscillation: A * sin(omega*t + phase).

        Args:
            amplitude (A): The peak amplitude of the oscillation.
            omega: The angular frequency (rad/time).

        Returns:
            A vectorized callable: f(t_array) -> A * sin(omega*t_array + phase).
        """
        amp_val = float(amplitude)
        omega_val = float(omega)

        def oscillating_field(t_array: NDArray[np.float64]) -> NDArray[np.float64]:
            return amp_val * np.sin(omega_val * t_array)

        return oscillating_field

    @staticmethod
    def ramped_oscillation(amplitude: float, omega: float, ramp_width: float) -> VectorizedFieldAmplitude:
        """
        Generates a callable for an oscillation with a smooth sine-squared turn-on ramp.
        Amplitude ramps from 0 to A over the interval [0, ramp_width].

        Args:
            amplitude (A): The peak amplitude after the ramp.
            omega: The angular frequency (rad/time).
            ramp_width: The duration of the initial ramp-up (time).

        Returns:
            A vectorized callable implementing the ramped oscillation.
        """
        amp_val = float(amplitude)
        omega_val = float(omega)
        width_val = float(ramp_width)

        # Combine ramp and oscillation
        def ramped_oscillating_field(t_array: NDArray[np.float64]) -> NDArray[np.float64]:
            # Ensure input is array and correct dtype for ramp function
            t_array_float = np.asarray(t_array, dtype=np.float64)
            ramp_factor = FieldAmplitudeGenerator._smooth_sine_squared_vectorized(t_array_float, width_val, dtype=np.float64)
            oscillation = amp_val * np.sin(omega_val * t_array_float)

            return (ramp_factor * oscillation).astype(t_array.dtype if isinstance(t_array, np.ndarray) else np.float64)

        return ramped_oscillating_field

    @staticmethod
    def gaussian_pulse(amplitude: float, center_time: float, width: float) -> VectorizedFieldAmplitude:
        """
        Generates a callable for a Gaussian pulse: A * exp(-0.5 * ((t - t0) / width)^2).

        Args:
            amplitude (A): The peak amplitude at the center of the pulse.
            center_time (t0): The time at which the pulse peaks.
            width (sigma): The standard deviation, controlling the pulse width.
                           Must be positive.

        Returns:
            A vectorized callable implementing the Gaussian pulse.
        """
        if width <= 0:
            raise ValueError("Pulse width (sigma) must be positive for Gaussian pulse.")

        amp_val = float(amplitude)
        t0 = float(center_time)
        sigma = float(width)

        def gaussian_pulse_field(t_array: NDArray[np.float64]) -> NDArray[np.float64]:
            exponent = -0.5 * ((t_array - t0) / sigma) ** 2
            return amp_val * np.exp(exponent)

        return gaussian_pulse_field