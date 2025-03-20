from dataclasses import dataclass
from matplotlib import pyplot as plt
import numpy as np

from lattice import Lattice2D

@dataclass
class SimulationData:
    t: np.ndarray
    E: np.ndarray
    P: np.ndarray
    M: np.ndarray
    P_current: np.ndarray
    M_current: np.ndarray
    freqs: np.ndarray
    E_fft: np.ndarray
    P_fft: np.ndarray
    M_fft: np.ndarray
    P_current_fft: np.ndarray
    M_current_fft: np.ndarray
    M_grad: np.ndarray = None
    M_grad_fft: np.ndarray = None

    @staticmethod
    def _get_fft_range(t_signal: np.ndarray, max_freq: float, h: float) -> tuple[list[float], list[float]]:
        """Computes DFT of t_signal and returns amplitudes for frequencies below max_freq."""
        freqs = np.fft.fftfreq(len(t_signal), d=h)
        mask = (freqs < max_freq) & (freqs > 0)
        return freqs[mask], np.abs(np.fft.fft(t_signal))[mask]

    @classmethod
    def from_lattice(cls, l: Lattice2D, omega: float, cutoff_freq=None) -> None:
        """Returns time series and FFT data for a simulated lattice."""
        t = np.arange(0, (l.steps + 1) * l.h, l.h)

        E = np.cos(omega * t)
        P = [state.polarisation[1] for state in l.states]
        P_current = np.diff(P) / l.h
        M = [state.orbital_charge_polarisation[0] for state in l.states]
        M_current = np.diff(M) / l.h

        # P /= np.max(np.abs(P))
        # P_current /= np.max(np.abs(P_current))
        # M /= np.max(np.abs(M))
        # M_current /= np.max(np.abs(M_current))

        if cutoff_freq is None:
            cutoff_freq = 10 * omega

        freqs, P_fft = SimulationData._get_fft_range(P, cutoff_freq, l.h)
        _, E_fft = SimulationData._get_fft_range(E, cutoff_freq, l.h)
        _, M_fft = SimulationData._get_fft_range(M, cutoff_freq, l.h)

        _, P_current_fft = SimulationData._get_fft_range(P_current, cutoff_freq, l.h)
        _, M_current_fft = SimulationData._get_fft_range(M_current, cutoff_freq, l.h)

        freqs /= omega / (2 * np.pi)

        M_grad = []
        for state in l.states:
            gradient = l.geometry.cell_field_gradient(state.orbital_charges)
            M_grad.append(np.mean(list(gradient.values()), axis=0))

        M_grad = np.array(M_grad)
        _, M_grad_fft = SimulationData._get_fft_range(M_grad[:, 0], cutoff_freq, l.h)

        return cls(
            t=t,
            E=E,
            P=P,
            M=M,
            P_current=P_current,
            M_current=M_current,
            freqs=freqs,
            E_fft=E_fft,
            P_fft=P_fft,
            M_fft=M_fft,
            P_current_fft=P_current_fft,
            M_current_fft=M_current_fft,
            M_grad=M_grad,
            M_grad_fft=M_grad_fft,
        )

    def plot_simulation_time_series(self, show_window: int = None) -> tuple[plt.Figure, plt.Axes]:
        """Plot time series data from simulation."""
        fig, axs = plt.subplots(5, 1, figsize=(10, 14))

        if show_window:
            t_slice = slice(-show_window, None)
        else:
            t_slice = slice(None)

        plot_data = {
            "$P_x(t) (e a^{-1})$": self.P,
            "$\\frac{\\partial P_x}{\\partial t}~(e t_{\\rm hop} a^{-1} \\hbar^{-1})$": np.concatenate((self.P_current, [self.P_current[-1]])),
            "$\\nabla q_{\\rm orb}(t)~(e a^{-1})$": self.M_grad[:, 0],
            "$P_{\\rm orb, y}(t)~(e a^{-1})$": self.M,
            "$\\frac{\\partial P_{\\rm orb, y}(t)}{\\partial t}~(e t_{\\rm hop} a^{-1} \\hbar^{-1})$": np.concatenate((self.M_current, [self.M_current[-1]])),
        }

        cmap = plt.get_cmap("viridis")

        for i, (label, data) in enumerate(plot_data.items()):
            axs[i].plot(self.t[t_slice], data[t_slice], label=label, color=cmap(i / len(plot_data)))
            axs[i].set_ylabel(label)

            norm = np.max(np.abs(data))
            axs[i].plot(self.t[t_slice], self.E[t_slice]*norm, label="$E(t)$", color="tab:blue", alpha=0.2) # ~(t_{\\rm hop} a^{-1} e^{-1})
            axs[i].legend(loc='upper right')

        # axs[0].plot(self.t[t_slice], self.P[t_slice], label="$P_x(t)$", color="tab:green")
        # axs[0].set_ylabel("P(t)")

        # axs[1].plot(self.t[:-1][t_slice], self.P_current[t_slice], label="$dP_x/dt$", color="tab:purple")
        # axs[1].set_ylabel("dP/dt")

        # axs[2].plot(self.t[t_slice], self.M[t_slice], label="$P_{\\rm orb, y}(t)$", color="tab:red")
        # axs[2].set_ylabel("$P_{\\rm orb, y}(t)$")

        # axs[3].plot(self.t[:-1][t_slice], self.M_current[t_slice], label="$dP_{\\rm orb, y}(t)/dt$", color="tab:orange")
        # axs[3].set_ylabel("dM/dt")

        # axs[4].plot(self.t[t_slice], self.M_grad[:, 0][t_slice], label="$\\nabla q_{\\rm orb}(t)$", color="tab:red")
        # axs[4].set_ylabel("$\\nabla q_{\\rm orb}(t)$")

        axs[-1].set_xlabel("t")

        plt.tight_layout(pad=2)
        return fig, axs

    def plot_simulation_fft(self, cutoff_freq: float = None):
        """Plot FFT data from simulation."""
        fig, axs = plt.subplots(4, 1, figsize=(10, 12))

        max_freq = cutoff_freq if cutoff_freq is not None else data.freqs[-1]

        mask = self.freqs <= max_freq

        axs[0].bar(self.freqs[mask], self.E_fft[mask], color="tab:blue")
        axs[0].set_title("FFT of $E$")
        axs[0].set_ylabel("Amplitude")

        axs[1].bar(self.freqs[mask], self.P_fft[mask], color="tab:green")
        axs[1].set_title("FFT of $P_x$")
        axs[1].set_ylabel("Amplitude")

        axs[2].bar(self.freqs[mask], self.M_fft[mask], color="tab:red")
        axs[2].set_title("FFT of $P_{\\rm orb, y}(t)$")
        axs[2].set_ylabel("Amplitude")

        axs[3].bar(self.freqs[mask], self.M_grad_fft[mask], color="tab:green")
        axs[3].set_title("FFT of $\\nabla q_{\\rm orb}(t)$")
        axs[3].set_ylabel("Amplitude")

        axs[3].set_xlabel("Frequency / $\\omega$")

        plt.tight_layout()
        return fig, axs


if __name__ == "__main__":
    import sys
    import lattice
    import lattice_geometry
    sys.modules['lattice.lattice'] = lattice
    sys.modules['lattice.lattice_geometry'] = lattice_geometry

    def E(t):
        return np.sin(omega*t)

    omega = 2*np.pi / 3.5

    data = SimulationData.from_lattice(Lattice2D.load("results/brickwall_7x14_omega_3-5.lattice"), omega=omega)
    data.plot_simulation_time_series()
    
    plt.show()