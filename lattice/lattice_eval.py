from dataclasses import dataclass
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from .lattice import Lattice2D
from  .lattice_geometry import LatticeGeometry


class SimulationData:
    def __init__(self, lattice: Lattice2D, omega: float = 1.0, cutoff_freq: float = float("inf")):
        self.t = np.array([t * lattice.h for t in range(len(lattice.states))])
        self.dt = lattice.h
        self.E = lattice.E(self.t)
        self.main_freq = omega

        P, P_orb = [], []
        self.P_rows, self.P_cols, self.P_orb_rows, self.P_orb_cols = [], [], [], []

        for state in tqdm(lattice.states):
            lattice_state = lattice.compute_lattice_state(state)
            P.append(lattice_state.polarisation[1])  # TODO support arbitrary polarisation projection
            P_orb.append(lattice_state.orbital_charge_polarisation[0])

            P_row, P_col = self._polarisation_rows_cols(state, lattice.geometry)
            self.P_rows.append(P_row)
            self.P_cols.append(P_col)

            P_orb_row, P_orb_col = SimulationData._orbital_polarisation_rows_cols(lattice_state.orbital_charges, lattice.geometry)
            self.P_orb_rows.append(P_orb_row)
            self.P_orb_cols.append(P_orb_col)

        self.P = np.array(P)
        self.P_orb = np.array(P_orb)

    @staticmethod
    def _get_fft_range(t_signal: np.ndarray, dt: float, max_freq: float = float('inf')) -> tuple[list[float], list[float]]:
        """Computes DFT of t_signal and returns amplitudes for frequencies below max_freq."""
        freqs = np.fft.fftfreq(len(t_signal), d=dt)
        mask = (freqs < max_freq) & (freqs > 0)
        return freqs[mask], np.abs(np.fft.fft(t_signal))[mask]

    @staticmethod
    def _polarisation_rows_cols(
        state_matrix: np.ndarray, lattice_geometry: LatticeGeometry
    ) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray]]:
        P_rows = dict[float, float]()
        P_cols = dict[float, float]()
        for i, n in enumerate(state_matrix.diagonal().real):
            x, y = lattice_geometry.site_to_position(i)
            # P_rows[round(y, 5)] = P_rows.get(round(y, 5), 0.0) + (x - lattice_geometry.origin[0]) * n
            # P_cols[round(x, 5)] = P_cols.get(round(x, 5), 0.0) + (y - lattice_geometry.origin[1]) * n
            P_rows[round(y, 5)] = P_rows.get(round(y, 5), 0.0) + abs(n)
            P_cols[round(x, 5)] = P_cols.get(round(x, 5), 0.0) + abs(n)
        return P_rows, P_cols

    @staticmethod
    def _orbital_polarisation_rows_cols(
        orbital_charges: dict[int, float], lattice_geometry: LatticeGeometry
    ) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray]]:
        P_orb_rows = dict[float, float]()
        P_orb_cols = dict[float, float]()
        for i, n in orbital_charges.items():
            x, y = lattice_geometry.site_to_position(i)
            # P_orb_rows[round(y, 5)] = P_orb_rows.get(round(y, 5), 0.0) + (x - lattice_geometry.curl_origin[0]) * n
            # P_orb_cols[round(x, 5)] = P_orb_cols.get(round(x, 5), 0.0) + (y - lattice_geometry.curl_origin[1]) * n
            P_orb_rows[round(y, 5)] = P_orb_rows.get(round(y, 5), 0.0) + abs(n)
            P_orb_cols[round(x, 5)] = P_orb_cols.get(round(x, 5), 0.0) + abs(n)

        return P_orb_rows, P_orb_cols


    def plot_simulation_time_series(self, show_window: int = None) -> tuple[plt.Figure, plt.Axes]:
        """Plot time series data from simulation."""
        if show_window:
            t_slice = slice(-show_window, None)
        else:
            t_slice = slice(None)

        P_current = np.pad(np.diff(self.P), pad_width=(1, 0), mode='edge') / self.dt
        P_orb_current = np.pad(np.diff(self.P_orb), pad_width=(1, 0), mode='edge') / self.dt

        plot_data = {
            "$P_y(t) (e a^{-1})$": self.P,
            "$\\frac{\\partial P_y}{\\partial t}~(e t_{\\rm hop} a^{-1} \\hbar^{-1})$": P_current,
            # "$\\nabla q_{\\rm orb}(t)~(e a^{-1})$": self.M_grad[:, 0],
            "$P_{\\rm orb, x}(t)~(e a^{-1})$": self.P_orb,
            "$\\frac{\\partial P_{\\rm orb, x}(t)}{\\partial t}~(e t_{\\rm hop} a^{-1} \\hbar^{-1})$": P_orb_current,
        }

        fig, axs = plt.subplots(len(plot_data), 1, figsize=(10, len(plot_data) * 3))
        cmap = plt.get_cmap("viridis")

        for i, (label, data) in enumerate(plot_data.items()):
            axs[i].plot(self.t[t_slice], data[t_slice], label=label, color=cmap(i / len(plot_data)))
            axs[i].set_ylabel(label)

            norm = np.max(np.abs(data)) / np.max(np.abs(self.E))
            axs[i].plot(self.t[t_slice], self.E[t_slice] * norm, label="$E(t)$", color="tab:blue", alpha=0.2)  # ~(t_{\\rm hop} a^{-1} e^{-1})
            axs[i].legend(loc="upper right")

        axs[-1].set_xlabel("t")

        plt.tight_layout(pad=2)
        return fig, axs

    def plot_simulation_fft(self, cutoff_freq: float = float('inf')) -> tuple[plt.Figure, plt.Axes]:
        """Plot FFT data from simulation."""

        freqs, P_fft = SimulationData._get_fft_range(self.P, self.dt, cutoff_freq)
        _, E_fft = SimulationData._get_fft_range(self.E, self.dt, cutoff_freq)
        _, P_orb_fft = SimulationData._get_fft_range(self.P_orb, self.dt, cutoff_freq)

        freqs /= self.main_freq / (2 * np.pi)

        plot_data = {
            "FFT of $E$": E_fft,
            "FFT of $P_y$": P_fft,
            "FFT of $P_{\\rm orb, x}$": P_orb_fft,
        }

        fig, axs = plt.subplots(len(plot_data), 1, figsize=(10, len(plot_data) * 3))

        for i, (label, data) in enumerate(plot_data.items()):
            axs[i].plot(freqs, data, ".", label=label)
            axs[i].set_ylabel("Amplitude")
            axs[i].set_title(label)
        
        axs[-1].set_xlabel("Frequency / $\\omega$")

        plt.tight_layout()
        return fig, axs