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


def get_fft_range(t_signal: np.ndarray, max_freq: float, h: float) -> tuple[list[float], list[float]]:
    """Computes DFT of t_signal and returns amplitudes for frequencies below max_freq."""
    freqs = np.fft.fftfreq(len(t_signal), d=h)
    mask = (freqs < max_freq) & (freqs > 0)
    return freqs[mask], np.abs(np.fft.fft(t_signal))[mask]


def get_simulation_data(l: Lattice2D, omega: float) -> SimulationData:
    """Returns time series and FFT data for a simulated lattice."""
    t = np.arange(0, l.steps * l.h, l.h)

    E = np.cos(omega * t)
    P = [state.polarisation[1] for state in l.states]
    P_current = np.diff(P) / l.h
    M = [state.curl_polarisation[0] for state in l.states]
    M_current = np.diff(M) / l.h

    P /= np.max(np.abs(P))
    P_current /= np.max(np.abs(P_current))
    M /= np.max(np.abs(M))
    M_current /= np.max(np.abs(M_current))

    cutoff_freq = 10 * omega
    freqs, P_fft = get_fft_range(P, cutoff_freq, l.h)
    _, E_fft = get_fft_range(E, cutoff_freq, l.h)
    _, M_fft = get_fft_range(M, cutoff_freq, l.h)

    _, P_current_fft = get_fft_range(P_current, cutoff_freq, l.h)
    _, M_current_fft = get_fft_range(M_current, cutoff_freq, l.h)

    freqs /= omega / (2 * np.pi)

    return SimulationData(
        t=t, E=E, P=P, M=M, P_current=P_current, M_current=M_current,
        freqs=freqs, E_fft=E_fft, P_fft=P_fft, M_fft=M_fft, P_current_fft=P_current_fft, M_current_fft=M_current_fft
    )


def plot_simulation_time_series(data: SimulationData, show_window: int = None):
    """Plot time series data from simulation."""
    fig, axs = plt.subplots(5, 1, figsize=(10, 12))
    
    if show_window:
        t_slice = slice(-show_window, None)
    else:
        t_slice = slice(None)
    
    axs[0].plot(data.t[t_slice], data.E[t_slice], label="E(t)", color="tab:blue")
    axs[0].set_ylabel("E(t)")
    
    axs[1].plot(data.t[t_slice], data.P[t_slice], label="P(t)", color="tab:green")
    axs[1].set_ylabel("P(t)")
    
    axs[2].plot(data.t[:-1][t_slice], data.P_current[t_slice], label="dP/dt", color="tab:purple")
    axs[2].set_ylabel("dP/dt")
    
    axs[3].plot(data.t[t_slice], data.M[t_slice], label="M(t)", color="tab:red")
    axs[3].set_ylabel("M(t)")
    
    axs[4].plot(data.t[:-1][t_slice], data.M_current[t_slice], label="dM/dt", color="tab:orange")
    axs[4].set_ylabel("dM/dt")
    axs[4].set_xlabel("t")
    
    for ax in axs:
        ax.legend()
    
    plt.tight_layout()
    return fig, axs


def plot_simulation_fft(data: SimulationData, cutoff_freq: float = None):
    """Plot FFT data from simulation."""
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    max_freq = cutoff_freq if cutoff_freq is not None else data.freqs[-1]
    mask = data.freqs <= max_freq
    
    axs[0].bar(data.freqs[mask], data.E_fft[mask], color='green')
    axs[0].set_title('FFT of E')
    axs[0].set_ylabel('Amplitude')
    
    axs[1].bar(data.freqs[mask], data.P_fft[mask], color='blue')
    axs[1].set_title('FFT of P')
    axs[1].set_ylabel('Amplitude')
    
    axs[2].bar(data.freqs[mask], data.M_fft[mask], color='red')
    axs[2].set_title('FFT of M')
    axs[2].set_ylabel('Amplitude')
    axs[2].set_xlabel('Frequency / $\\omega$')
    
    plt.tight_layout()
    return fig, axs
       