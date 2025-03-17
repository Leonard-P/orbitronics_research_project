import warnings
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Union, Callable, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import qutip as qu
from scipy import linalg
from tqdm import tqdm
from numpy.typing import NDArray
from .lattice_geometry import LatticeGeometry, RectangularLatticeGeometry, SquareLatticeGeometry, BrickwallLatticeGeometry
from . import lattice_utils as utils
from . import lattice_rk4 as rk4

# Units a=1, h_bar=1 and e=1, t_hop=1
# [P] = e*a/a**2


@dataclass
class LatticeState:
    density: NDArray[np.complex64]
    current: NDArray[np.float64]
    polarisation: NDArray[np.float64]
    curl: Dict[int, float]
    curl_polarisation: NDArray[np.float64]


@dataclass
class SimulationParameters:
    t_hop: float
    E_amplitude: Union[float, Callable[[float], float]]
    E_direction: np.ndarray
    h: float
    T: float
    charge: int = 1
    initial_occupation: float = 0.5

    @property
    def simulation_n_steps(self):
        return int(self.T / self.h)


class Lattice2D:
    def __init__(
        self,
        geometry: LatticeGeometry,
        simulation_parameters: SimulationParameters,
        origin: Optional[Tuple[float, float]] = None,
    ):
        self.geometry = geometry
        self.Lx, self.Ly = geometry.dimensions
        self.N = self.Lx * self.Ly

        if self.N % 2:
            warnings.warn(
                "Site number should be even--then integer # of states can be set with half occupation.",
            )

        self.simulation_parameters = simulation_parameters

        print("Initialize Hamiltonian and eigenstates...", end=" ")
        self.H_onsite = self.create_onsite_potentials()
        self.H_hop = self.create_hopping_hamiltonian()
        self.eigen_energies, self.energy_states = linalg.eigh(self.H_hop)
        print("Done.")

        self.states: list[LatticeState] = None

        if simulation_parameters.initial_occupation:
            self.set_fractional_occupation(simulation_parameters.initial_occupation)
        else:
            self.density_matrix = np.zeros((self.N, self.N), dtype=complex)

        if origin is not None:
            self.origin = np.array(origin)
        else:
            self.origin = (np.array([self.Lx, self.Ly]) - 1) / 2  # center of lattice. Polarisation is dependent of origin.

    # Properties to access simulation parameters
    @property
    def t_hop(self) -> float:
        return self.simulation_parameters.t_hop

    @property
    def E_dir(self) -> np.ndarray:
        return self.simulation_parameters.E_direction

    @property
    def h(self) -> float:
        return self.simulation_parameters.h

    @property
    def steps(self) -> int:
        return self.simulation_parameters.simulation_n_steps

    @property
    def curl_origin(self):
        return self.origin - np.array([self.geometry.cell_width, self.geometry.cell_height]) / 2

    def E(self, t) -> Union[float, Callable]:
        if callable(self.simulation_parameters.E_amplitude):
            return self.simulation_parameters.E_amplitude(t)

        return self.simulation_parameters.E_amplitude

    def create_hopping_hamiltonian(self) -> np.ndarray:
        """Create the hopping Hamiltonian based on the geometry's hopping matrix"""
        return self.t_hop * self.geometry.get_hopping_matrix()

    def create_onsite_potentials(self) -> np.ndarray:
        """Create the on-site potentials based on the site positions"""
        potentials = [-np.dot(self.geometry.site_to_position(i), self.E_dir) for i in range(self.N)]  # - E . r
        potentials -= np.mean(potentials)  # center around 0
        return np.diag(potentials)

    def H(self, time: float) -> np.ndarray:
        """Get the Hamiltonian at a specific time"""
        return self.H_hop + self.E(time) * self.H_onsite

    def set_fractional_occupation(self, occupation_fraction=0.5) -> None:
        rho_energy_basis = np.diag([1 if i / self.N < occupation_fraction else 0 for i in range(self.N)])
        self.density_matrix = 1 / (occupation_fraction * self.N) * self.energy_states @ rho_energy_basis @ self.energy_states.T.conj()
        self.occupation_fraction = occupation_fraction

        print(f"Occupation set to {rho_energy_basis.trace()/self.N:.2f}.")

    def evolve(self, force_reevolve=False, solver="qutip", **solver_kwargs) -> None:
        """Evolve the system over time and compute all derived quantities"""
        if self.states is not None and not force_reevolve:
            print("Lattice was already evolved, call with force_reevolve=True to simulate again.")
            return
        
        if solver == "qutip":
            H = [qu.Qobj(self.H_hop), [qu.Qobj(self.H_onsite), self.E]]
            rho = qu.Qobj(self.density_matrix)
            step_list = np.linspace(0, self.h * self.steps, self.steps)
            sim = qu.mesolve(H, rho, step_list, **solver_kwargs)

            # Compute all derived quantities for each time step
            self.states = [self.compute_lattice_state(state.data_as(format="ndarray")) for state in sim.states]
        elif solver == "rk4":
            sim = rk4.evolve_density_matrix_rk4(self.H_hop, self.H_onsite, self.density_matrix, self.E, self.h, self.simulation_parameters.T, **solver_kwargs)
            self.states = [self.compute_lattice_state(state) for state in sim]

    def compute_lattice_state(self, density_matrix: np.ndarray) -> LatticeState:
        """Compute all derived quantities from the density matrix"""
        current_matrix = self._current_density(density_matrix)
        polarisation = self._polarisation(density_matrix)
        curl = self._curl(current_matrix)
        curl_polarisation = self._curl_polarisation(curl)

        return LatticeState(
            density=density_matrix,
            current=current_matrix,
            polarisation=polarisation,
            curl=curl,
            curl_polarisation=curl_polarisation,
        )

    def _current_density(self, state_matrix: np.ndarray) -> np.ndarray:
        """Calculate current density from the density matrix"""
        return 2 * (self.H_hop * state_matrix.T).imag  # [J] = e/[t] = h_bar e/t_hop

    def _curl(self, J: np.ndarray) -> Dict[int, float]:
        """Calculate curl around each unit cell using the geometry's curl sites"""
        cell_area = self.geometry.cell_width * self.geometry.cell_height
        return {
            site_idx: sum([J[site_idx + di, site_idx + dj] for di, dj in self.geometry.cell_path]) / cell_area
            for site_idx in self.geometry.get_curl_sites()
        }  # [curl J] = [J]/a**2 = h_bar e/a**2 t_hop

    def _polarisation(self, state_matrix: np.ndarray) -> np.ndarray:
        """Calculate polarization from density matrix"""
        return (
            np.sum([(np.array(self.geometry.site_to_position(i)) - self.origin) * state_matrix.diagonal().real[i] for i in range(self.N)], axis=0)
            / self.N
        )  # [P] = e*a/a**2 = e/a

    def _curl_polarisation(self, curl_J: Dict[int, float]) -> np.ndarray:
        """Calculate curl of polarization"""
        return np.sum(
            [(np.array(self.geometry.site_to_position(site_index)) - self.curl_origin) * curl_val for site_index, curl_val in curl_J.items()], axis=0
        )

    def _polarisation_current(self, state_matrix: np.ndarray, previous_step_state_matrix: np.ndarray) -> np.ndarray:
        """Calculate time derivative of polarization"""
        return (self._polarisation(state_matrix) - self._polarisation(previous_step_state_matrix)) / self.h

    def _curl_polarisation_current(self, curl_J: Dict[int, float], previous_curl_J: Dict[int, float]) -> np.ndarray:
        """Calculate time derivative of curl polarization"""
        return (self._curl_polarisation(curl_J) - self._curl_polarisation(previous_curl_J)) / self.h

    def plot_current_density(
        self,
        state_index: int,
        ax: Optional[matplotlib.axes] = None,
        curl_norm: float = 1,
        E_norm: float = 1,
        curl_pol_norm: float = 1,
        pol_norm: float = 1,
        auto_normalize: bool = False,
    ) -> None:
        show_plot = ax is None

        if self.states is None:
            lattice_state = self.compute_lattice_state(self.density_matrix)
        else:
            lattice_state = self.states[state_index]

        if auto_normalize:
            curl_norm = max([max(np.abs(list(state.curl.values()))) for state in self.states])
            E_norm = max([np.linalg.norm(self.E(i * self.h)) for i in range(self.steps)])
            pol_norm = max([np.linalg.norm(state.polarisation) for state in self.states])
            curl_pol_norm = max([np.linalg.norm(state.curl_polarisation) for state in self.states])

        if ax is None:
            _, ax = plt.subplots(figsize=(2 * self.Lx + 2, 2 * self.Ly))
        else:
            ax.clear()

        site_values = np.real(np.diag(lattice_state.density)) * self.N * self.occupation_fraction
        utils.plot_site_grid(site_values, self.Lx, self.Ly, ax, vmin=0.0, vmax=1, cmap_name="Greys")
        utils.plot_site_connections(
            lattice_state.current, self.Lx, self.Ly, ax, max_flow=np.max(np.abs(lattice_state.current)), label_connection_strength=False
        )

        # Plot curl indicators
        for site_idx, curl_val in lattice_state.curl.items():
            x, y = self.geometry.site_to_position(site_idx)
            curl_val_norm = curl_val / curl_norm
            curl_circle = plt.Circle(
                (x + 0.5, y + 0.5),
                0.2,
                facecolor="blue" if curl_val_norm > 0 else "red",
                zorder=2,
            )
            ax.add_patch(curl_circle)
            ax.text(x + 0.5, y + 0.5, f"{curl_val_norm:.2f}", color="white", ha="center", va="center", fontsize=10, zorder=3)
            ax.plot(x + 0.5, y + 0.5, alpha=0)

        arrow_x, arrow_y = self.Lx - 0.1, self.Ly - 2
        polarisation = self._polarisation(lattice_state.density) / pol_norm
        polarisation_current = (
            self._polarisation_current(lattice_state.density, self.states[state_index - 1].density if state_index > 0 else lattice_state.density)
            / pol_norm
        )

        curl_polarisation = self._curl_polarisation(lattice_state.curl) / curl_pol_norm
        curl_polarisation_current = (
            self._curl_polarisation_current(
                lattice_state.curl,
                self._curl(self._current_density(self.states[state_index - 1].density)) if state_index > 0 else lattice_state.curl,
            )
            / curl_pol_norm
        )

        if np.linalg.norm(polarisation_current) > 1:
            polarisation_current /= np.linalg.norm(polarisation_current)
        if np.linalg.norm(curl_polarisation_current) > 1:
            curl_polarisation_current /= np.linalg.norm(curl_polarisation_current)

        Ex, Ey = 1 / E_norm * self.E(state_index * self.h) * self.E_dir

        utils.plot_arrow(ax, arrow_x + 0.5, arrow_y, *polarisation, color="black", label="$\\vec P$")
        utils.plot_arrow(ax, arrow_x + 1, arrow_y, *polarisation_current, color="blue", label="$\\frac{\\partial \\vec P}{\\partial t}$")
        utils.plot_arrow(ax, arrow_x, arrow_y, Ex, Ey, color="red", label="$\\vec E$")
        utils.plot_arrow(ax, arrow_x + 0.5, arrow_y - 2, *curl_polarisation, color="green", label="$\\nabla \\times \\vec J$")
        utils.plot_arrow(
            ax,
            arrow_x + 0.5,
            arrow_y - 3,
            *curl_polarisation_current,
            color="orange",
            label="$\\frac{\\partial}{\\partial t} (\\nabla \\times \\vec J)$",
        )

        ax.plot(arrow_x + 3, 0, alpha=0)

        # Add annotation for time step
        ax.text(
            self.Lx + 0.5,
            self.Ly - 0.9,
            f"t = {state_index*self.h:.2f}",
            fontsize=14,
            color="black",
        )

        ax.set_aspect("equal")
        fig = ax.get_figure()
        fig.tight_layout()
        ax.axis("off")

        if show_plot:
            plt.show()

    def plot_hamiltonian(self) -> None:
        _, ax = plt.subplots(figsize=(2 * self.Lx, 2 * self.Ly))
        utils.plot_site_grid(np.diag(self.H_onsite), self.Lx, self.Ly, ax, cmap_name="viridis")
        utils.plot_site_connections(self.H_hop, self.Lx, self.Ly, ax, max_flow=self.t_hop, plot_flow_direction_arrows=False)
        plt.tight_layout()
        plt.show()

    def save_current_density_animation(self, filename: str, sample_every: int = 1, curl_norm: float = 1, **save_format_kwargs) -> None:
        curl_norm = max([max(np.abs(list(self._curl(self._current_density(state.density)).values()))) for state in self.states])
        E_norm = max([np.abs(self.E(i * self.h)) for i in range(self.steps)])
        polarisation_norm = max([np.linalg.norm(state.polarisation) for state in self.states])
        curl_polarisation_norm = max([np.linalg.norm(state.curl_polarisation) for state in self.states])

        n_frames = len(self.states) // sample_every
        fig, ax = plt.subplots(figsize=(2 * self.Lx + 2, 2 * self.Ly))
        animation = matplotlib.animation.FuncAnimation(
            fig,
            lambda frame: self.plot_current_density(
                sample_every * frame,
                ax,
                curl_norm=curl_norm,
                E_norm=E_norm,
                pol_norm=polarisation_norm,
                curl_pol_norm=curl_polarisation_norm,
            ),
            frames=n_frames,
        )

        progress_bar = tqdm(total=n_frames, desc="Generating animation", unit="frame")

        def update_progress(current_frame, _):
            # Reset the position each time to avoid empty lines
            progress_bar.n = current_frame + 1
            progress_bar.refresh()

        animation.save(filename, progress_callback=update_progress, **save_format_kwargs)
        progress_bar.close()
        plt.close(fig)

    def save(self, filename: str) -> None:
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str):
        with open(filename, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    params = SimulationParameters(
        t_hop=1,
        E_amplitude=1.0,
        E_direction=np.array([0, -1]),
        h=0.01,
        T=6,
        initial_occupation=0.5,
    )

    l = Lattice2D(geometry=BrickwallLatticeGeometry((5, 5)), simulation_parameters=params)
    # l.plot_hamiltonian()
    l.evolve(options={"progress_bar": True})
    # l.save_current_density_animation("refactored_bw.gif", fps=10, sample_every=20)
