from typing import Tuple, Union, Callable, Optional, Dict
import warnings
import pickle
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import qutip as qu
from scipy import linalg
from tqdm import tqdm
from numpy.typing import NDArray
from lattice_geometry import LatticeGeometry, BrickwallLatticeGeometry
import lattice_utils as utils
import lattice_rk4 as rk4

# Units a=1, h_bar=1 and e=1, t_hop=1
# [P] = e*a/a**2


@dataclass
class LatticeState:
    density: NDArray[np.complex64]
    current: NDArray[np.float64]
    polarisation: NDArray[np.float64]
    orbital_charges: Dict[int, float]
    orbital_charge_polarisation: NDArray[np.float64]


@dataclass
class SimulationParameters:
    t_hop: float
    E_amplitude: Union[float, Callable[[float], float]]
    E_direction: np.ndarray
    h: float
    T: float
    charge: int = 1  # TODO: Implement charge
    initial_occupation: float = 0.5
    sample_every: int = 1

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
            sim = rk4.evolve_density_matrix_rk4(
                self.H_hop, self.H_onsite, self.density_matrix, self.E, self.h, self.simulation_parameters.T, **solver_kwargs
            )
            if solver_kwargs.get("sample_every", 1) != 1:
                self.simulation_parameters.h = self.h * solver_kwargs["sample_every"]
            self.states = [self.compute_lattice_state(state) for state in sim]

    def compute_lattice_state(self, density_matrix: np.ndarray) -> LatticeState:
        """Compute all derived quantities from the density matrix"""
        current_matrix = self._current_density(density_matrix)
        polarisation = self._polarisation(density_matrix)
        curl = self._orbital_charges(current_matrix)
        curl_polarisation = self._orbital_polarisation(curl)

        return LatticeState(
            density=density_matrix,
            current=current_matrix,
            polarisation=polarisation,
            orbital_charges=curl,
            orbital_charge_polarisation=curl_polarisation,
        )

    def _current_density(self, state_matrix: np.ndarray) -> np.ndarray:
        """Calculate current density from the density matrix"""
        return -2 * (self.H_hop * state_matrix.T).imag  # [J] = e/[t] = h_bar e/t_hop

    def _orbital_charges(self, J: np.ndarray) -> Dict[int, float]:
        """Calculate curl around each unit cell using the geometry's curl sites"""
        cell_area = self.geometry.cell_width * self.geometry.cell_height
        return {
            site_idx: sum([J[site_idx + di, site_idx + dj] for di, dj in self.geometry.cell_path]) / self.t_hop
            for site_idx in self.geometry.get_curl_sites()
        }  # [q_orb] = e

    def _polarisation(self, state_matrix: np.ndarray) -> np.ndarray:
        """Calculate polarization from density matrix"""
        return (
            np.sum([(np.array(self.geometry.site_to_position(i)) - self.origin) * state_matrix.diagonal().real[i] for i in range(self.N)], axis=0)
            / self.N
        )  # [P] = e*a/a**2 = e/a

    def _orbital_polarisation(self, curl_J: Dict[int, float]) -> np.ndarray:
        return (
            1
            / self.N
            * np.sum(
                [(np.array(self.geometry.site_to_position(site_index)) - self.curl_origin) * curl_val for site_index, curl_val in curl_J.items()],
                axis=0,
            )
        )  # [P_orb] = e/a

    def _polarisation_current(self, state_matrix: np.ndarray, previous_step_state_matrix: np.ndarray) -> np.ndarray:
        """Calculate time derivative of polarization"""
        return (self._polarisation(state_matrix) - self._polarisation(previous_step_state_matrix)) / self.h

    def _orbital_polarisation_current(self, curl_J: Dict[int, float], previous_curl_J: Dict[int, float]) -> np.ndarray:
        """Calculate time derivative of curl polarization"""
        return (self._orbital_polarisation(curl_J) - self._orbital_polarisation(previous_curl_J)) / self.h

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
            curl_norm = max([max(np.abs(list(state.orbital_charges.values()))) for state in self.states])
            E_norm = max([np.linalg.norm(self.E(i * self.h)) for i in range(self.steps)])
            pol_norm = max([np.linalg.norm(state.polarisation) for state in self.states])
            curl_pol_norm = max([np.linalg.norm(state.orbital_charge_polarisation) for state in self.states])

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
        for site_idx, curl_val in lattice_state.orbital_charges.items():
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

        curl_polarisation = self._orbital_polarisation(lattice_state.orbital_charges) / curl_pol_norm
        curl_polarisation_current = (
            self._orbital_polarisation_current(
                lattice_state.orbital_charges,
                (
                    self._orbital_charges(self._current_density(self.states[state_index - 1].density))
                    if state_index > 0
                    else lattice_state.orbital_charges
                ),
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
        utils.plot_arrow(ax, arrow_x + 0.5, arrow_y - 2, *curl_polarisation, color="green", label="$\\vec P_{\\rm orb}$")
        utils.plot_arrow(ax, arrow_x + 0.5, arrow_y - 3, *curl_polarisation_current, color="orange", label="$\\frac{\\partial \\vec P_{\\rm orb}}{\\partial t}$")

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

    def plot_field_and_gradient(
        self,
        field: dict,
        gradient: dict,
        label: str = "",
        field_cmap: str = "bwr_r",
        arrow_color: str = "black",
        arrow_scale: float = 1,
        ax: Optional[matplotlib.axes] = None,
    ) -> matplotlib.axes:
        """Plot a scalar field and its gradient on a lattice."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(2 * self.Lx, 2 * self.Ly))
        else:
            ax.clear()
            fig = ax.figure

        field_array = np.empty(self.Lx * self.Ly)
        field_array.fill(np.nan)
        for idx, val in field.items():
            field_array[idx] = val

        max_grad = max(np.linalg.norm(grad) for grad in gradient.values())
        site_norm = np.nanmax(np.abs(field_array))
        if site_norm < 1e-12:
            site_norm = 1
        utils.plot_site_grid(field_array / site_norm, self.Lx, self.Ly, ax, cmap_name=field_cmap)

        for idx, grad in gradient.items():
            x, y = idx % self.Lx, idx // self.Lx
            dx, dy = grad * arrow_scale / max_grad if max_grad > 0 else grad * arrow_scale

            ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.15, fc=arrow_color, ec=arrow_color, width=0.02, zorder=2, length_includes_head=True)

        # Plot transparent dots to extend boundaries
        ax.plot(-1, -1, alpha=0)
        ax.plot(self.Lx // self.geometry.cell_width, -1, alpha=0)
        ax.plot(self.Lx // self.geometry.cell_width, self.Ly // self.geometry.cell_height, alpha=0)

        # Plot avg gradient on the right side
        avg_grad = arrow_scale * np.mean(list(gradient.values()), axis=0)
        if max_grad:
            avg_grad /= max_grad
        utils.plot_arrow(
            ax, self.Lx - self.geometry.cell_width + 0.5, self.Ly / 2, *avg_grad, color=arrow_color, label=f"$\\langle {label} \\rangle$"
        )

        ax.set_aspect("equal")
        ax.set_axis_off()
        fig.tight_layout()

        return ax

    def plot_combined_current_and_curl(self, state_index: int = 0) -> Tuple[matplotlib.axes, matplotlib.axes]:
        """Create a combined plot with current density on the left and curl+gradient on the right"""
        if self.states is None:
            state = self.compute_lattice_state(self.density_matrix)
        else:
            state = self.states[state_index]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4 * self.Lx, 2 * self.Ly), width_ratios=[4, 2])

        # Left plot: current density
        self.plot_current_density(state_index, ax=ax1, auto_normalize=True)
        # ax1.set_title("Current Density")

        # Right plot: curl and gradient
        curl = state.orbital_charges
        grad = self.geometry.cell_field_gradient(curl)

        # Use the plot_field_and_gradient function
        self.plot_field_and_gradient(curl, grad, label="\\nabla \\times \\mathbf{J}", ax=ax2)

        # ax2.set_title("Curl and Gradient")
        return ax1, ax2

    def save_lattice_animation(self, filename, sample_every=1, **save_kwargs):
        """Save an animation with current density plot on the left and curl+gradient on the right"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4 * self.Lx, 2 * self.Ly), width_ratios=[3, 2])

        def update(frame):
            ax1.clear()
            ax2.clear()

            idx = frame * sample_every
            if idx >= len(self.states):
                idx = len(self.states) - 1

            # Plot current density on left
            self.plot_current_density(idx, ax=ax1, auto_normalize=True)
            # ax1.set_title(f"Current Density")

            # Plot curl and gradient on right
            state = self.states[idx]
            curl = state.orbital_charges
            grad = self.geometry.cell_field_gradient(curl)

            self.plot_field_and_gradient(
                curl, grad, label="\\nabla q_{\\rm orb}", field_cmap="bwr_r", arrow_color="black", arrow_scale=1, ax=ax2
            )

            # ax2.set_title(f"Curl and Gradient")
            return ax1, ax2

        frames = len(self.states) // sample_every
        if len(self.states) % sample_every != 0:
            frames += 1

        progress_bar = tqdm(total=frames, desc="Generating animation", unit="frame")

        def update_progress(current_frame, _):
            # Reset the position each time to avoid empty lines
            progress_bar.n = current_frame + 1
            progress_bar.refresh()

        anim = matplotlib.animation.FuncAnimation(fig, update, frames=frames, blit=False)
        anim.save(filename, progress_callback=update_progress, **save_kwargs)
        plt.close(fig)

    def save_current_density_animation(self, filename: str, sample_every: int = 1, curl_norm: float = 1, **save_format_kwargs) -> None:
        curl_norm = max([max(np.abs(list(self._orbital_charges(self._current_density(state.density)).values()))) for state in self.states])
        E_norm = max([np.abs(self.E(i * self.h)) for i in range(self.steps)])
        polarisation_norm = max([np.linalg.norm(state.polarisation) for state in self.states])
        curl_polarisation_norm = max([np.linalg.norm(state.orbital_charge_polarisation) for state in self.states])

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
