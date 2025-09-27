from typing import Tuple, Union, Callable, Optional, Dict
import warnings
import dill
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from matplotlib.colors import Normalize
import matplotlib.lines as mlines

from scipy import linalg
from tqdm import tqdm
from numpy.typing import NDArray

from .lattice_rk4 import Observable
from .lattice_geometry import LatticeGeometry, BrickwallLatticeGeometry, HexagonalLatticeGeometry
from . import lattice_utils as utils
from .field_generator import FieldAmplitudeGenerator

# Units a=1, h_bar=1 and e=1, t_hop=1
# [P] = e*a/a**2


@dataclass
class LatticeState:
    density: NDArray[np.complex64]
    current: NDArray[np.float64]
    polarisation: NDArray[np.float64]
    bulk_polarisation: NDArray[np.float64]
    boundary_polarisation_dipole: NDArray[np.float64]
    boundary_polarisation_form: NDArray[np.float64]
    orbital_charges: Dict[int, float]
    orbital_charge_polarisation: NDArray[np.float64]
    orbital_charge_polarisation_corrected: NDArray[np.float64]


@dataclass
class SimulationParameters:
    t_hop: float
    E_amplitude: Callable[[np.typing.ArrayLike], np.typing.ArrayLike]
    E_direction: np.ndarray
    h: float
    T: float
    substeps: int
    initial_occupation: float = 0.5

    @property
    def simulation_n_steps(self):
        return int(self.T / self.h)

    @classmethod
    def default(cls):
        return cls(
            t_hop=-1,
            E_amplitude=FieldAmplitudeGenerator.constant(0.01),
            E_direction=np.array([0, -1]),
            h=0.01,
            T=1,
            initial_occupation=0.5,
            substeps=1,
        )


class Lattice2D:
    def __init__(
        self,
        geometry: LatticeGeometry,
        simulation_parameters: SimulationParameters,
    ):
        self.geometry = geometry
        self.Lx, self.Ly = geometry.dimensions
        self.N = self.Lx * self.Ly

        if self.N % 2:
            warnings.warn(
                "Site number should be even. Then integer # of states can be set with half occupation.",
            )

        self.simulation_parameters = simulation_parameters

        print("Initialize Hamiltonian and eigenstates...", end=" ")
        self.H_onsite = self.create_onsite_potentials()
        self.H_hop = self.create_hopping_hamiltonian()
        self.eigen_energies, self.energy_states = linalg.eigh(self.H_hop)
        print("Done.")

        self.states: list[NDArray] = None

        if simulation_parameters.initial_occupation:
            self.set_fractional_occupation(simulation_parameters.initial_occupation)
        else:
            self.density_matrix = np.zeros((self.N, self.N), dtype=complex)

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
    def origin(self):
        return self.geometry.origin

    @property
    def curl_origin(self):
        return self.geometry.curl_origin

    @property
    def E(self) -> float:
        return self.simulation_parameters.E_amplitude

    def create_hopping_hamiltonian(self) -> np.ndarray:
        """Create the hopping Hamiltonian based on the geometry's hopping matrix"""
        return self.geometry.get_hopping_matrix(self.t_hop)

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

    def evolve(
        self, force_reevolve: bool = False, solver: str = "rk4", use_gpu: bool = False, **solver_kwargs
    ) -> None:
        """Evolve the system over time and compute all derived quantities"""
        if self.states is not None and not force_reevolve:
            print("Lattice was already evolved, call with force_reevolve=True to simulate again.")
            return

        if solver.lower() == "qutip":
            if self.simulation_parameters.substeps > 1:
                warnings.warn("Substeps not implemented for qutip solver. Using substeps=1.")

            import qutip as qu

            H = [qu.Qobj(self.H_hop), [qu.Qobj(self.H_onsite), self.E]]
            rho = qu.Qobj(self.density_matrix)
            step_list = np.linspace(0, self.h * self.steps, self.steps)
            sim = qu.mesolve(H, rho, step_list, **solver_kwargs)

            # Compute all derived quantities for each time step
            self.states = [state.data_as(format="ndarray") for state in sim.states]
        elif solver == "rk4":
            dt = self.h / self.simulation_parameters.substeps
            sample_every = self.simulation_parameters.substeps

            if solver_kwargs.get("sample_every", 1) != 1:
                warnings.warn("Use substeps parameter to decrease the number of samples and ensure appropriate h time scale.")
                self.simulation_parameters.h = self.h * solver_kwargs["sample_every"]
                sample_every = solver_kwargs["sample_every"]

            if use_gpu:
                from . import lattice_rk4_gpu as rk4

                self.states = rk4.evolve_density_matrix_rk4_gpu(
                    self.H_hop,
                    self.H_onsite,
                    self.density_matrix,
                    self.E,
                    dt,
                    self.simulation_parameters.T,
                    sample_every=sample_every,
                    **solver_kwargs,
                )
            else:
                from . import lattice_rk4 as rk4

                self.orbital_polarizations = []
                class OrbitalPolarizationObservable(Observable):
                    def measure(self_pol, density_matrix: np.ndarray, step_index:int) -> float:
                        if (step_index % sample_every):
                            #return
                            ...
                        self.orbital_polarizations.append(self._orbital_polarisation_with_shape(self._current_density(density_matrix)))

                self.states = rk4.evolve_density_matrix_rk4(
                    self.H_hop,
                    self.H_onsite,
                    self.density_matrix,
                    self.E,
                    dt,
                    self.simulation_parameters.T,
                    sample_every=sample_every,
                    observables=[OrbitalPolarizationObservable()],
                    **solver_kwargs,
                )


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
            bulk_polarisation=self._orbital_polarisation_bulk(current_matrix),
            boundary_polarisation_dipole=self._orbital_polarisation_boundary_correction(current_matrix)[0],
            boundary_polarisation_form=self._orbital_polarisation_boundary_correction(current_matrix)[1],
            orbital_charge_polarisation_corrected=self._orbital_polarisation_with_shape(current_matrix)
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
    
    def _orbital_polarisation_with_shape(self, J: np.ndarray) -> np.ndarray:
        P = np.zeros(2)
        m = 1 # TODO
        a_NN = 1 # TODO
        R = np.array([[0, -1], [1, 0]])
        for cell_index in self.geometry.get_curl_sites():
            R_i = np.array(self.geometry.site_to_position(cell_index)) - self.geometry.curl_origin
            for (k, l) in [(0, 1), (1, 2), (2, self.Lx + 2), (self.Lx + 2, self.Lx + 1), (self.Lx + 1, self.Lx), (self.Lx, 0)]:
                r_k = np.array(self.geometry.site_to_position(cell_index + k))
                r_l = np.array(self.geometry.site_to_position(cell_index + l))
                P += (-m) * a_NN**2 * (
                    (np.sqrt(3)/2 * R_i)
                    -  (5 / 24) * (R @ (r_k - r_l))
                ) * J[cell_index + k, cell_index + l]
        # TODO Normalize to lattice area etc
        return P

    def _orbital_polarisation_bulk(self, J: np.ndarray) -> np.ndarray:
        """Calculate the bulk summand of orbital polarization from the current density. Could be implemented way more efficiently by using the lattice structure."""
        assert self.geometry.__class__ == HexagonalLatticeGeometry, "Bulk orbital polarization only implemented for hexagonal lattices."
        m = 1 # TODO
        a_NN = 1 # TODO
        P = np.zeros(2)
        w = m * a_NN**2 * (3 / 2 + 5 / 12)
        R = np.array([[0, -1], [1, 0]])

        for (k, l) in self.geometry.edges:
            r_k, r_l = np.array(self.geometry.site_to_position(k)), np.array(self.geometry.site_to_position(l))
            P += w * J[k, l] * (R @ (r_k - r_l))
        return P

    def _orbital_polarisation_boundary_correction(self, J: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """This and the bulk polarisation add up to the full orbital polarisation. Since boundary edges are shared only by one unit cell, they were added twice in the bulk term.
        Results are split into a contribution stemming from the global dipole moment of the loop current momentum and a form term stemming form unbalanced currents within the cell. 
        """
        assert self.geometry.__class__ == HexagonalLatticeGeometry, "Boundary correction only implemented for hexagonal lattices."
       # assert self.Ly % 2 == 0, "Boundary correction only implemented for odd Ly, even would have ill-defined cells at the top/bottom edges."
       # assert self.Lx % 2, "Boundary correction only implemented for odd Lx, even would have ill-defined cells at the top/bottom edges."
        
        m = 1 # TODO
        a_NN = 1 # TODO
        P_dipole = np.zeros(2)
        P_form = np.zeros(2)
        w_dipole = -m * a_NN**2 * np.sqrt(3)/2 
        w_form = m * a_NN**2 * 5/24
        R = np.array([[0, -1], [1, 0]])

        for i in self.geometry.get_curl_sites():
            R_i = np.array(self.geometry.site_to_position(i))
            row_index = np.rint(R_i[1] / self.geometry.row_height - 1/6)
            col_index = np.rint(R_i[0] / self.geometry.col_width)

            kl = []
            R_offsets = []
            # Left outer Armchair edge
            if col_index <= 1:
                kl.append( (self.Lx, 0) )
                R_offsets.append(np.array([-2 * self.geometry.col_width, 0]))
            if col_index == 0 and row_index > 0:
                kl.append( (0, 1) ) # we dont overlap with the bottom zigzag edge, so add bottom left
                R_offsets.append(np.array([-self.geometry.col_width, -self.geometry.row_height]))
            if col_index == 0 and row_index < self.Ly - 2:
                kl.append( (self.Lx+1, self.Lx) ) # we dont overlap with the top zigzag edge, so add top left
                R_offsets.append(np.array([-self.geometry.col_width, self.geometry.row_height]))
            
            # Right outer Armchair edge
            if col_index >= self.Lx - 4:
                kl.append( (2, self.Lx+2) )
                R_offsets.append(np.array([2 * self.geometry.col_width, 0]))
            if col_index == self.Lx - 3 and row_index > 0: # we dont overlap with the bottom zigzag edge, so add bottom right
                kl.append( (1, 2) )
                R_offsets.append(np.array([self.geometry.col_width, -self.geometry.row_height]))
            if col_index == self.Lx - 3 and row_index < self.Ly - 2: # we dont overlap with the top zigzag edge, so add top right
                kl.append( (self.Lx + 2, self.Lx + 1) )
                R_offsets.append(np.array([self.geometry.col_width, self.geometry.row_height]))

            # Top outer Zigzag edge
            if row_index == self.Ly - 2:
                kl.append( (self.Lx + 1, self.Lx) )
                R_offsets.append(np.array([-self.geometry.col_width, self.geometry.row_height]))

                kl.append( (self.Lx + 2, self.Lx + 1) )
                R_offsets.append(np.array([self.geometry.col_width, self.geometry.row_height]))

            # Bottom outer Zigzag edge
            if row_index == 0:
                kl.append( (0, 1) )
                R_offsets.append(np.array([-self.geometry.col_width, -self.geometry.row_height]))
                
                kl.append( (1, 2) )
                R_offsets.append(np.array([self.geometry.col_width, -self.geometry.row_height]))

            for (l, k), R_offset in zip(kl, R_offsets):
                r_k = np.array(self.geometry.site_to_position(i+k))
                r_l = np.array(self.geometry.site_to_position(i+l))
                P_form -= w_form * J[i+k, i+l] * (R @ (r_k - r_l))
                P_dipole -= w_dipole * J[i+k, i+l] * (R_i + R_offset - self.geometry.curl_origin)

        #plt.show()
        return P_dipole, P_form


    def _polarisation_current(self, state_matrix: np.ndarray, previous_step_state_matrix: np.ndarray) -> np.ndarray:
        """Calculate time derivative of polarization"""
        return (self._polarisation(state_matrix) - self._polarisation(previous_step_state_matrix)) / self.h

    def _orbital_polarisation_current(self, J: np.ndarray, prev_J: np.ndarray) -> np.ndarray:
        """Calculate time derivative of curl polarization"""
        return (self._orbital_polarisation_with_shape(J) - self._orbital_polarisation_with_shape(prev_J)) / self.h

    def orbital_hall_conductivity(self, omega: float, steady_state_start_time: float = 0.0) -> complex:
        if self.states is None:
            raise ValueError("Lattice must be evolved before calculating conductivity.")
        
        t_values = np.arange(0, self.h * len(self.states), self.h)
        E_values = self.E(t_values)
        J_orb_values = np.array(
            [self._orbital_polarisation_current(self._current_density(self.states[i]), self._current_density(self.states[i-1]))[0] for i in range(1, len(self.states))]
        )

        start_index = np.argmax(t_values >= steady_state_start_time)
        t_steady = t_values[start_index:]
        E_steady = E_values[start_index:]
        J_orb_steady = J_orb_values[start_index:]

        J_omega = np.fft.fft(J_orb_steady)
        E_omega = np.fft.fft(E_steady)

        plt.plot(t_steady, E_steady, label="E(t)")
        plt.plot(t_steady[:-1], J_orb_steady, label="J_orb(t)")
        plt.xlabel("Time")
        plt.legend()
        plt.show()


        freqs = np.fft.fftfreq(len(t_steady), self.h) # In Hz
        freqs2 = np.fft.fftfreq(len(t_steady)-1, self.h) # In Hz
    
        plt.plot(freqs, np.abs(E_omega), label="|E(ω)|")
        plt.plot(freqs2, np.abs(J_omega), label="|J_orb(ω)|")
        plt.xlim(-3*omega/(2*np.pi), 3*omega/(2*np.pi))
        plt.xlabel("Frequency (Hz)")
        plt.legend()
        plt.yscale("log")
        
        # Find the index of the frequency bin closest to our driving frequency.
        f_drive = omega / (2 * np.pi)
        index_drive = np.argmin(np.abs(freqs - f_drive))

        plt.vlines([freqs[index_drive]], ymin=1e-5, ymax=np.max(E_omega))
        plt.show()


        sigma_xy = J_omega[index_drive] / E_omega[index_drive]  
        return sigma_xy



    def _auto_normalize(self) -> Tuple[float, float, float, float]:
        current_densities = [self._current_density(state) for state in self.states]
        curl_densities = [self._orbital_charges(current_density) for current_density in current_densities]
        curl_pol = [self._orbital_polarisation(curl_density) for curl_density in curl_densities]

        curl_norm = max([max(np.abs(list(curl_density.values()))) for curl_density in curl_densities])
        curl_pol_norm = max([np.linalg.norm(curl_pol_density) for curl_pol_density in curl_pol])
        pol_norm = max(np.linalg.norm(self._polarisation(state)) for state in self.states)
        E_norm = max([np.linalg.norm(self.E(i * self.h)) for i in range(self.steps)])

        del current_densities, curl_densities, curl_pol

        return curl_norm, curl_pol_norm, pol_norm, E_norm

    def plot_current_density(
        self,
        state_index: int,
        ax: Optional[plt.axis] = None,
        curl_norm: float = 1,
        E_norm: float = 1,
        curl_pol_norm: float = 1,
        pol_norm: float = 1,
        current_norm: Optional[float] = None,
        auto_normalize: bool = False,
    ) -> None:
        show_plot = ax is None

        if self.states is None:
            lattice_state = self.compute_lattice_state(self.density_matrix)
        else:
            lattice_state = self.compute_lattice_state(self.states[state_index])

        if auto_normalize:
            curl_norm, curl_pol_norm, pol_norm, E_norm = self._auto_normalize()

        if ax is None:
            _, ax = plt.subplots(figsize=(2 * self.Lx + 2, 2 * self.Ly))
        else:
            ax.clear()

        site_values = np.real(np.diag(lattice_state.density)) * self.N * self.occupation_fraction
        utils.plot_site_grid(site_values, self.geometry, ax, vmin=0.0, vmax=1, cmap_name="Greys", print_text_labels=False)
        if current_norm is None:
            current_norm = np.max(np.abs(lattice_state.current))
        utils.plot_site_connections(
            lattice_state.current, self.geometry, ax, max_flow=current_norm, label_connection_strength=False
        )

        #norm = plt.Normalize(vmin=vmin if vmin is not None else np.nanmin(site_values), vmax=vmax if vmax is not None else np.nanmax(site_values))
        cmap = plt.get_cmap("RdBu")

        # Plot curl indicators
        for site_idx, curl_val in lattice_state.orbital_charges.items():
            radius = 0.2
            x, y = self.geometry.site_to_position(site_idx)
            x += 0.5  # self.geometry.cell_width / 2 - radius
            y += 0.5  # self.geometry.cell_height / 2 - radius
            if self.geometry.__class__ == HexagonalLatticeGeometry:
                x += np.sqrt(3) / 2 - 0.5
            curl_val_norm = curl_val / curl_norm
            curl_circle = plt.Circle(
                (x, y),
                radius,
                facecolor=cmap(curl_val_norm / 2 + 0.5), # "blue" if curl_val_norm > 0 else "red",
                zorder=2,
            )
            ax.add_patch(curl_circle)
            #ax.text(x, y, f"{curl_val_norm:.2f}", color="white", ha="center", va="center", fontsize=10, zorder=3, path_effects=[patheffects.withStroke(linewidth=1, foreground="black")])
            ax.plot(x, y, alpha=0)

            if abs(curl_val_norm) >= 0.01:
                utils.drawCirc(ax, 0.7, x, y, 125, 310, "blue" if curl_val_norm > 0 else "red", lw=1.5, anticlockwise=curl_val_norm > 0)

        arrow_x, arrow_y = self.Lx - 0.1, self.Ly - 2
        polarisation = self._polarisation(lattice_state.density) / pol_norm
        polarisation_current = (
            self._polarisation_current(lattice_state.density, self.states[state_index - 1] if state_index > 0 else lattice_state.density) / pol_norm
        )

        curl_polarisation = self._orbital_polarisation(lattice_state.orbital_charges) / curl_pol_norm
        curl_polarisation_current = (
            self._orbital_polarisation_current(
                lattice_state.current,
                self._current_density(self.states[state_index - 1]) if state_index > 0 else lattice_state.current,
            )
            / curl_pol_norm
        )

        if np.linalg.norm(polarisation_current) > 1:
            polarisation_current /= np.linalg.norm(polarisation_current)
        if np.linalg.norm(curl_polarisation_current) > 1:
            curl_polarisation_current /= np.linalg.norm(curl_polarisation_current)

        Ex, Ey = 1 / E_norm * self.E(state_index * self.h) * self.E_dir

        #utils.plot_arrow(ax, arrow_x + 0.5, arrow_y, *polarisation, color="black", label="$\\vec P$")
        #utils.plot_arrow(ax, arrow_x + 1, arrow_y, *polarisation_current, color="blue", label="$\\frac{\\partial \\vec P}{\\partial t}$")
        #utils.plot_arrow(ax, arrow_x, arrow_y, Ex, Ey, color="red", label="$\\vec E$")
        #utils.plot_arrow(ax, arrow_x + 0.5, arrow_y - 2, *curl_polarisation, color="green", label="$\\vec P_{\\rm orb}$")
        #utils.plot_arrow(
        #    ax, arrow_x + 0.5, arrow_y - 3, *curl_polarisation_current, color="orange", label="$\\frac{\\partial \\vec P_{\\rm orb}}{\\partial t}$"
        #)

        #ax.plot(arrow_x + 3, 0, alpha=0)

        # Add annotation for time step
        # ax.text(
        #     self.Lx + 0.5,
        #     self.Ly - 0.9,
        #     f"t = {state_index*self.h:.2f}",
        #     fontsize=14,
        #     color="black",
        # )



        # --- User Parameters ---

        # Main plot area adjustment
        figure_left_margin = 0.08
        figure_right_margin = 0.67 # Main plot takes up to 58% of fig width. Tune this!
        figure_top_margin = 0.92
        figure_bottom_margin = 0.15 # For P_orb arrow and some breathing room

        # Legend Positioning
        legend_anchor_to_ax_x = 1.01  # X: Distance from ax right edge (as fraction of ax width)
        legend_anchor_to_ax_y = 0.5   # Y: Vertical centering relative to ax (0.5 = center)
        legend_location_on_box = 'center left' # Which part of legend box is anchored

        # Colorbar settings
        occupation_cmap_name = 'Greys' # Or 'viridis', 'cividis'
        oam_cmap_name = 'RdBu'       # Good for bipolar OAM data
        min_occupation, max_occupation = 0.0, 1 # Your actual data range
        min_oam, max_oam = -1, 1       # Your actual OAM data range (from image)

        # Layout parameters for colorbars
        cbar_padding_from_legend_fig = -0.17 #Fig fraction: space between legend and cbar stack
        cbar_width_fig = 0.02               # Fig fraction: width of each cbar
        cbar_height_relative_to_legend = 1.4 # Each cbar is X% of legend height.
        cbar_vertical_spacing_fig = 0.2    # Fig fraction: vertical space between cbars
        colorbar_label_pad = 5             # Space for cbar labels from cbar
        colorbar_tick_pad = 5               # Space for cbar tick labels from cbar axis

        offset = .15


        fig = ax.get_figure()

        # ax.set_title("Honeycomb Lattice Dynamics at t=0.10", fontsize='x-large', pad=15)
        ax.set_xticks([]) # Cleaner look for the main visualization
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box') # Important for geometric representations

        # Adjust main plot to make space for legend and other elements
        fig.subplots_adjust(left=figure_left_margin, right=figure_right_margin,
                            top=figure_top_margin, bottom=figure_bottom_margin)

        # --- Create Custom Legend Handles and Labels ---
        handles = []
        labels = []
        legend_marker_size = 11
        legend_handle_length = 1.8 # Length of the line/patch in the legend

            # 2. Inter-site Current
        handles.append(mlines.Line2D([], [], color='black', linestyle='-', marker='>',
                                    markersize=7, mfc='black', mec='black', lw=1.5))
        labels.append(r'Inter-Site Current')

        # 1. Crystal Sites (Orbital Occupation) - Circular patch with border
        handles.append(mlines.Line2D([], [], color=cm.get_cmap(occupation_cmap_name)(0.6), # Mid-gray
                                    marker='o', linestyle='None', markersize=legend_marker_size,
                                    markeredgecolor='black', mew=1.0)) # mew is markeredgewidth
        labels.append("Site Occupation")

        
        # 3. Local OAM Density - Circular patch, no visible border
        # Color is a representative from the RdBu map. Edge color matches face color.
        oam_legend_patch_color = cm.get_cmap(oam_cmap_name)(0.75) # Light blue from RdBu
        handles.append(mlines.Line2D([], [], color=oam_legend_patch_color, marker='o', linestyle='None',
                                    markersize=legend_marker_size,
                                    markeredgecolor=oam_legend_patch_color)) # Edge same as face
        labels.append("Local OAM Density")

    


        # handles.append(mlines.Line2D([], [], color='green', lw=0, marker='>', markersize=8, fillstyle='full'))
        # labels.append(r'Average OAM Current')

        # --- Create the Legend ---
        legend = ax.legend(handles, labels,
                        # title=legend_title,
                        loc=legend_location_on_box,
                        bbox_to_anchor=(legend_anchor_to_ax_x, legend_anchor_to_ax_y),
                        bbox_transform=ax.transAxes,
                        fontsize='medium',
                        title_fontsize='large',
                        frameon=True, edgecolor='black',
                        labelspacing=1.2, # Vertical spacing between legend items
                        handletextpad=0.8, # Horizontal space between handle and text
                        handlelength=legend_handle_length,
                        borderpad=0.5) # Padding inside legend border
        fig.canvas.draw() # IMPORTANT: Draw to calculate legend's true size and position

        # --- Create Colorbars ---
        legend_bbox_figcoords = legend.get_window_extent().transformed(fig.transFigure.inverted())

        cbar_x_start_fig = legend_bbox_figcoords.x1 + cbar_padding_from_legend_fig
        each_cbar_height_fig = legend_bbox_figcoords.height * cbar_height_relative_to_legend

        legend_center_y_fig = legend_bbox_figcoords.y0 + legend_bbox_figcoords.height / 2
        # Y position for top colorbar (its bottom-left corner)
        top_cbar_y_start_fig = legend_center_y_fig + 0.1
        # Y position for bottom colorbar (its bottom-left corner)
        bottom_cbar_y_start_fig = top_cbar_y_start_fig #legend_center_y_fig - each_cbar_height_fig - cbar_vertical_spacing_fig / 2

        # Occupation Colorbar (Top)
        occ_norm = Normalize(vmin=min_occupation, vmax=max_occupation)
        occ_sm = cm.ScalarMappable(norm=occ_norm, cmap=plt.get_cmap(occupation_cmap_name))
        occ_sm.set_array([]) # Important for standalone colorbar
        cax_occ = fig.add_axes([cbar_x_start_fig, top_cbar_y_start_fig, cbar_width_fig, each_cbar_height_fig])
        cb_occ = fig.colorbar(occ_sm, cax=cax_occ, orientation='vertical')
        cb_occ.set_label(r'Site Occupation $/N$', size='medium', labelpad=colorbar_label_pad)
        cb_occ.ax.tick_params(labelsize='small', pad=colorbar_tick_pad)
        cb_occ.set_ticks(np.linspace(min_occupation, max_occupation, 3)) # Example: 3 ticks

        # OAM Density Colorbar (Bottom)
        oam_norm = Normalize(vmin=min_oam, vmax=max_oam)
        oam_sm = cm.ScalarMappable(norm=oam_norm, cmap=plt.get_cmap(oam_cmap_name))
        oam_sm.set_array([])
        cax_oam = fig.add_axes([cbar_x_start_fig + offset, bottom_cbar_y_start_fig, cbar_width_fig, each_cbar_height_fig])
        cb_oam = fig.colorbar(oam_sm, cax=cax_oam, orientation='vertical')
        cb_oam.set_label('Local OAM Density ($\\times 10^{-6}$)', size='medium', labelpad=colorbar_label_pad)

        cb_oam.ax.tick_params(labelsize='small', pad=colorbar_tick_pad)
        oam_ticks = sorted(list(set([min_oam, 0, max_oam]))) # Ticks at min, 0, max
        if len(oam_ticks) == 1 and oam_ticks[0] == 0: oam_ticks = [0] # Handle case where min=max=0
        cb_oam.set_ticks(oam_ticks)

        
        ax.set_aspect("equal")
        fig = ax.get_figure()
        fig.tight_layout()
        ax.axis("off")

        if show_plot:
            plt.show()

    def plot_hamiltonian(self) -> None:
        _, ax = plt.subplots(figsize=(2 * self.Lx, 2 * self.Ly))
        utils.plot_site_grid(np.diag(self.H_onsite), self.geometry, ax, cmap_name="viridis")
        utils.plot_site_connections(self.H_hop, self.geometry, ax, max_flow=self.t_hop, plot_flow_direction_arrows=False)
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
        ax: Optional[plt.axis] = None,
    ) -> plt.axis:
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
        utils.plot_site_grid(field_array / site_norm, self.geometry, ax, cmap_name=field_cmap)

        for idx, grad in gradient.items():
            x, y = idx % self.Lx, idx // self.Lx
            dx, dy = grad * arrow_scale / max_grad if max_grad > 0 else grad * arrow_scale

            ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.15, fc=arrow_color, ec=arrow_color, width=0.02, zorder=2, length_includes_head=True)

        # Plot transparent dots to extend boundaries
        ax.plot(-1, -1, alpha=0)
        ax.plot(self.Lx // self.geometry.cell_width + 1, self.Ly // self.geometry.cell_height, alpha=0)

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

    def plot_combined_current_and_curl(self, state_index: int = 0) -> Tuple[plt.axis, plt.axis]:
        """Create a combined plot with current density on the left and curl+gradient on the right"""
        if self.states is None:
            state = self.compute_lattice_state(self.density_matrix)
        else:
            state = self.compute_lattice_state(self.states[state_index])

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
            state = self.compute_lattice_state(self.states[idx])
            curl = state.orbital_charges
            grad = self.geometry.cell_field_gradient(curl)

            self.plot_field_and_gradient(curl, grad, label="\\nabla q_{\\rm orb}", field_cmap="bwr_r", arrow_color="black", arrow_scale=1, ax=ax2)

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

        anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
        anim.save(filename, progress_callback=update_progress, **save_kwargs)
        plt.close(fig)

    def save_current_density_animation(self, filename: str, sample_every: int = 1, curl_norm: float = 1, **save_format_kwargs) -> None:
        curl_norm, curl_polarisation_norm, polarisation_norm, E_norm = self._auto_normalize()

        n_frames = len(self.states) // sample_every
        fig, ax = plt.subplots(figsize=(2 * self.Lx + 2, 2 * self.Ly))
        anim = animation.FuncAnimation(
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

        anim.save(filename, progress_callback=update_progress, **save_format_kwargs)
        progress_bar.close()
        plt.close(fig)

    def save(self, filename: str) -> None:
        with open(filename, "wb") as f:
            dill.dump(self, f)

    @staticmethod
    def load(filename: str):
        with open(filename, "rb") as f:
            return dill.load(f)


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
