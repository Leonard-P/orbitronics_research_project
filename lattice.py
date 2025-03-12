import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patheffects as patheffects
from typing import List, Tuple, Union, Callable
import qutip as qu
from scipy import linalg
import warnings
import pickle
from tqdm import tqdm
from dataclasses import dataclass
from numpy.typing import NDArray


@dataclass
class LatticeState:
    density: NDArray[np.complex64]
    current: NDArray[np.float64]
    polarisation: NDArray[np.float64]
    curl: NDArray[np.float64]
    curl_polarisation: NDArray[np.float64]


class Lattice2D:
    def __init__(
        self,
        dimensions: Tuple[int],
        t_hop: float,
        E_amplitude: Union[float, Callable[[float], float]],
        E_dir: np.ndarray,
        h: float,
        steps: int = None,
        T: float = None,
        initial_occupation: float = None,
    ):
        self.Lx, self.Ly = dimensions
        self.N = self.Lx * self.Ly
        if self.N % 2:
            warnings.warn(
                "Site number should be even - then a whole number of states can be excited with half occupation.",
            )

        if all([T, steps]):
            warnings.warn("Both T and steps are set, steps will be ignored.")
        if T:
            steps = int(T / h)

        self.t_hop = t_hop
        self.t = 0  # time
        self.cell_path = np.array([(0, 1), (1, self.Lx + 1), (self.Lx + 1, self.Lx), (self.Lx, 0)])  # path around a cell for curl calculation

        if callable(E_amplitude):
            self.E = E_amplitude
        else:
            self.E = lambda _: E_amplitude
        self.E_dir = E_dir

        print("Creating Hamiltonians...")
        self.H_onsite = self.create_onsite_potentials()
        self.H_hop = self.create_hopping_hamiltonian()
        print("Done.")

        self.density_matrix = np.zeros((self.N, self.N), dtype=complex)

        self.states: list[LatticeState] = None

        self.h = h
        self.steps = steps

        print("Calculating energy eigenstates...")
        self.eigen_energies, self.energy_states = linalg.eigh(self.H_hop)
        print("Done.")

        if initial_occupation:
            self.set_fractional_occupation(initial_occupation)
            print(f"{100 * initial_occupation:.0f} % of states were set as initially occupied.")

        self.origin = (np.array([self.Lx, self.Ly]) - 1)/2 # center of lattice. Polarisation is dependent of origin.


    @property
    def cell_path(self):
        return self._cell_path
    

    @property
    def curl_origin(self):
        return self.origin - np.array([self.cell_width, self.cell_height]) / 2
    

    @cell_path.setter
    def cell_path(self, path: np.ndarray):
        self._cell_path = path
        self.cell_width = (self.cell_path.flatten() % self.Lx).max()
        self.cell_height = (self.cell_path.flatten() // self.Lx).max()


    def create_hopping_hamiltonian(self) -> np.ndarray:
        x_hop = np.tile([self.t_hop] * (self.Lx - 1) + [0], self.Ly)[:-1]
        y_hop = np.array([self.t_hop] * self.Lx * (self.Ly - 1))

        H = np.diag(x_hop, 1) + np.diag(y_hop, self.Lx)
        return H + H.conj().T

    def create_onsite_potentials(self) -> np.ndarray:
        potentials = [-np.dot([i, j], self.E_dir) for i in range(self.Ly) for j in range(self.Lx)]  # - E . r
        potentials -= np.mean(potentials)  # center around 0
        return np.diag(potentials)
    

    @property
    def H(self):
        return self.H_hop + self.E(self.t) * self.H_onsite

    def plot_hamiltonian(self) -> None:
        # Dictionary to store positions of each node.
        positions = {}
        for i in range(self.Ly):
            for j in range(self.Lx):
                idx = self.Lx * i + j
                # Set node positions: x coordinate is j, y coordinate is -i (so row 0 is top)
                positions[idx] = (j, -i)

        # Create plot
        fig, ax = plt.subplots(figsize=(2 * self.Lx, 2 * self.Ly))

        norm = plt.Normalize(vmin=self.H.diagonal().min(), vmax=self.H.diagonal().max())
        cmap = plt.get_cmap("inferno")

        # Plot nodes with their diagonal values.
        for idx, (x, y) in positions.items():
            diag_val = self.H[idx, idx]
            color_val = complex(norm(diag_val)).real
            circle_color = cmap(color_val)
            circle = plt.Circle((x, y), 0.3, color=circle_color, zorder=2)
            ax.add_patch(circle)
            # Add text with shadow for better contrast

            ax.text(
                x,
                y,
                f"{diag_val:.5f}",
                color="white",
                ha="center",
                va="center",
                fontsize=10,
                zorder=3,
                path_effects=[patheffects.withStroke(linewidth=1, foreground="black")],
            )

        # Plot edges for off-diagonal nonzero elements.
        # To avoid drawing each edge twice, we loop only over pairs with u < v.
        for u in range(self.N):
            for v in range(u + 1, self.N):
                if self.H[u, v] != 0:
                    x1, y1 = positions[u]
                    x2, y2 = positions[v]
                    # Draw a line connecting the two nodes.
                    ax.plot([x1, x2], [y1, y2], "r-", linewidth=1, zorder=1)
                    # Compute the midpoint of the edge.
                    xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
                    # Place the value of H[u,v] near the midpoint.
                    ax.text(
                        xm,
                        ym + 0.1,
                        f"{self.H[u, v]:.1f}",
                        color="red",
                        ha="center",
                        va="center",
                        fontsize=8,
                        zorder=4,
                    )

        # Set plot parameters.
        ax.set_aspect("equal")
        ax.axis("off")
        plt.show()

        # Imshow real and imag part of H
        fig, (ax1, ax2) = plt.subplots(1, 2)

        im1 = ax1.imshow(self.H.real)
        ax1.set_title("$\\Re (H)$")
        fig.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(self.H.imag)
        ax2.set_title("$\\Im (H)$")
        fig.colorbar(im2, ax=ax2)

        plt.show()

    def set_fractional_occupation(self, occupation_fraction=0.5) -> None:
        rho_energy_basis = np.diag([1 if i / self.N < occupation_fraction else 0 for i in range(self.N)])
        self.density_matrix = 1 / (occupation_fraction * self.N) * self.energy_states @ rho_energy_basis @ self.energy_states.T.conj()
        self.occupation_fraction = occupation_fraction

    def evolve(self, force_reevolve=False, **mesolve_kwargs) -> None:
        if self.states is not None and not force_reevolve:
            print("Lattice was already evolved, call with force_reevolve=True to simulate again.")
            return
        H = [qu.Qobj(self.H_hop), [qu.Qobj(self.H_onsite), self.E]]
        rho = qu.Qobj(self.density_matrix)
        step_list = np.linspace(0, self.h * self.steps, self.steps)
        sim = qu.mesolve(H, rho, step_list, **mesolve_kwargs)

        self.states = [
            self.get_lattice_state(state.data_as(format="ndarray"))
            for state in sim.states
        ]
        
        
    def get_lattice_state(self, state_matrix: np.ndarray) -> LatticeState:
        current_matrix = self.get_current_density(state_matrix)
        polarisation = self.polarisation(state_matrix)
        curl = self.curl(current_matrix)
        curl_polarisation = self.curl_polarisation(curl)
        return LatticeState(
            density=state_matrix,
            current=current_matrix,
            polarisation=polarisation,
            curl=curl,
            curl_polarisation=curl_polarisation,
        )
    

    def save_current_density_animation(self, filename: str, sample_every: int = 1, curl_norm: float = 1, **save_format_kwargs) -> None:
        fig, ax = plt.subplots(figsize=(2 * self.Lx + 2, 2 * self.Ly))

        n_frames = len(self.states) // sample_every

        curl_norm = max([max(np.abs(list(self.curl(self.get_current_density(state.density)).values()))) for state in self.states])
        Emax = max([np.abs(self.E(i * self.h)) for i in range(self.steps)])
        polarisation_norm = max([
            np.linalg.norm(state.polarisation) for state in self.states
        ])
        curl_polarisation_norm = max([
            np.linalg.norm(state.curl_polarisation) for state in self.states
        ])

        animation = matplotlib.animation.FuncAnimation(
            fig,
            lambda frame: self.plot_current_density(
                sample_every * frame,
                ax,
                curl_norm=curl_norm,
                Emax=Emax,
                pol_norm=polarisation_norm,
                curl_pol_norm=curl_polarisation_norm,
            ),
            frames=n_frames,
        )
        animation._draw_all_frames = True

        progress_bar = tqdm(total=n_frames, desc="Generating animation", unit="frame")

        def update_progress(current_frame, total_frames):
            # Reset the position each time to avoid empty lines
            progress_bar.n = current_frame
            progress_bar.refresh()

        animation.save(filename, progress_callback=update_progress, **save_format_kwargs)
        progress_bar.close()
        plt.close(fig)

    def plot_state(self):
        self.plot_current_density(0)

    def get_state(self, state_index: int) -> LatticeState:
        if self.states is None:
            if state_index:
                print("Lattice was not evolved, call evolve() first.")
                return
            return LatticeState(self.density_matrix)
        else:
            return self.states[state_index]

    def get_current_density(self, state_matrix: np.ndarray) -> np.ndarray:
        return (self.H_hop * state_matrix.T).imag

    def plot_current_density(self, state_index: int, ax: matplotlib.axes = None, curl_norm: float = 1, Emax: float=1, curl_pol_norm: float = 1, pol_norm: float = 1, auto_normalize: bool = False) -> None:
        show_plot = ax is None

        lattice_state = self.get_state(state_index)

        if auto_normalize:
            curl_norm = max([max(np.abs(list(self.curl(self.get_current_density(state.density)).values()))) for state in self.states])
            Emax = max([np.abs(self.E(i * self.h)) for i in range(self.steps)])
            pol_norm = max([
                np.linalg.norm(state.polarisation) for state in self.states
            ])
            curl_pol_norm = max([
                np.linalg.norm(state.curl_polarisation) for state in self.states
            ])

        if ax is None:
            _, ax = plt.subplots(figsize=(2 * self.Lx + 2, 2 * self.Ly))
        else:
            ax.clear()
            # ax.set_xlim(2*self.L)
            # ax.set_ylim(2*self.L)

        positions = {(i, j): (j, -i) for i in range(self.Ly) for j in range(self.Lx)}

        # Normalize node colors based on diagonal values
        diag_values = np.real(np.diag(lattice_state.density))
        norm = plt.Normalize(vmin=0.0, vmax=1 / self.N / self.occupation_fraction)
        cmap = plt.get_cmap("Greys_r").reversed()

        # Plot nodes
        for (i, j), (x, y) in positions.items():
            idx = self.Lx * i + j
            color_val = cmap(norm(diag_values[idx]))
            circle = plt.Circle((x, y), 0.3, facecolor=color_val, zorder=2, edgecolor="black")

            if idx in lattice_state.curl.keys():
                curl_val = lattice_state.curl[idx] / curl_norm
                curl_circle = plt.Circle(
                    (x + 0.5, y - 0.5),
                    0.2,
                    facecolor="blue" if curl_val > 0 else "red",
                    zorder=3,
                )

                ax.text(
                    x + 0.5,
                    y - 0.5,
                    f"{curl_val:.2f}",
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=10,
                    zorder=4
                )

                ax.add_patch(curl_circle)

            ax.add_patch(circle)
            ax.text(
                x,
                y,
                f"{diag_values[idx]:.2f}",
                color="white",
                ha="center",
                va="center",
                fontsize=10,
                path_effects=[patheffects.withStroke(linewidth=1, foreground="black")],
            )

        # Normalize current values for thickness
        abs_current = np.abs(lattice_state.current)
        max_current = np.max(abs_current) if np.max(abs_current) > 0 else 1

        # Plot currents as lines between nodes
        for u in range(self.N):
            for v in range(u + 1, self.N):
                if self.H_hop[u, v] != 0:
                    x1, y1 = positions[u // self.Lx, u % self.Lx]
                    x2, y2 = positions[v // self.Lx, v % self.Lx]

                    current_val = lattice_state.current[u, v]

                    linewidth = 3 * abs(current_val) / max_current  # Scale line thickness
                    color = "blue" if current_val > 0 else "red"

                    xval = np.linspace(x1, x2, 8)
                    yval = np.linspace(y1, y2, 8)

                    if x1 == x2:
                        arrow = "v" if current_val < 0 else "^"
                    else:
                        arrow = ">" if current_val < 0 else "<"

                    ax.plot(
                        xval,
                        yval,
                        "-",
                        linewidth=min(50, 5 * linewidth),
                        color=color,
                        alpha=0.2,
                        zorder=1,
                    )
                    ax.plot(
                        xval,
                        yval,
                        arrow,
                        linewidth=linewidth,
                        color=color,
                        alpha=min(1, 5 * linewidth),
                        markersize=(5 + 4 * linewidth),
                        zorder=0,
                    )

        circle = plt.Circle((self.Lx + 1, 0), 0, facecolor="red")
        ax.add_patch(circle)
        arrow_x, arrow_y = self.Lx - 0.1, -1  # Position outside grid

        Emax = max([np.abs(self.E(i * self.h)) for i in range(self.steps)])
        Ey, Ex = 1 / Emax * self.E(state_index * self.h) * self.E_dir

        polarisation = self.polarisation(lattice_state.density) / pol_norm
        polarisation_current = self.polarisation_current(lattice_state.density, self.get_state(state_index - 1).density if state_index > 0 else lattice_state.density) / pol_norm

        curl_polarisation = self.curl_polarisation(lattice_state.curl) / curl_pol_norm
        curl_polarisation_current = self.curl_polarisation_current(lattice_state.curl, self.curl(self.get_current_density(self.get_state(state_index - 1).density)) if state_index > 0 else lattice_state.curl) / curl_pol_norm

        Lattice2D.plot_arrow(ax, arrow_x+0.5, arrow_y, *polarisation, color="black", label="$\\vec P$")
        Lattice2D.plot_arrow(ax, arrow_x+1, arrow_y, *polarisation_current, color="blue", label="$\\frac{\\partial \\vec P}{\\partial t}$")
        Lattice2D.plot_arrow(ax, arrow_x, arrow_y, Ex, Ey, color="red", label="$\\vec E$")
        Lattice2D.plot_arrow(ax, arrow_x+0.5, arrow_y-2, *curl_polarisation, color="green", label="$\\nabla \\times \\vec J$")
        Lattice2D.plot_arrow(ax, arrow_x+0.5, arrow_y-3, *curl_polarisation_current, color="orange", label="$\\frac{\\partial}{\\partial t} (\\nabla \\times \\vec J)$")
        
        ax.plot(arrow_x+3, 0, alpha=0)


        # annotate step
        ax.text(
            self.Lx + 0.5,
            -self.Ly + 1,
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


    def polarisation(self, state_matrix: np.ndarray) -> np.ndarray:
        return np.sum([
            (np.array([i, j]) - self.origin) * state_matrix.diagonal().real[self.Lx * j + i]
            for j in range(self.Ly)
            for i in range(self.Lx)
        ], axis=0)
    

    def polarisation_current(self, state_matrix: np.ndarray, previous_step_state_matrix: np.ndarray) -> np.ndarray:
        return (self.polarisation(state_matrix) - self.polarisation(previous_step_state_matrix)) / self.h


    def curl_polarisation(self, curl_J: np.ndarray) -> np.ndarray:
        return np.sum([
            (np.array([site_index % self.Lx, site_index // self.Lx]) - self.curl_origin) * curl_J[site_index]
            for site_index in curl_J.keys()
        ], axis=0) 
    

    def curl_polarisation_current(self, curl_J: np.ndarray, previous_curl_J: np.ndarray) -> np.ndarray:
        return (self.curl_polarisation(curl_J) - self.curl_polarisation(previous_curl_J)) / self.h


    @staticmethod
    def plot_arrow(ax, x, y, dx, dy, color="black", label=""):
        ax.annotate(
            "",
            xy=(x + dx, y - dy),
            xytext=(x, y),
            arrowprops=dict(arrowstyle="->", color=color, lw=4),
        )

        if label: ax.text(x + 0.2, y+0.2, label, fontsize=14, color=color)

        # plot transparent dot at arrow end
        plt.plot(x + dx, y - dy, alpha=0)


    def curl(self, J: np.ndarray) -> dict[int: np.float64]:
        curl = dict[int: np.float64]()
        # curl_row_length = self.L-cell_width

        for i in range(0, self.Ly - self.cell_height, self.cell_height):
            for j in range(0, self.Lx - self.cell_width, self.cell_width):
                site_index = self.Lx * i + j
                curl[site_index] = sum([J[site_index + di, site_index + dj] for di, dj in self.cell_path])
        return curl
        
    
    def save(self, filename: str) -> None:
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    
    @staticmethod
    def load(filename: str):
        with open(filename, "rb") as f:
            return pickle.load(f)



class BrickwallLattice(Lattice2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.cell_path = np.array([(0, 1), (1, 2), (2, self.Lx+2), (self.Lx+2, self.Lx+1), (self.Lx+1, self.Lx), (self.Lx, 0)])


    def get_brickwall_lattice_sites(self) -> np.ndarray:
        if self.Lx % 2 == 0:
            y_hop_row = np.tile([0, 1], self.Lx//2)
            y_hop = np.concatenate([y_hop_row, 1 - y_hop_row] * (self.Ly // 2 - 1) + [y_hop_row])
        else:
            y_hop = np.tile([0, 1], (self.Ly-1)*self.Lx//2)
            y_hop = np.concatenate([y_hop, [0] * (1 - self.Ly % 2)])
        
        erase_positions = np.diag(y_hop, self.Lx) + np.diag(y_hop, -self.Lx)
        return 1-erase_positions
    
    
    def create_hopping_hamiltonian(self):
        return super().create_hopping_hamiltonian() * self.get_brickwall_lattice_sites()
    
    def curl(self, J: np.ndarray) -> np.ndarray:
        curl = dict()
        # curl_row_length = self.L-cell_width

        for i in range(0, self.Ly - self.cell_height, self.cell_height):
            for j in range(i%2, self.Lx - self.cell_width, self.cell_width):
                site_index = self.Lx * i + j
                curl[site_index] = sum([J[site_index + di, site_index + dj] for di, dj in self.cell_path])
        return curl
    

class SquareLattice(Lattice2D):
    def __init__(self, L: int, *args, **kwargs):
        super().__init__(dimensions=(L, L), *args, **kwargs)
        self.L = L


