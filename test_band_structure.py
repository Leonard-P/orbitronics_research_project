import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dst
from mpl_toolkits.mplot3d import Axes3D
from lattice import Lattice2D, BrickwallLatticeGeometry, SimulationParameters
# For interactive plots in Jupyter environments (choose one):
# %matplotlib notebook
# or use pip install ipympl and then:
# %matplotlib widget

# Assume the project_to_sine_basis function from the previous answer is available:
def project_to_sine_basis(eigenstate_matrix: np.ndarray) -> np.ndarray:
    """
    Projects a 2D eigenstate defined on an Nx x Ny grid onto a 2D sine basis
    using DST-I. Returns the complex amplitudes.
    (Implementation from the previous response)
    """
    if not isinstance(eigenstate_matrix, np.ndarray) or eigenstate_matrix.ndim != 2:
        raise ValueError("Input must be a 2D numpy array.")

    Nx, Ny = eigenstate_matrix.shape

    if Nx == 0 or Ny == 0:
        return np.zeros((Nx, Ny), dtype=eigenstate_matrix.dtype)

    # Apply 2D DST-I
    dst_y = dst(eigenstate_matrix, type=1, axis=1)
    dst_xy = dst(dst_y, type=1, axis=0)

    # Apply normalization factor for orthonormal basis projection
    norm_factor = np.sqrt((Nx + 1) * (Ny + 1))
    amplitudes = dst_xy / norm_factor

    return amplitudes

def graphene_dispersion(kx, ky, t, a):
    # Factor used in the dispersion relation argument
    # This form corresponds to a common choice of lattice vectors
    term1 = 2 * np.cos(ky * a * np.sqrt(3) / 2)
    term2 = np.cos(kx * a * 3 / 2)

    # Handle potential numerical issues inside sqrt for real results
    f_k_sq = 1 + term1**2 + 2 * term1 * term2 # |f(k)|^2
    # Clamp values slightly below zero to zero before sqrt
    f_k_sq = np.maximum(f_k_sq, 0)

    energy_magnitude = np.abs(t) * np.sqrt(f_k_sq)

    # E = +/- energy_magnitude (assuming onsite energy is zero)
    return energy_magnitude, -energy_magnitude

# --- Main Analysis and Plotting ---


# 1. Initialize your Lattice object (replace with your actual initialization)
Nx_sim, Ny_sim = 19, 10
l = Lattice2D(BrickwallLatticeGeometry((Nx_sim, Ny_sim)), SimulationParameters.default())

l.plot_hamiltonian()

# 2. Prepare lists to store plot data
kx_list = []
ky_list = []
energy_list = []

print(f"\nProcessing {len(l.eigen_energies)} eigenstates...")

# 3. Loop through eigenstates and eigenvalues
# Ensure eigenvalues are sorted and eigenvectors correspond correctly
# l.energy_states has eigenvectors as columns
for i, energy in enumerate(l.eigen_energies):
    eigenvector_flat = l.energy_states[:, i]

    # Ensure the eigenvector is normalized (optional but good practice)
    # eigenvector_flat = eigenvector_flat / np.linalg.norm(eigenvector_flat)

    # Reshape the flat eigenvector into an Nx x Ny matrix
    # IMPORTANT: Assumes the eigenvector stores site data in row-major (C-style) order.
    # If your lattice indexing is different (e.g., column-major), adjust reshape accordingly.
    try:
        eigenstate_matrix = eigenvector_flat.reshape((l.geometry.Ly, l.geometry.Lx))
    except ValueError as e:
        print(f"Error reshaping eigenvector {i}: {e}. Check Nx, Ny and eigenvector length.")
        continue

    # Project the eigenstate onto the sine basis
    sine_amplitudes = project_to_sine_basis(eigenstate_matrix)

    # Find the indices (mx-1, my-1) of the maximum absolute amplitude
    if sine_amplitudes.size > 0:
        max_amp_index_flat = np.argmax(np.abs(sine_amplitudes))
        # Convert flat index to 2D index (index_mx corresponds to axis 0, index_my to axis 1)
        idx_my, idx_mx = np.unravel_index(max_amp_index_flat, sine_amplitudes.shape)
    else:
        idx_mx, idx_my = -1, -1 # Handle empty case

    if idx_mx != -1:
        # Convert indices to mode numbers (mx = 1..Nx, my = 1..Ny)
        mx = idx_mx + 1
        my = idx_my + 1

        # Calculate the corresponding kx and ky
        kx = mx * np.pi / (l.geometry.Lx + 1)
        ky = my * np.pi / (l.geometry.Ly + 1)

    
        # Store the data for plotting
        kx_list.append(kx)
        ky_list.append(ky)
        energy_list.append(energy)
    else:
         print(f"Warning: Could not process eigenstate {i} (empty projection?).")


print("Processing complete.")

# 4. Create the 3D Band Structure Plot
print("Plotting effective band structure...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

k_points = 50 # Number of points along each k-axis for the surface
# Generate k values covering the range shown by the simulation data (approx 0 to pi)
kx_th = np.linspace(0.001, np.pi, k_points) # Start slightly > 0 to avoid potential issues at Gamma
ky_th = np.linspace(0.001, np.pi, k_points)
KX_th, KY_th = np.meshgrid(kx_th, ky_th)
E_plus, E_minus = graphene_dispersion(KX_th, KY_th, l.t_hop, 2/np.sqrt(3))
ax.plot_wireframe(KX_th, KY_th, E_plus, color='grey', alpha=0.4, linewidth=0.7, label='Theory (+ band)')
ax.plot_wireframe(KX_th, KY_th, E_minus, color='lightblue', alpha=0.4, linewidth=0.7, label='Theory (- band)')

# Scatter plot: kx, ky, Energy
# Color points by energy
sc = ax.scatter(kx_list, ky_list, energy_list,
                c=energy_list, cmap='viridis', marker='.', alpha=0.8)

ax.set_xlabel('$k_x = m_x \pi / (N_x+1)$')
ax.set_ylabel('$k_y = m_y \pi / (N_y+1)$')
ax.set_zlabel('Energy $E$')

# Set reasonable k limits (0 to pi)
ax.set_xlim(0, np.pi * 1.05)
ax.set_ylim(0, np.pi * 1.05)

# Add a color bar
cbar = fig.colorbar(sc)
cbar.set_label('Energy $E$')

# Optional: Adjust view angle for better visibility
# ax.view_init(elev=25, azim=-75)

plt.show()
print("Plot displayed.")