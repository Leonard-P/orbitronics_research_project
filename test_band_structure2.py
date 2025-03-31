from lattice import Lattice2D, SimulationParameters
from lattice import BrickwallLatticeGeometry
import numpy as np
from scipy.fft import dst
import matplotlib.pyplot as plt

Nx_sim, Ny_sim = 40, 11

l = Lattice2D(BrickwallLatticeGeometry((Nx_sim, Ny_sim)), SimulationParameters.default())


from scipy.fft import dst
# Assume l is your Lattice2D object, initialized and diagonalized
# Assume Nx_sim, Ny_sim = 40, 25 were used for l.geometry
# Assume l.energy_states and l.eigen_energies exist

xyz = []
for i_state in range(l.N):

    eigenvector_flat = l.energy_states[:, i_state]

    # --- This part requires knowledge of your specific lattice indexing ---
    # --- It ASSUMES row-major order and extracts ONE sublattice (e.g., A) ---
    # --- This specific extraction might be INCORRECT for your Brickwall class ---

    Ly, Lx = l.geometry.Ly, l.geometry.Lx # e.g., 25, 40
    eigenstate_matrix = eigenvector_flat.reshape((Ly, Lx))

    # Example extraction: Take elements where (row%2 == col%2) -> Sublattice A
    # Arrange them into an approx Ly x (Lx/2) matrix
    Nx_uc = Lx // 2
    Ny_uc = Ly
    psi_A_matrix = np.zeros((Ny_uc, Nx_uc), dtype=eigenvector_flat.dtype)
    for r in range(Ny_uc):
        col_a_idx = 0
        for c in range(Lx):
            shifted_c = (c + (r % 2)) % Lx
            if (r % 2) == (shifted_c % 2): # If it's an A site
                if col_a_idx < Nx_uc:
                    psi_A_matrix[r, col_a_idx] = eigenstate_matrix[r, c]
                    col_a_idx += 1
                else: # Should not happen if Lx is even
                    pass


    # --- Use slices of the sublattice matrix for DST ---

    # Slice along a row (approx constant y) -> estimate kx
    row_idx = psi_A_matrix.shape[0] // 2
    psi_slice_x = psi_A_matrix[row_idx, :]
    L_slice_x = len(psi_slice_x)
    if L_slice_x > 1:
        dst_x = dst(psi_slice_x, type=1)
        dominant_mx = np.argmax(np.abs(dst_x)) + 1
        # kx related value: mx * pi / (Num A sites along X + 1)
        kx_est = dominant_mx * np.pi / (L_slice_x + 1)
    else:
        kx_est = np.nan

    # Slice along a column (approx constant x) -> estimate ky
    col_idx = psi_A_matrix.shape[1] // 2
    psi_slice_y = psi_A_matrix[:, col_idx]
    L_slice_y = len(psi_slice_y)
    if L_slice_y > 1:
        dst_y = dst(psi_slice_y, type=1)
        dominant_my = np.argmax(np.abs(dst_y)) + 1
        # ky related value: my * pi / (Num A sites along Y + 1)
        ky_est = dominant_my * np.pi / (L_slice_y + 1)
    else:
        ky_est = np.nan

    print(f"State {i_state}:")
    print(f"  kx (estimated from row {row_idx} of A-sublattice) ~ {kx_est:.4f}")
    print(f"  ky (estimated from col {col_idx} of A-sublattice) ~ {ky_est:.4f}")


    xyz.append([kx_est, ky_est, l.eigen_energies[i_state]])

    # NOTE:
    # - This provides kx, ky relative to the A-sublattice grid dimensions (Nx_uc, Ny_uc).
    # - The accuracy depends heavily on correct sublattice extraction and reshaping.
    # - The simple middle slice might not be representative for all states.
    # - Using 2D DST on psi_A_matrix (as in previous answers) is generally more robust.


xyz = np.array(xyz)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2])
ax.set_xlabel('kx')
ax.set_ylabel('ky')

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

kx_th = np.linspace(0.001, np.pi, 100) # Start slightly > 0 to avoid potential issues at Gamma
ky_th = np.linspace(np.pi, 0.001, 100)
KX_th, KY_th = np.meshgrid(kx_th, ky_th)
E_plus, E_minus = graphene_dispersion(KX_th, KY_th, l.t_hop, 1/np.sqrt(3))
ax.plot_wireframe(KX_th, KY_th, E_plus, color='grey', alpha=0.4, linewidth=0.7, label='Theory (+ band)')
ax.plot_wireframe(KX_th, KY_th, E_minus, color='lightblue', alpha=0.4, linewidth=0.7, label='Theory (- band)')


plt.show()