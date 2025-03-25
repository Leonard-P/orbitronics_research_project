from lattice import plot_site_grid, Lattice2D, RectangularLatticeGeometry, SimulationParameters, BrickwallLatticeGeometry
import numpy as np
import matplotlib.pyplot as plt


# Define 2D lattice dimensions
Lx, Ly = 30, 30  # 2D lattice size

# Create 2D lattice
l = Lattice2D(
    BrickwallLatticeGeometry((Lx, Ly)), 
    SimulationParameters(t_hop=-1, E_amplitude=1, E_direction=np.array([0, -1]), h=1, T=1, substeps=1)
)

l.plot_hamiltonian()

# Dictionary to store E(kx, ky)
E_k = {}

# Loop over all eigenstates
for i in range(Lx * Ly):
    state = l.energy_states[:, i].reshape(Lx, Ly)  # Reshape to 2D grid
    fft2 = np.fft.fft2(state)  # Compute 2D FFT
    freqs_x = np.fft.fftfreq(Lx)
    freqs_y = np.fft.fftfreq(Ly)

    # Find index of maximum amplitude in FFT (dominant wavevector)
    max_idx = np.unravel_index(np.argmax(np.abs(fft2)), fft2.shape)
    kx, ky = abs(freqs_x[max_idx[0]]), abs(freqs_y[max_idx[1]])

    # Store energy
    E_k[(kx, ky)] = l.eigen_energies[i]

# Convert to arrays for plotting
kx_vals, ky_vals, E_vals = zip(*[(kx, ky, E) for (kx, ky), E in E_k.items()])

# Plot band structure E(kx, ky) in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(kx_vals, ky_vals, E_vals, c=E_vals, cmap='viridis', marker='o')

ax.set_xlabel("$k_x$")
ax.set_ylabel("$k_y$")
ax.set_zlabel("Energy $E$")
ax.set_title("2D Band Structure $E(k_x, k_y)$")

plt.show()
