import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Define reciprocal lattice vectors for a hexagonal lattice
a = 1.0  # Lattice constant
b1 = (2 * np.pi / (a * np.sqrt(3))) * np.array([np.sqrt(3)/2, -1/2])
b2 = (2 * np.pi / (a * np.sqrt(3))) * np.array([0, 1])

# High symmetry points in the Brillouin zone
Gamma = np.array([0, 0])
K = (2/3) * b1 + (1/3) * b2
M = (0.5) * b1 + (0.5) * b2

# Path along high symmetry points
k_path = np.linspace(Gamma, K, 50).tolist() + np.linspace(K, M, 50).tolist() + np.linspace(M, Gamma, 50).tolist()
k_path = np.array(k_path)

# Function to compute H(k) for a given tight-binding Hamiltonian
def compute_hk(H_real, k, R_vectors):
    """Constructs the Bloch Hamiltonian H(k) using hopping terms."""
    H_k = np.zeros(H_real.shape, dtype=complex)
    for i in range(H_real.shape[0]):
        for j in range(H_real.shape[1]):
            H_k[i, j] = sum(H_real[i, j] * np.exp(1j * np.dot(k, R)) for R in R_vectors)
    return H_k

# Example tight-binding Hamiltonian (nearest-neighbor hexagonal lattice)
def hexagonal_tb_hamiltonian():
    """Returns a simple tight-binding Hamiltonian and hopping vectors."""
    from lattice import Lattice2D, HexagonalLatticeGeometry, SimulationParameters
    t = -1  # Hopping parameter
    # H = np.array([[0, t, t],
    #               [t, 0, t],
    #               [t, t, 0]])

    H = Lattice2D(HexagonalLatticeGeometry((10, 10)), SimulationParameters.default()).H_hop
    
    R_vectors = [np.array([0, 0]), b1, b2]  # Nearest-neighbor connections
    return H, R_vectors

# Compute and plot the band structure
def plot_band_structure():
    H_real, R_vectors = hexagonal_tb_hamiltonian()
    energies = []
    
    for k in k_path:
        H_k = compute_hk(H_real, k, R_vectors)
        eigvals, _ = eigh(H_k)
        energies.append(eigvals)
    
    energies = np.array(energies)
    
    plt.figure(figsize=(6, 5))
    for i in range(energies.shape[1]):
        plt.plot(energies[:, i], 'k-')
    
    plt.xticks([0, 50, 100, 150], [r'$\Gamma$', 'K', 'M', r'$\Gamma$'])
    plt.ylabel('Energy (eV)')
    plt.title('Band Structure of Hexagonal Lattice')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

plot_band_structure()
