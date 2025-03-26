from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib.pyplot as plt
from lattice import Lattice2D, RectangularLatticeGeometry, SimulationParameters

N = 10

def E_osc(t, w, A):
    return A * np.sin(w * t)


def get_1D_momentum_distributions_osc(T=10, h=0.1, substeps=50, **pulse_kwargs):
    l = Lattice2D(
        RectangularLatticeGeometry((N, 1)),
        SimulationParameters(
            t_hop=-1,
            E_amplitude=lambda t: E_osc(t, **pulse_kwargs),
            E_direction=np.array([1, 0]),
            h=h,
            T=T,
            substeps=substeps,
            initial_occupation=1 / (N),
        ),
    )

    rho_energy_basis = np.diag(np.zeros(N))
    rho_energy_basis[N//2, N//2] = 1
    l.density_matrix = l.energy_states @ rho_energy_basis @ l.energy_states.T.conj()

    l.evolve(solver="rk4")

    x = np.arange(1, N + 1)

    # Allowed discrete momenta for a 1D lattice with open boundaries:
    # k_n = n*pi/(N+1), with n=1,...,N.
    k_list = np.array([np.pi * n / (N + 1) for n in range(1, N + 1)])

    momentum_distributions = []

    for state in l.states:
        rho_t = state.density
        P_k = np.zeros(N)
        for idx, k in enumerate(k_list):
            # Construct the 1D sine basis function for momentum k:
            phi_k = np.sqrt(2 / (N + 1)) * np.sin(k * x)
            # Projection: <phi_k|rho|phi_k>
            P_k[idx] = np.real(np.dot(phi_k.conj(), rho_t @ phi_k))
        momentum_distributions.append(P_k)

    return np.array(momentum_distributions)

# Define your function that computes the excitation response for a given frequency.
# This function wraps your call to get_1D_momentum_distributions_osc.
def compute_excitation(w):
    # Here, T is set as 5*pi/w
    T = 5 * np.pi / w
    # The function get_1D_momentum_distributions_osc must be accessible here.
    # It should return an array with shape (n_time_steps, n_momentum_modes)
    momentum_distributions = get_1D_momentum_distributions_osc(
        T=T, h=T/100, substeps=200, w=w, A=0.01
    )
    # Measure total excitation: using the mean over last 5 time steps at momentum index 5.
    total_excitation = 1 - momentum_distributions[-5:, 5].mean()
    return total_excitation



if __name__ == "__main__":
    omega_values = np.linspace(0.01, 2.5, 200)  # Sweep from low to high frequencies


    # Use Pool to map compute_excitation over omega_values.
    num_cores = cpu_count()
    print(num_cores)
    with Pool(processes=num_cores) as pool:
        responses = pool.map(compute_excitation, omega_values)

    import pickle

    # Save the results to a file.
    with open("excitation_responses.pkl", "wb") as f:
        pickle.dump((omega_values, responses), f)

    # Plot the results.
    plt.plot(omega_values, responses, "o-", label=r"Excitation vs. $\omega$")
    plt.xlabel(r"Field Frequency $\omega$")
    plt.ylabel("Excitation Response")
    plt.title("Excitation Response vs. Field Frequency")
    plt.legend()
    plt.xscale("log")
    plt.show()
