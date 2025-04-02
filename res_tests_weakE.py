from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib.pyplot as plt
from lattice import Lattice2D, RectangularLatticeGeometry, SimulationParameters, SimulationData, HexagonalLatticeGeometry


def smooth_sine_step(t: float, width: float) -> float:
    if t > width:
        return 1
    return np.sin(0.5 * np.pi * t / width) ** 2


def comp_excitation(nw):
    Ny = 40
    omega = np.sqrt(3) * np.pi / (1.5 * (Ny + 1))  # v_F = a t_hop = sqrt(3) a_nn = sqrt(3), L = 1.5(Ny + 1)

    def E(t):
        return 0.0001 * np.sin(omega * nw * t) * smooth_sine_step(t, 10 / nw)

    l = Lattice2D(
        HexagonalLatticeGeometry(
            (15, Ny),
        ),
        SimulationParameters(
            t_hop=-1,
            E_amplitude=E,
            E_direction=np.array([0, -1]),
            h=1 / nw,
            T=60 / nw,
            substeps=200,
        ),
    )

    l.evolve(solver="rk4", decay_time=7)
    l.save(f"experiments/01-04_response/l_{int(100*nw)}w_100_hx_small.lattice")
    data = SimulationData.from_lattice(l, omega=omega * nw)
    return np.abs(np.array(data.M)).mean()


if __name__ == "__main__":
    omega_values = np.concatenate(
        [
            np.linspace(0.1, 2.0, 8),
        ]
    )

    # Use Pool to map compute_excitation over omega_values.
    num_cores = cpu_count()
    print(f"Running on {num_cores=}")
    with Pool(processes=num_cores) as pool:
        responses = pool.map(comp_excitation, omega_values)

    import pickle

    # Save the results to a file.
    with open("experiments/01-04_response/responses_hx_small_stronger_damp.pkl", "wb") as f:
        pickle.dump((omega_values, responses), f)

    # with open("experiments/01-04_response/responses.pkl", "rb") as f:
    #     omega_values, responses = pickle.load(f)

    # Plot the results.
    plt.plot(omega_values, responses, "o-", label=r"Excitation vs. $\omega$")
    plt.xlabel(r"Field Frequency $\omega$")
    plt.ylabel("Excitation Response")
    plt.title("Excitation Response vs. Field Frequency")
    plt.legend()
    plt.show()
