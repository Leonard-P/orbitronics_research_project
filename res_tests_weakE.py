from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib.pyplot as plt
from lattice import Lattice2D, RectangularLatticeGeometry, SimulationParameters, SimulationData

def smooth_sine_step(t:float, width: float) -> float:
    if t > width:
        return 1
    return np.sin(0.5*np.pi*t/width)**2



def comp_excitation(nw):
    def E(t):
        return 0.0001*np.sin(0.36959882*nw*t) * smooth_sine_step(t, 5/nw)
    
    l = Lattice2D(
        RectangularLatticeGeometry(
            (10, 16),
        ),
        SimulationParameters(
            t_hop=-1,
            E_amplitude=E,
            E_direction=np.array([0, -1]),
            h=1/nw,
            T=2/nw,
            substeps=200,
        ),
    )

    l.evolve(solver="rk4", decay_time=5)
    l.save(f"experiments/01-04_response/l_{int(100*nw)}w_100_2.lattice")
    data = SimulationData.from_lattice(l, omega=0.36959882*nw)
    return np.abs(np.array(data.M)).mean()


if __name__ == "__main__":
    omega_values = [0.1, 0.8, 1, 1.5, 1.8,  2, 2.2, 3, 5, 10, 15, 20, 50, 100]


    # Use Pool to map compute_excitation over omega_values.
    num_cores = cpu_count()
    print(f"Running on {num_cores=}")
    with Pool(processes=num_cores) as pool:
        responses = pool.map(comp_excitation, omega_values)

    import pickle

    # Save the results to a file.
    with open("experiments/01-04_response/responsesG.pkl", "wb") as f:
        pickle.dump((omega_values, responses), f)

    # Plot the results.
    plt.plot(omega_values, responses, "o-", label=r"Excitation vs. $\omega$")
    plt.xlabel(r"Field Frequency $\omega$")
    plt.ylabel("Excitation Response")
    plt.title("Excitation Response vs. Field Frequency")
    plt.legend()
    plt.show()
