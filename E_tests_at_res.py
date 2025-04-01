from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib.pyplot as plt
from lattice import Lattice2D, RectangularLatticeGeometry, SimulationParameters, SimulationData

def smooth_sine_step(t:float, width: float) -> float:
    if t > width:
        return 1
    return np.sin(0.5*np.pi*t/width)**2



def comp_excitation(nE):
    def E(t):
        return nE*np.sin(2*0.08975714*t) * smooth_sine_step(t, 5)
    
    l = Lattice2D(
        RectangularLatticeGeometry(
            (16, 34),
        ),
        SimulationParameters(
            t_hop=-1,
            E_amplitude=E,
            E_direction=np.array([0, -1]),
            h=1,
            T=20,
            substeps=200,
        ),
    )

    l.evolve(solver="rk4", decay_time=5)
    l.save(f"experiments/01-04_response/l_{int(100000*nE)}E_100000.lattice")
    data = SimulationData.from_lattice(l, omega=2*0.08975714)
    return (np.array(data.M)**2).mean()


if __name__ == "__main__":
    E_values = [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1]


    # Use Pool to map compute_excitation over omega_values.
    num_cores = cpu_count()
    print(num_cores)
    with Pool(processes=num_cores) as pool:
        responses = pool.map(comp_excitation, E_values)

    import pickle

    # Save the results to a file.
    with open("experiments/01-04_response/responsesE.pkl", "wb") as f:
        pickle.dump((E_values, responses), f)

    # Plot the results.
    plt.plot(E_values, responses, "o-", label=r"Excitation vs. $\omega$")
    plt.xlabel(r"E $\omega$")
    plt.ylabel("Excitation Response")
    plt.title("Excitation Response vs. E")
    plt.legend()
    plt.show()
