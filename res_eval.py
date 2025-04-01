from lattice import Lattice2D, RectangularLatticeGeometry, SimulationParameters, SimulationData 
import numpy as np
import matplotlib.pyplot as plt 

    # l.save(f"experiments/01-04_response/l_{int(100*nw)}w_100.lattice")
    # data = SimulationData.from_lattice(l, omega=2*0.08975714*nw)
    # return np.array(data.M).mean()


def smooth_sine_step(t:float, width: float) -> float:
    if t > width:
        return 1
    return np.sin(0.5*np.pi*t/width)**2


def comp_excitation(nw):
    def E(t):
        return 0.0001*np.sin(2*0.08975714*nw*t) * smooth_sine_step(t, 5/nw)
    
    l = Lattice2D(
        RectangularLatticeGeometry(
            (16, 34),
        ),
        SimulationParameters(
            t_hop=-1,
            E_amplitude=E,
            E_direction=np.array([0, -1]),
            h=1/nw,
            T=20/nw,
            substeps=200,
        ),
    )

    l.evolve(solver="rk4", decay_time=5)
    l.save(f"experiments/01-04_response/l_{int(100*nw)}w_100.lattice")
    data = SimulationData.from_lattice(l, omega=2*0.08975714*nw)
    return np.array(data.M).mean()




# omega_values = [0.1, 0.5, 0.8, 1, 1.2, 1.5, 2, 5] #, 10, 20, 50, 110, 250]
omega_values = [1, 2, 5, 10, 15, 20, 50, 100]

data = []
response = []

for omega in omega_values:
    l = Lattice2D.load(f"experiments/01-04_response/l_{int(100*omega)}w_100_2.lattice")
    data.append(SimulationData.from_lattice(l, omega=2*0.08975714*omega))
    response.append((np.array(data[-1].M_current)**2).max())

plt.plot(omega_values, response, "o-", label=r"Excitation vs. $\omega$")
plt.show()