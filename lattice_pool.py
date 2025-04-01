from lattice import Lattice2D, SimulationParameters, SimulationData, RectangularLatticeGeometry
import multiprocessing as mp
import numpy as np

def smooth_sine_step(t:float, width: float) -> float:
    if t > width:
        return 1
    return np.sin(0.5*np.pi*t/width)**2

def E_sin(t):
    return 0.0001*np.sin(2*t)

def E_sin_step(t):
    return 0.0001*np.sin(2*t)*smooth_sine_step(t, 5)

l = Lattice2D(RectangularLatticeGeometry((12, 25)), SimulationParameters(
    t_hop=-1,
    E_amplitude=E_sin,
    E_direction=np.array([0, -1]),
    h=0.01,
    T=100,
    substeps=5,
))

l2 = Lattice2D(RectangularLatticeGeometry((12, 25)), SimulationParameters(
    t_hop=-1,
    E_amplitude=lambda t: E_sin_step,
    E_direction=np.array([0, -1]),
    h=0.01,
    T=100,
    substeps=5,
))

def simulate(simulation, options):
    simulation.evolve(**options)
    return SimulationData.from_lattice(simulation)

if __name__ == "__main__":
    with mp.Pool(3) as pool:
        res1 = pool.apply_async(simulate, args=(l, {"solver": "rk4"}))
        res2 = pool.apply_async(simulate, args=(l2, {"solver": "rk4", "force_reevolve": True}))
        res3 = pool.apply_async(simulate, args=(l2, {"solver": "rk4", "force_reevolve": True, "decay_time": 2}))
        data1, data2, data3 = res1.get(), res2.get(), res3.get()


    data1.plot_simulation_time_series()
    data2.plot_simulation_time_series()
    data3.plot_simulation_time_series()
    
    data1.plot_simulation_fft()
    data2.plot_simulation_fft()
    data3.plot_simulation_fft()
