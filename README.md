# 2D Tight‑Binding Real‑Space Simulations
This repository contains the realspace_tb module for basic real-space tight-binding simulations. It accompanies a research project on "Boundary-Induced Orbital Hall Effect from Intersite Currents in Finite Honeycomb Lattices".

For an overview of the code functionality, see the [quickstart notebook](examples/quickstart.ipynb).

Repository structure:
- /realspace_tb: basic module for real-space RK4 time evolution.
- /realspace_tb/orbitronics_2d: submodule for simulating the orbital Hall effect in a finite graphene-like system, resulting from the research project.
- /results: data, plots and animations from the report.
- /notebooks: Jupyter notebooks to reproduce the result data and plots.
- /examples: minimal example notebooks demonstrating code features.
- /archive: contains (unorganized) code of the initial module version used for the report simulations.

[Open in Colab](https://colab.research.google.com/github/Leonard-P/orbitronics_research_project/blob/main/notebooks/run_simulations.ipynb)

# realspace_tb module
## Units
Module assumes natural units with $\hbar=1$, in orbitronics_2d submodule also the honeycomb nearest neighbor distance $a_\mathrm{NN}=1$, the hopping energy $t_\mathrm{hop}=1$ and electron charge $e=1$ are used. The default electron mass (adjust via the ```electron_mass``` parameter of the respective observables) $m_e = 0.741$ was calculated with $t_\mathrm{hop} = 2.8\text{ eV}, a_\mathrm{NN} = 0.142\text{ nm}$ from graphene.

## Install
Requires Python 3.11+. Install at repository root with:
```bash
pip install -e .
```
## Usage
Relevant functions are documented in code. Their use is best demonstrated in the [example notebooks](examples). The text below summarizes the features briefly, but for more details see the code docstrings and example notebooks.

### realspace_tb
- ```realspace_tb.RK4NeumannSolver().evolve(rho, H, dt, total_time, tau, observables=[...])```: Evolves the density matrix rho under Hamiltonian H and measures the given observables at each RK4 step. The RK4 solver integrates the von Neumann equation $\dot{\rho} = -i[H,\rho] + (\rho_0 - \rho)/\tau$. For this, provide:
- Hamiltonian H that implements the ```realspace_tb.Hamiltonian()``` abstract base class (that provides the sparse Hermitian operator at time t via ```H.at_time(t)```).
- Initial density matrix rho as a complex dense array. Hamiltonian base class provides ```H.ground_state_density_matrix(fermi_level)``` to compute the zero-temperature many-body ground state density matrix from eigenstates.
- List of observables that implement the ```realspace_tb.Observable()``` abstract base class (that provides per-step measurement methods, called at each RK4 step via ```measure(rho, t, step_index)```). The measured values are stored in the observable's ```.values``` and ```.measurement_times``` attributes after evolution.
- Backend/precision: Before instantiating Hamiltonians/Observables, set the backend and precision via tb.backend.set_backend(use_gpu=..., precision={"single","double"}). Uses CuPy instead of NumPy if use_gpu=True and a compatible GPU is available.

## Orbitronics (report module): orbitronics_2d
The realspace_tb.orbitronics_2d submodule implements the system and observables used in the research project, namely a finite honeycomb lattice with nearest-neighbor hopping under a time-dependent homogeneous electric field. The OrbitalPolarization is observed to estimate the orbital Hall conductivity (where the orbital angular momentum is approximated from single-plaquette intersite currents).
- ```realspace_tb.orbitronics_2d.HoneycombLatticeGeometry()``` implements the ```realspace_tb.orbitronics_2d.Lattice2DGeometry``` for a finite honeycomb lattice with nearest-neighbor hopping. Class contains information about lattice sites, bonds and plaquettes.
- Field driving: ```realspace_tb.orbitronics_2d.LinearFieldHamiltonian``` implements the Hamiltonian class with a spatially homogeneous field via onsite potentials. The amplitude is given by e.g. the ```realspace_tb.orbitronics_2d.RampedACFieldAmplitude()``` class as a ramped sinusoidal field amplitude. Hence it depends on the provided Lattice2DGeometry and FieldAmplitude.
- Observables in ```realspace_tb.orbitronics_2d.observables```: ```BondCurrentObservable```, ```SiteDensityObservable``` and ```PlaquetteOAMObservable``` record the bond currents, site densities and plaquette-wise local orbital angular momenta, respectively. Based on these, ```OrbitalPolarizationObservable``` computes the orbital polarization and ```LatticeFrameObservable``` records data that can be used for visualizations.
- ```realspace_tb.orbitronics_2d.ohc()```: Function to compute the orbital Hall conductivity from an array of orbital current values and applied field amplitudes (simply as the ratio of fourier coefficients at provided frequency). Orbital current can be approximated as a polarization current by using the (discrete) time derivative of the ```OrbitalPolarizationObservable``` measurements.

## Backend and dtype usage
- The code supports both NumPy (CPU) and CuPy (GPU) backends with single/double precision. This means, all code that is executed during time evolution must be backend- and dtype-aware.
  - Arrays: backend.xp() for np/cp; sparse: backend.xp_sparse() for scipy.sparse/cupy.sparse.
  - Dtypes: backend.DTYPE (complex), backend.FDTYPE (float).
- Code that needs to be backend/dtype-aware is usually the per-step methods of Hamiltonians (also e.g. the HomogeneousFieldAmplitude) and Observables.
- Existing module code does not need additional backend awareness, as long as fields/Hamiltonians are constructed after calling tb.backend.set_backend(...).

## Typing
- The code passes mypy type checking.
- However, mypy does not verify CPU/GPU device transfers.

## Examples
See the [example notebooks](examples) for usage examples. A minimal example is shown below.
```python
import realspace_tb as tb
from realspace_tb import orbitronics_2d as tb_orb
import numpy as np
import matplotlib.pyplot as plt

tb.backend.set_backend(use_gpu=False, precision="double") # optional, default is CPU double

H = tb_orb.LinearFieldHamiltonian(
    tb_orb.HoneycombLatticeGeometry(Lx=6, Ly=6),
    tb_orb.RampedACFieldAmplitude(
        E0=1e-3, omega=1.0, T_ramp=6*np.pi, 
        direction=np.array([0.0, 1.0]),
    ),
)

rho = H.ground_state_density_matrix(fermi_level=0.0)
oam_obs = tb_orb.observables.OrbitalPolarizationObservable(H.geometry)

tb.RK4NeumannSolver().evolve(rho, H, dt=0.01, total_time=20*np.pi, tau=10.0, observables=[oam_obs])

plt.plot(oam_obs.measurement_times, oam_obs.values[:,0])
plt.xlabel("Time"); plt.ylabel("Orbital Polarization $P_{OAM,x} ($")
plt.show()
```
