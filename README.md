# 2D Tight‑Binding Real‑Space Simulations

This repository contains the realspace_tb module for basic real-space tight-binding simulations. It accompanies a research project on "Boundary-Induced Orbital Hall Effect from Intersite Currents in Finite Honeycomb Lattices". The report PDF is useful for context and details and available on request.

For an overview of the code functionality, see the [quickstart notebook](examples/quickstart.ipynb).

Repository structure:
- /realspace_tb: basic core module for time-domain real-space RK4 evolution.
- /realspace_tb/orbitronics_2d: submodule for simulating the orbital Hall effect in a finite graphene-like system, resulting from the research project.
- /results: data, plots and animations from the report.
- /notebooks: Jupyter notebooks to reproduce the result data and plots.
- /examples: minimal example notebooks demonstrating code features.
- /archive: contains (unorganized) code of the initial module version used for the report simulations.

[Open in Colab](https://colab.research.google.com/github/Leonard-P/orbitronics_research_project/blob/main/notebooks/run_simulations.ipynb)

# realspace_tb module
## Install
Requires Python 3.11+. Install at repository root with:
```bash
pip install -e .
```
## Features
Relevant functions are documented in code. Their use is best demonstrated in the [example notebooks](examples). Documentation below summarizes the features briefly.

### realspace_tb
- Hamiltonian abstract base class. Needs to provide a sparse Hermitian operator via at_time(t). Provides ground_state_density_matrix(fermi_level) method for initial state preparation.
- Observable abstract base class. Needs to provide per-step measurement methods, called at each RK4 step via measure(rho, t, step_index). Provides values as .values and .measurement_times attributes after evolution. 
- Density matrix evolution via RK4NeumannSolver().evolve(rho, H, dt, total_time, observables=[...]). rho is a complex dense array that is updated in place. The RK4 solver integrates the von Neumann equation $\dot{\rho} = -i[H,\rho] + (\rho_0 - \rho)/\tau$.
- Backend/precision: Before instantiating Hamiltonians/Observables, set the backend and precision via tb.backend.set_backend(use_gpu=..., precision={"single","double"}). Uses CuPy instead of NumPy if use_gpu=True and a compatible GPU is available.

## Orbitronics (report module): orbitronics_2d
The realspace_tb.orbitronics_2d submodule implements the system and observables used in the research project.
- HoneycombLatticeGeometry implements the Lattice2DGeometry for a finite honeycomb lattice with nearest-neighbor hopping.
- Field driving: LinearFieldHamiltonian with homogeneous E(t) field via onsite potentials. The RampedACFieldAmplitude class implements a ramped sinusoidal field amplitude.
- Observables: BondCurrentObservable, SiteDensityObservable and PlaquetteOAMObservable record the bond currents, site densities and plaquette orbital angular momenta, respectively. Based on these, OrbitalPolarizationObservable computes the orbital polarization and LatticeFrameObservable records data that can be used for visualizations.
- ohc: Function to compute the orbital Hall conductivity from an array of orbital current values and applied field amplitudes. Orbital current can be approximated as a polarization current by using the (discrete) time derivative of the orbital polarization observable.

## Backend and dtype usage
- The code supports both NumPy (CPU) and CuPy (GPU) backends with single/double precision. This means, all code that is executed during time evolution must be backend- and dtype-aware.
  - Arrays: backend.xp() for np/cp; sparse: backend.xp_sparse().
  - Dtypes: backend.DTYPE (complex), backend.FDTYPE (float).
- Code that needs to be backend/dtype-aware is usually the per-step methods of Hamiltonians (also e.g. the HomogeneousFieldAmplitude) and Observables.
  - Cast scalars to backend dtypes before math (avoid implicit np.float64 upcasts).
  - Provide inputs in correct dtypes (e.g., rho.astype(backend.DTYPE)), and keep all per‑step methods (Hamiltonian/Observables) backend‑aware.
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
        E0=1e-3, omega=1.0, T_ramp=np.pi, 
        direction=np.array([0.0,1.0]),
    ),
)

rho = H.ground_state_density_matrix(fermi_level=0.0)
oam_obs = tb_orb.observables.OrbitalPolarizationObservable(H.geometry)

tb.RK4NeumannSolver().evolve(rho, H, dt=0.01, total_time=1.0, tau=20.0, observables=[oam_obs])

plt.plot(oam_obs.measurement_times, oam_obs.values[:,0], label="P_{OAM,x}")
plt.xlabel("Time"); plt.ylabel("Orbital Polarization")
```
