Todo:
- make new module supporting different tight binding hamiltonians
  - support DTYPE in every array
  - reduce backend scope. Only use in solver and hamiltonian
  - add ohc funcionality
  - add global density observable
  - add bond_current 2d observable
  - add plaquette_oam observable and make the orbital polarization observable inherit from it to use the row, col, edge_vec setup
  - add animation observable that combines density, bond current, plaquette oam and provides a single frame plot and video save function.



Idea:
- Solver class RK4
  - maybe triple_fold to analyze convergence?
- TightBindingHamiltonian classes
  - has site index to position mapping
  - obtain bravais lattice indices
  - has method to build Hamiltonian
  - has origin property
  - what about rho_0? Could be built manually from eigenstates
- Observable, general have setup and measure methods
  - Animation and Pol observable that inherit from a curl observable?
  - idempotency tracker?
  - how do observables depend on hamiltonian
- plotting/animation utils in submodule - plot a frame and animate functions

- What about 2D vs 3D?
- maybe keep solver, obsABC, idempotABC, tbHamABC rooted, have a 2D submodule for 2DHex, poss. 2D Rect H, PlaquetteObs, animation export

Roadmap:
Start everything on CPU.
- start rk4 solver
- start hamiltonian class
- observables
- plotting/animation utils
- submodule structure
- readme and examples
- gpu versions
- tests

name: realspace_tb.orbitronics_2d?

realspace_tb
- rk4.py
- hamiltonian.py
- observable.py
- orbitronics_2d/
  - hamiltonian_2d.py
  - orbital_polarisation_2dhex.py
  - animation_2d.py