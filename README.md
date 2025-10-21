# 2D Tight Binding Simulations
* numerically solve von-Neumann equation for 2D Lattices
* supports Hamiltonians of shape $H_{\rm hop} + E(t) \cdot H_{\rm onsite}$
* $H_{\rm hop}$ has non-zero matrix elements for hoppings along lattice with square/brickwall/hexagonal cells with constant energy $t_{\rm hop}$

## Code Example
```python
import numpy as np
from lattice import Lattice2D, SimulationParameters, BrickwallLatticeGeometry

l = Lattice2D(
    geometry=BrickwallLatticeGeometry((7, 10)), # 7x10 sites, connected brickwall-like
    simulation_parameters=SimulationParameters(
        t_hop=1,                         # hopping energy
        E_amplitude=lambda t: np.sin(t), # driving field with oscillating amplitude
        E_direction=np.array([1, 0]),    # field in x-direction
        h=0.01,                          # time spacing between simulated states
        T=1,                             
        initial_occupation=0.5,      
        substeps=10)
    )
l.evolve(solver="rk4")                   # run simulation
l.save("brickwall_simulation.lattice")
```

## Object-Oriented $\rightarrow$ Extends To Other 2D Geometries
```python
import numpy as np
from lattice import Lattice2D, SimulationParameters, BrickwallLatticeGeometry


class HexagonalLatticeGeometry(BrickwallLatticeGeometry):
    """2D Graphene-like hexagonal structure"""

    def __init__(self, dimensions):
        super().__init__(dimensions)     
        self.row_height = 1.5
        self.col_width = np.sqrt(3) / 2
        self.origin = np.array([(self.Lx-1) * self.col_width, (self.Ly-1) * self.row_height]) / 2

    def site_to_position(self, site_index: int) -> tuple[float, float]:
        """map each site to 2D (x, y) position"""
        row, col = divmod(site_index, self.Lx)
        y_offset = 0.25 * (-1) ** ((col + row) % 2)

        x = self.col_width * (site_index % self.Lx)
        y = self.row_height * row + y_offset

        return x, y

    def cell_field_gradient(self, f: dict[int, float]) -> dict[int, np.ndarray]:
        # TODO
        raise NotImplementedError


l = Lattice2D(HexagonalLatticeGeometry((10, 10)), SimulationParameters(...))
l.evolve(solver="rk4")
l.plot_current_density(-1, auto_normalize=True)
```