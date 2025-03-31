# TODOs

## bugs/fixes/mistakes

### IMPORTANT: geometry fixes for multiatom basis
- [ ] think through: Lattice width vs. nearest neighbor distance vs. t_hop vs. norm area for polarisation
  - possible solution: t_hop = 1, lattice width = 1, nearest neighbor dist != 1, cell_area param
- [ ] get sublattice sites
  - classify param in site_to_pos? -> Hex lattice is just Brickwall with vertically shifted sublattices
  - split list over whole lattice in sublists?

### theory checks
- mode spacing: since bandwidth finite, probably correct. 
- check if current resonates at \Delta E_y 
- check resonance also for other basis
- check 2D band width
- check required field - does it depend on length?
  - check max polarisation for different sine field amplitudes
  - rerun check at double length to see if smaller field required

### less important
- [x] Simulations with sine E-field  
- [x] units [solved for now] 
- [x] flux gradient  
- [ ] make sure to divide by Area
- [ ] type annotations float/int for site to position
- [ ] right hand coordinates: positive charge along positive z direction
- [ ] negative t_hop!!

## new module functionality
- [x] RK4 solver  
- [x] move origin param to LatticeGeometry()  
- [ ] implement charge param != 1  
- [ ] defect scattering experiments by removing sites

## experiment runs
- [~] Smaller frequencies of E but with \(T/2 < T_{\text{reflection}}\)  

## smaller implementations/checks
- [ ] define orbital charge, get sum
- [ ] define total positive and negative charge.
- [ ] transform / convolve OR poissson equation
- [ ] type annotations float/int for site to position
- [ ] change curl direction + current sign
- [ ] bias drop ~ band width
- [ ] calculate band width (const for different dimensions?)
- [ ] band structure from eigvals? finite system != bloch theorem
- [ ] frequency smaller than bandwidth, larger then k mode spacing -> EL/Delta, w/Delta >~ 1 (-> regime plot)
- [ ] sign of t_hop negative?