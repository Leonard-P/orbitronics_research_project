# TODOs

## bugs/fixes/mistakes
- [x] Simulations with sine E-field  
- [x] units [solved for now] 
- [x] flux gradient  
- [ ] make sure to divide by Area
- [ ] type annotations float/int for site to position
- [ ] right hand coordinates: positive charge along positive z direction
- [ ] negative t_hop!!

## new module functionality
- [x] RK4 solver  
- [ ] windows for fft  
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