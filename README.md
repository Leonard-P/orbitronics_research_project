# Orbital Hall Effect in Graphene — Numerical Results

Tight-binding simulations of the orbital Hall effect in a honeycomb lattice, for a small research project.

**[Project Report](report.pdf)**

The code to produce the data in the [results](results) folder was and can be run on Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Leonard-P/orbitronics-research-project/blob/main/run_simulations.ipynb)

## Repository structure

```
results/          Pre-computed simulation data, animations, and figures
  lattice/        Tight-binding module used for results
  run_simulations.ipynb
                  Reproduces all data in results/, runs in Google Colab
  report_calculations_and_plots.ipynb
                  Analysis and figure generation from the report
```

## Reproducing

The simulation notebook pins the commit of the code used to generate the results in the report. Click the Colab badge above or run locally:

```bash
pip install numpy scipy matplotlib tqdm
jupyter notebook run_simulations.ipynb
```

## Related

The simulation library used in this project was later rewritten and extended to
[`realspace-tb`](https://github.com/Leonard-P/realspace-tb)
([PyPI](https://pypi.org/project/realspace-tb/)).