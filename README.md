# SGCMC Coexistence Tracing

This package provides a workflow for tracing solid-liquid phase coexistence lines in alloy systems using Semi-Grand Canonical Monte Carlo (SGCMC) and free energy anchoring from `calphy`.

## Features
- **Thermodynamic Integration**: Calculates semi-grand free energy ($\Phi$) from SGCMC scans.
- **Coexistence Finding**: Locates phase crossing points on a fine interpolated grid.
- **Clausius-Clapeyron Tracing**: Automatically steps in temperature to map the phase envelope.
- **Direction-Agnostic**: Supports both forward (heating) and reverse (cooling) tracing.

## Methodology
The detailed thermodynamic derivations are available in [methodology.pdf](./methodology.pdf).

## Installation

You can install this package in your preferred conda environment:

```bash
# Clone the repository (once uploaded to GitHub)
git clone https://github.com/yourusername/sgcmc_coexistence.git
cd sgcmc_coexistence

# Install in editable mode
pip install -e .
```

Alternatively, install directly from GitHub:
```bash
pip install git+https://github.com/yourusername/sgcmc_coexistence.git
```

## Usage

1. Prepare your LAMMPS data files for solid and liquid phases.
2. Edit the configuration in `examples/example_run.py`.
3. Run the workflow:
   ```bash
   python examples/example_run.py
   ```

The results will be saved to a CSV file (e.g., `coexistence_line.csv`).
