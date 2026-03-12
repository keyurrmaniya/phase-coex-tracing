"""
sgcmc_coexistence
=================
Automated workflow to trace the solid-liquid coexistence line using
Semi-Grand Canonical Monte Carlo (SGCMC) simulations combined with
calphy free-energy calculations.

Modules
-------
io              - Read SGCMC output files and calphy report.yaml
sgcmc           - Average SGCMC data, compute semi-grand free energy φ
coexistence     - Find coexistence, compute entropy, Clausius-Clapeyron step
calphy_runner   - Run calphy `fe` mode for pure phases
lammps_runner   - Launch pylammpsmpi SGCMC runs
workflow        - Main entry point: trace_coexistence(config)
"""
