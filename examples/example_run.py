"""
example_run.py
==============
Run the solid-liquid coexistence tracing workflow for Cu-Ag.

Edit the config below and run:
    python example_run.py

Output → coexistence_output/coexistence_line.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sgcmc_coexistence.workflow import trace_coexistence

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit here
# ─────────────────────────────────────────────────────────────────────────────
config = {
    # Solid phase (Cu_fcc)
    "phase_solid": {
        "datafile":        "/home/users/maniykxj/Structures/Cu_fcc_10x10x10.data",
        "element":         ["Cu", "Ag"],
        "pair_style":      "eam/alloy",
        "pair_coeff":      "* * /home/users/maniykxj/Potentials/CuAg_eam_2.alloy Cu Ag",
        "masses":          [63.546, 107.8682],
        "reference_phase": "solid",
        "pressure":        [0.0],
    },
    # Liquid phase (lqd_Cu)
    "phase_liquid": {
        "datafile":        "/home/users/maniykxj/Structures/Cu_lqd_10x10x10.data",
        "element":         ["Cu", "Ag"],
        "pair_style":      "eam/alloy",
        "pair_coeff":      "* * /home/users/maniykxj/Potentials/CuAg_eam_2.alloy Cu Ag",
        "masses":          [63.546, 107.8682],
        "reference_phase": "liquid",
        "pressure":        [0.0],
    },

    # Temperature
    "T_start":      950.0,   # K
    "T_end":       1300.0,   # K
    "dT":            20.0,   # K  ← change this to adjust step size

    # SGCMC scan (at T_start only)
    "chem_pot_start":   0.0,   # eV
    "chem_pot_stop":    1.3,   # eV
    "n_chem_pots":       50,   # number of delta_mu points
    "n_atoms":         4000,
    "neq":            10000,   # NPT equilibration steps
    "nsw":            25000,   # SGCMC production steps per delta_mu
    "nevery":           100,   # swap frequency
    "nattempts":        100,   # swap attempts per call
    "seed":            2311,
    "cores":             4,

    # Averaging & interpolation
    "n_last":         13000,   # rows used from average_k.dat ← user-configurable
    "n_grid":          1000,   # interpolation grid for phi   ← user-configurable

    # Coexistence prediction
    "prediction_method": "clausius-clapeyron", # "clausius-clapeyron" or "tau"

    # Calphy
    "calphy_exec":   "calphy",
    "calphy_n_equil":   10000,
    "calphy_n_switch":  25000,
    "calphy_n_iterations":  1,
    # calphy uses the same `cores` as SGCMC (they run sequentially, so cores are free)

    # Output directory (all simulation folders + CSV go here)
    "output_dir": "coexistence_output",
}

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = trace_coexistence(config)

    print("\n=== Coexistence line ===")
    print(f"{'T (K)':<10} {'δμ (eV)':<12} {'x_solid':<12} {'x_liquid':<12}")
    print("-" * 46)
    for r in results:
        print(f"{r['T']:<10.1f} {r['delta_mu_coex']:<12.4f} "
              f"{r['x_solid']:<12.4f} {r['x_liquid']:<12.4f}")
    print(f"\nSaved → {config['output_dir']}/coexistence_line.csv")
