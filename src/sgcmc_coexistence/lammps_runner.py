"""
sgcmc_coexistence.lammps_runner
================================
Launch SGCMC simulations via `pylammpsmpi`.

Two modes:
* **scan** – sweep over a range of delta_mu values (used at T_start)
* **single** – run at exactly one delta_mu (used at T+dT, T+2dT, …)

The LAMMPS protocol mirrors `run.py` exactly:
  1. NPT equilibration
  2. For each delta_mu:
     a. Production run with atom/swap (no output) → equilibrate at this mu
     b. Production run with atom/swap + fix print → collect statistics

Output for each delta_mu index k:
  * ``average_k.dat`` – columns: pe  count1  count2
  * ``traj_k.dat``    – LAMMPS dump file
"""

import os
import logging
import numpy as np
from pylammpsmpi import LammpsLibrary

log = logging.getLogger(__name__)



def _build_lmp(config, working_dir, cores):
    """Create a LammpsLibrary instance with the common setup commands."""
    lmp = LammpsLibrary(cores=cores, working_directory=working_dir)

    lmp.command(f"variable neq      equal {config['neq']}")
    lmp.command(f"variable nsw      equal {config['nsw']}")
    lmp.command(f"variable nevery   equal {config['nevery']}")
    lmp.command(f"variable nattempts equal {config['nattempts']}")

    lmp.command("units           metal")
    lmp.command("boundary        p p p")
    lmp.command("atom_style      atomic")
    lmp.command(f"timestep        {config.get('timestep', 0.001)}")
    lmp.command("box             tilt large")

    lmp.command(f"read_data       {config['datafile']}")
    lmp.command(f"pair_style      {config['pair_style']}")
    lmp.command(f"pair_coeff      {config['pair_coeff']}")

    for i, m in enumerate(config["masses"], start=1):
        lmp.command(f"mass            {i} {m}")

    return lmp


def _equilibrate(lmp, T, seed):
    """NPT equilibration at temperature T."""
    lmp.command(
        f"velocity all create {T} {seed} mom yes rot yes dist gaussian"
    )
    lmp.command(
        f"fix f1 all npt temp {T} {T} 0.1 iso 0.000000 0.000000 0.1"
    )
    lmp.command("thermo_style    custom step pe")
    lmp.command("thermo          1000")
    lmp.command("run             ${neq}")


def _define_composition_groups(lmp):
    """Define per-type atom groups and count variables."""
    lmp.command("variable atomone atom type==1")
    lmp.command("variable atomtwo atom type==2")
    lmp.command("group groupone dynamic all var atomone every ${nevery}")
    lmp.command("group grouptwo dynamic all var atomtwo every ${nevery}")
    lmp.command("variable countone equal count(groupone)")
    lmp.command("variable counttwo equal count(grouptwo)")


def _run_one_mu(lmp, T, chem_pot, count, seed):
    """Run equilibration + production for a single delta_mu.

    Produces ``average_{count}.dat`` and ``traj_{count}.dat``.
    """
    # ── Step 1: equilibrate at this mu (no output) ──────────────────
    lmp.command(
        f"semi-grand yes types 1 2 mu 0.0 {chem_pot:.4f} noforce yes"
    )
    lmp.command("run             ${nsw}")
    lmp.command("unfix           swap")

    # ── Step 2: production with statistics ──────────────────────────
    lmp.command(
        f"semi-grand yes types 1 2 mu 0.0 {chem_pot:.4f} noforce yes"
    )
    lmp.command(f"dump d1 all custom 5000 traj_{count}.dat id type mass x y z")
    lmp.command(
        f'fix f2 all print 1 "$(pe) ${{countone}} ${{counttwo}}" '
        f'screen no file average_{count}.dat'
    )
    lmp.command("run             ${nsw}")
    lmp.command("unfix           swap")
    lmp.command("unfix           f2")
    lmp.command(f"undump          d1")


def run_sgcmc_scan(config, T, chem_pots, working_dir, cores=48):
    """Run SGCMC for a full scan of delta_mu values.

    Parameters
    ----------
    config : dict
        Simulation parameters (see :mod:`sgcmc_coexistence.workflow` for keys).
    T : float
        Temperature in K.
    chem_pots : array-like of float
        Chemical potential differences (delta_mu) to scan.
    working_dir : str
        Directory for LAMMPS input/output.
    cores : int
        Number of MPI cores.

    Returns
    -------
    str
        Absolute path to ``working_dir``.
    """
    os.makedirs(working_dir, exist_ok=True)
    chem_pots = list(chem_pots)

    # Check if all output already exists
    missing = False
    for count in range(len(chem_pots)):
        if not os.path.isfile(os.path.join(working_dir, f"average_{count}.dat")):
            missing = True
            break
    if not missing:
        log.info("SGCMC scan results already exist in %s — skipping.", working_dir)
        return os.path.abspath(working_dir)

    seed = config.get("seed", 2311)

    lmp = _build_lmp(config, working_dir, cores)
    _equilibrate(lmp, T, seed)
    _define_composition_groups(lmp)

    for count, mu in enumerate(chem_pots):
        _run_one_mu(lmp, T, mu, count, seed)

    lmp.command("unfix           f1")
    lmp.close()
    return os.path.abspath(working_dir)


def run_sgcmc_single(config, T, delta_mu, working_dir, cores=48):
    """Run SGCMC for a single delta_mu value.

    Produces ``average_0.dat`` and ``traj_0.dat`` in ``working_dir``.
    Used at T+dT, T+2dT, … steps where only one delta_mu is needed.

    Parameters
    ----------
    config : dict
        Simulation parameters.
    T : float
        Temperature in K.
    delta_mu : float
        Chemical potential difference (eV).
    working_dir : str
        Directory for LAMMPS input/output.
    cores : int
        Number of MPI cores.

    Returns
    -------
    str
        Absolute path to ``working_dir``.
    """
    os.makedirs(working_dir, exist_ok=True)

    # Check if output already exists
    if os.path.isfile(os.path.join(working_dir, "average_0.dat")):
        log.info("SGCMC single result already exists in %s — skipping.", working_dir)
        return os.path.abspath(working_dir)

    seed = config.get("seed", 2311)

    lmp = _build_lmp(config, working_dir, cores)
    _equilibrate(lmp, T, seed)
    _define_composition_groups(lmp)
    _run_one_mu(lmp, T, delta_mu, count=0, seed=seed)
    lmp.command("unfix           f1")
    lmp.close()

    return os.path.abspath(working_dir)
