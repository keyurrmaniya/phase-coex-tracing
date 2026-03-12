"""
sgcmc_coexistence.calphy_runner
================================
Run calphy `fe` (Frenkel-Ladd / Einstein crystal) free-energy
calculations for pure phases and collect the result.

Calphy output structure
-----------------------
When calphy runs in a working directory it creates a subdirectory::

    {working_dir}/
    └── fe-{datafile_basename}-{reference_phase}-{T:.0f}-0/
        └── report.yaml   ← free energy lives here

We construct this path explicitly so we never read the wrong file.
"""

import os

try:
    from amstools.calphy import run_calphy
    _AMSTOOLS_AVAILABLE = True
except ImportError:
    _AMSTOOLS_AVAILABLE = False

from sgcmc_coexistence.io import collect_calphy_fe


def _calphy_subdir_name(datafile, reference_phase, T, mode="fe", iteration=0):
    """Return the subdirectory name calphy creates inside working_dir.

    Pattern: ``{mode}-{datafile_basename}-{reference_phase}-{T:.0f}-{iteration}``

    Examples
    --------
    >>> _calphy_subdir_name("cu_fcc_10x10x10.data", "solid", 950)
    'fe-cu_fcc_10x10x10.data-solid-950-0'
    """
    basename = os.path.basename(datafile).lower()
    return f"{mode}-{basename}-{reference_phase}-{int(T)}-{iteration}"


def get_calphy_report_path(working_dir, datafile, reference_phase, T,
                           mode="fe", iteration=0):
    """Return the full path to the calphy report.yaml.

    Parameters
    ----------
    working_dir : str
        The directory passed to calphy as its working directory.
    datafile : str
        Path or basename of the LAMMPS data file used in the calculation.
    reference_phase : str
        ``"solid"`` or ``"liquid"``.
    T : float
        Temperature in K.
    mode : str
        Calphy mode string (default ``"fe"``).
    iteration : int
        Calphy iteration index (starts at 0).

    Returns
    -------
    str
        Absolute path to ``report.yaml``.
    """
    subdir = _calphy_subdir_name(datafile, reference_phase, T, mode, iteration)
    report = os.path.join(working_dir, subdir, "report.yaml")
    return report


def run_pure_phase_fe(phase_config, T, working_dir,
                      calphy_exec="calphy",
                      n_equilibration_steps=10000,
                      n_switching_steps=25000,
                      n_iterations=1,
                      cores=4,
                      queue_commands=None,
                      rerun=False):
    """Run calphy ``fe`` mode for a pure phase and return its free energy.

    Parameters
    ----------
    phase_config : dict
        Required keys:

        * ``element``         – str or list[str], e.g. ``"Cu"``
        * ``pair_style``      – LAMMPS pair_style string
        * ``pair_coeff``      – LAMMPS pair_coeff string
        * ``datafile``        – LAMMPS data file (also used as ``lattice``)
        * ``reference_phase`` – ``"solid"`` or ``"liquid"``
        * ``pressure``        – list[float], e.g. ``[0.0]``
        * ``mass``            – float or list[float] (optional; auto if omitted)

    T : float
        Temperature in K.
    working_dir : str
        Directory where calphy input/output will be placed.
    calphy_exec : str
        calphy executable.
    n_equilibration_steps, n_switching_steps, n_iterations : int
        calphy integration parameters.
    cores : int
        Number of MPI cores for calphy.
    queue_commands : list[str] or None
        Extra shell lines for the scheduler script (e.g. conda activate).
    rerun : bool
        Force re-run even if output already exists.

    Returns
    -------
    float
        Free energy of the pure phase in eV/atom.
    """
    if not _AMSTOOLS_AVAILABLE:
        raise ImportError(
            "amstools is required to run calphy. "
            "Install it from the ams-tools repository."
        )

    working_dir = os.path.abspath(working_dir)
    datafile     = phase_config["datafile"]
    ref_phase    = phase_config.get("reference_phase", "solid")

    # Step 1: Check if the report already exists (skip if rerun=False)
    report_path = get_calphy_report_path(working_dir, datafile, ref_phase, T)
    if os.path.isfile(report_path) and not rerun:
        print(f"calphy report already exists: {report_path} — skipping.")
        return collect_calphy_fe(report_path)

    # Step 2: Check if input.yaml exists (indicates simulation might be in progress)
    input_yaml = os.path.join(working_dir, "input.yaml")
    if os.path.isfile(input_yaml) and not rerun:
        print(f"calphy input.yaml exists in {working_dir} but report is missing. "
              "Assuming simulation is in progress — waiting for report.")
    else:
        # Step 3: Run calphy (or rerun if requested)
        run_calphy(
            element=phase_config["element"],
            pair_style=phase_config["pair_style"],
            pair_coeff=phase_config["pair_coeff"],
            working_dir=working_dir,
            mass=phase_config.get("mass"),
            mode="fe",
            lattice=datafile,           # pass the data file as lattice
            reference_phase=ref_phase,
            temperature=[T, T],
            pressure=phase_config.get("pressure", [0.0]),
            n_equilibration_steps=n_equilibration_steps,
            n_switching_steps=n_switching_steps,
            n_iterations=n_iterations,
            scheduler="local",
            cores=cores,
            queue_commands=queue_commands,
            calphy_exec=calphy_exec,
            rerun=rerun,
        )

    # Step 4: Wait for report (this will poll if file is missing or being written)
    return collect_calphy_fe(report_path)
