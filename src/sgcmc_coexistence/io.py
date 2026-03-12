"""
sgcmc_coexistence.io
====================
Read SGCMC output files (average_k.dat) and calphy report.yaml.
"""

import os
import yaml
import numpy as np
import pandas as pd


def read_average_dat(filepath, n_last=None):
    """Read a LAMMPS fix-print output file produced by the SGCMC run.

    The file has one header comment line followed by rows of:
        pe   count1   count2

    Parameters
    ----------
    filepath : str
        Path to ``average_k.dat``.
    n_last : int or None
        If given, return only the last *n_last* rows (for averaging
        after equilibration).  ``None`` returns all rows.

    Returns
    -------
    pd.DataFrame
        Columns: ``pe`` (total potential energy, eV),
        ``count1`` (# atoms of type 1, Cu),
        ``count2`` (# atoms of type 2, Ag).
    """
    data = np.loadtxt(filepath, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    df = pd.DataFrame(data, columns=["pe", "count1", "count2"])
    if n_last is not None:
        df = df.iloc[-n_last:].reset_index(drop=True)
    return df


import time
import logging

log = logging.getLogger(__name__)

def collect_calphy_fe(report_yaml_path, timeout=3600, poll_interval=30):
    """Read the free energy from a calphy ``report.yaml``.

    Wait up to *timeout* seconds for the file to exist and be valid YAML.

    Parameters
    ----------
    report_yaml_path : str
        Path to the calphy ``report.yaml`` produced by a ``fe`` run.
    timeout : int
        Maximum wait time in seconds (default 1 hour).
    poll_interval : int
        Seconds between checks (default 30 seconds).

    Returns
    -------
    float
        Free energy in eV/atom.

    Raises
    ------
    FileNotFoundError
        If the file does not appear within the timeout.
    RuntimeError
        If the file exists but cannot be parsed as a calphy report.
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if os.path.isfile(report_yaml_path):
            try:
                # Check if file is not empty and can be loaded
                if os.path.getsize(report_yaml_path) > 0:
                    with open(report_yaml_path, "r") as fh:
                        report = yaml.safe_load(fh)
                    
                    if report and "results" in report and "free_energy" in report["results"]:
                        return float(report["results"]["free_energy"])
            except Exception:
                pass # Wait and retry if file is being written
        
        log.info("Waiting for calphy report at %s ...", report_yaml_path)
        time.sleep(poll_interval)

    raise FileNotFoundError(
        f"calphy report not found or invalid after {timeout}s: {report_yaml_path}"
    )
