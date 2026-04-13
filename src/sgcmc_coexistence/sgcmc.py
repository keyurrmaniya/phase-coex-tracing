"""
sgcmc_coexistence.sgcmc
=======================
Process raw SGCMC output to compute thermodynamic averages and the
semi-grand canonical free energy phi via thermodynamic integration.

Formula
-------
    phi(delta_mu) = phi_0 - ∫ x * d_delta_mu

The integral is computed on a fine interpolated grid (default 1000 pts)
so that the phi curve is smooth even when only ~50 simulation points exist.
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, PchipInterpolator

from sgcmc_coexistence.io import read_average_dat


def compute_sgcmc_averages(average_dat_dir, chem_pots, n_atoms, n_last=13000):
    """Compute mean PE and composition from SGCMC output files.

    Reads ``average_0.dat``, ``average_1.dat``, … and for each file
    averages the **last** ``n_last`` rows (to discard early transients).

    Parameters
    ----------
    average_dat_dir : str
        Directory containing ``average_k.dat`` files.
    chem_pots : array-like of float
        Chemical potential values (delta_mu) used in the SGCMC run,
        one per file index.
    n_atoms : int
        Total number of atoms in the simulation box.
    n_last : int
        Number of rows at the end of each file used for averaging.
        Default: 13000.

    Returns
    -------
    pd.DataFrame
        Columns:

        * ``delta_mu``  – chemical potential difference (eV)
        * ``pe_mean``   – mean total potential energy (eV)
        * ``x_mean``    – mean Ag mole fraction (species 2 / total)
    """
    chem_pots = np.asarray(chem_pots)
    records = []

    for idx, mu in enumerate(chem_pots):
        fpath = os.path.join(average_dat_dir, f"average_{idx}.dat")
        df = read_average_dat(fpath, n_last=n_last)

        pe_mean = df["pe"].mean()
        x_mean  = df["count2"].mean() / n_atoms

        records.append({
            "delta_mu": mu,
            "pe_mean":  pe_mean,
            "x_mean":   x_mean,
        })

    return pd.DataFrame(records)


def compute_semi_grand_fe(df_sgcmc, phi0, n_grid=1000, mu_ref=None):
    """Compute phi(delta_mu) on a fine interpolated grid.

    Equation
    --------
    phi = phi_0 - ∫ (2x - 1) d(delta_mu_scaled)
    where delta_mu_scaled = delta_mu / 2.

    Steps
    -----
    1. Fit a cubic spline to the raw (delta_mu, x) simulation points.
    2. Evaluate the spline on a uniform grid of ``n_grid`` points.
    3. Integrate cumulatively with the trapezoidal rule.

    Parameters
    ----------
    df_sgcmc : pd.DataFrame
        Output of :func:`compute_sgcmc_averages`, with columns
        ``delta_mu`` and ``x_mean``, sorted in ascending ``delta_mu``.
    phi0 : float
        Reference semi-grand free energy (eV/atom) at ``mu_ref``.
    n_grid : int
        Number of points in the interpolated grid.  Default: 1000.
    mu_ref : float or None
        The chemical potential value (Ag - Cu) where phi = phi0.
        If None, the first point in df_sgcmc is used.

    Returns
    -------
    mu_fine : np.ndarray, shape (n_grid,)
        Fine-grid delta_mu values (eV).
    x_fine : np.ndarray, shape (n_grid,)
        Interpolated Ag composition at each grid point.
    phi_fine : np.ndarray, shape (n_grid,)
        Semi-grand free energy phi (eV/atom) at each grid point.
    """
    df = df_sgcmc.sort_values("delta_mu").reset_index(drop=True)
    mu_raw = df["delta_mu"].values
    x_raw  = df["x_mean"].values

    if mu_ref is None:
        mu_ref = mu_raw[0]

    # ── Step 1: interpolate x onto fine grid ─────────────────────────
    # Ensure the grid includes mu_ref
    mu_min = min(mu_raw[0], mu_ref)
    mu_max = max(mu_raw[-1], mu_ref)
    mu_fine = np.linspace(mu_min, mu_max, n_grid)

    spline = PchipInterpolator(mu_raw, x_raw)
    x_fine = spline(mu_fine)
    x_fine = np.clip(x_fine, 0.0, 1.0)

    # ── Step 2: cumulative trapezoid integration ──────────────────────
    # We integrate (2x - 1) over d(mu/2) = 0.5 * dmu
    integrand = (2 * x_fine - 1) * 0.5
    
    # Cumulative integral from mu_ref
    # Find the index closest to mu_ref
    idx_ref = np.argmin(np.abs(mu_fine - mu_ref))
    
    phi_fine = np.empty(n_grid)
    phi_fine[idx_ref] = phi0
    
    # Integrate forward
    for i in range(idx_ref + 1, n_grid):
        # trapezoid area for the last segment
        area = 0.5 * (integrand[i-1] + integrand[i]) * (mu_fine[i] - mu_fine[i-1])
        phi_fine[i] = phi_fine[i-1] - area
        
    # Integrate backward
    for i in range(idx_ref - 1, -1, -1):
        area = 0.5 * (integrand[i] + integrand[i+1]) * (mu_fine[i+1] - mu_fine[i])
        phi_fine[i] = phi_fine[i+1] + area

    return mu_fine, x_fine, phi_fine
