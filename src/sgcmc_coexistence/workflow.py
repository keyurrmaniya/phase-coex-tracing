"""
sgcmc_coexistence.workflow
==========================
Main entry point for tracing the solid-liquid coexistence line.

Usage
-----
Edit the config dict at the top of ``example_run.py``, then run::

    python example_run.py

Physics recap
-------------
phi(delta_mu) = phi_0 - ∫ (2x - 1) d(delta_mu / 2)   (on a 1000-pt interpolated grid)
S             = (U - (x - 0.5) * delta_mu - phi) / T
d_delta_mu    = -(S_solid - S_liquid) / (x_solid - x_liquid) * dT   [Clausius-Clapeyron]
d(delta_mu/2) = ((U_s - U_l) * d_tau) / (2*(x_s - x_l))            [tau-based; doubled before use]

Workflow loop
-------------
Step 0  (T_start only):
    • SGCMC scan over all delta_mu → average_k.dat for both phases
    • calphy fe for pure Cu_fcc + pure lqd_Cu → phi0_solid, phi0_liquid
    • Interpolate x onto 1000-pt grid; compute phi(delta_mu) for both phases
    • Find crossing of phi_solid and phi_liquid → delta_mu_coex
    • Compute entropy at coexistence; apply prediction → delta_mu_new

Steps 1, 2, … (T+dT, T+2dT, …):
    • phi0 method (config option):
      - "propagate": local scan only; phi_pred = phi_old - S_old * dT
      - "calphy":    anchor (Δμ=0) + bridge + local scan; calphy for fresh phi0
    • SGCMC scan over the chosen grid
    • Integrate (2x-1) from anchor/prediction to find crossing delta_mu_coex
    • Prediction → next delta_mu_new
"""

import os
import csv
import logging
import numpy as np
from scipy.interpolate import interp1d

from sgcmc_coexistence.io import read_average_dat
from sgcmc_coexistence.sgcmc import compute_sgcmc_averages, compute_semi_grand_fe
from sgcmc_coexistence.coexistence import (
    find_coexistence,
    compute_entropy,
    clausius_clapeyron_step,
    tau_based_prediction,
)
from sgcmc_coexistence.calphy_runner import run_pure_phase_fe
from sgcmc_coexistence.lammps_runner import run_sgcmc_scan, run_sgcmc_single

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # ── Solid phase (Cu_fcc) ─────────────────────────────────────────────
    "phase_solid": {
        "datafile":        "Cu_fcc_10x10x10.data",
        "element":         ["Cu", "Ag"],
        "pair_style":      "eam/alloy",
        "pair_coeff":      "* * CuAg_eam_2.alloy Cu Ag",
        "masses":          [63.546, 107.8682],
        "reference_phase": "solid",
        "pressure":        [0.0],
    },
    # ── Liquid phase (lqd_Cu) ────────────────────────────────────────────
    "phase_liquid": {
        "datafile":        "lqd_Cu_10x10x10.data",
        "element":         ["Cu", "Ag"],
        "pair_style":      "eam/alloy",
        "pair_coeff":      "* * CuAg_eam_2.alloy Cu Ag",
        "masses":          [63.546, 107.8682],
        "reference_phase": "liquid",
        "pressure":        [0.0],
    },

    # ── Temperature stepping ─────────────────────────────────────────────
    "T_start":       950.0,
    "T_end":        1300.0,
    "dT":             10.0,    # K — user-configurable

    # ── SGCMC scan parameters (at T_start) ───────────────────────────────
    "chem_pot_start":    0.0,
    "chem_pot_stop":     1.3,
    "n_chem_pots":        50,
    "n_atoms":          4000,
    "neq":             10000,
    "nsw":             25000,
    "nevery":            100,
    "nattempts":         100,
    "seed":             2311,
    "cores":              48,
    "timestep":        0.001,

    # ── Averaging & interpolation ─────────────────────────────────────────
    "n_last":          13000,   # rows from average_k.dat — user-configurable
    "n_grid":           1000,   # interpolation grid for phi — user-configurable

    # ── Calphy ───────────────────────────────────────────────────────────
    "calphy_exec":          "calphy",
    "calphy_n_equil":       25000,
    "calphy_n_switch":      50000,
    "calphy_n_iterations":      1,
    # calphy runs after SGCMC finishes; it reuses the same `cores` (no separate setting needed)

    # ── Coexistence prediction & Refinement ─────────────────────────────
    "prediction_method":    "clausius-clapeyron",  # "clausius-clapeyron" or "tau"
    "phi0_method":          "propagate",           # "propagate" (S-based) or "calphy" (anchor+bridge)
    "n_local_points":       5,      # total points in local scan (centered at pred)
    "local_spacing":        0.02,   # spacing between points in eV
    "n_bridge_points":      0,      # intermediate pts between anchor & local (calphy mode)

    # ── Output ───────────────────────────────────────────────────────────
    "output_dir": "coexistence_output",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lammps_config(phase_cfg, global_cfg):
    return {
        "datafile":    phase_cfg["datafile"],
        "pair_style":  phase_cfg["pair_style"],
        "pair_coeff":  phase_cfg["pair_coeff"],
        "masses":      phase_cfg["masses"],
        "neq":         global_cfg["neq"],
        "nsw":         global_cfg["nsw"],
        "nevery":      global_cfg["nevery"],
        "nattempts":   global_cfg["nattempts"],
        "seed":        global_cfg["seed"],
        "timestep":    global_cfg.get("timestep", 0.001),
    }


def _run_calphy_both(config, T, out_dir):
    """Run calphy fe for both pure phases at temperature T."""
    solid_fe_dir  = os.path.join(out_dir, f"calphy_solid_T{T:.0f}")
    liquid_fe_dir = os.path.join(out_dir, f"calphy_liquid_T{T:.0f}")

    log.info("calphy fe — pure Cu_fcc  @ T=%.0f K", T)
    phi0_solid = run_pure_phase_fe(
        config["phase_solid"], T, solid_fe_dir,
        calphy_exec=config["calphy_exec"],
        n_equilibration_steps=config["calphy_n_equil"],
        n_switching_steps=config["calphy_n_switch"],
        n_iterations=config["calphy_n_iterations"],
        cores=config["cores"],   # same cores as SGCMC — they're free now
    )

    log.info("calphy fe — pure lqd_Cu @ T=%.0f K", T)
    phi0_liquid = run_pure_phase_fe(
        config["phase_liquid"], T, liquid_fe_dir,
        calphy_exec=config["calphy_exec"],
        n_equilibration_steps=config["calphy_n_equil"],
        n_switching_steps=config["calphy_n_switch"],
        n_iterations=config["calphy_n_iterations"],
        cores=config["cores"],   # same cores as SGCMC — they're free now
    )

    log.info("phi0_solid=%.6f eV/atom   phi0_liquid=%.6f eV/atom",
             phi0_solid, phi0_liquid)
    return phi0_solid, phi0_liquid


def _phi_from_scan(average_dir, chem_pots, phi0, config):
    """Average SGCMC scan data and compute phi on the fine grid."""
    df = compute_sgcmc_averages(
        average_dir, chem_pots,
        n_atoms=config["n_atoms"],
        n_last=config["n_last"],
    )
    mu_fine, x_fine, phi_fine = compute_semi_grand_fe(
        df, phi0, n_grid=config["n_grid"])
    return df, mu_fine, x_fine, phi_fine


def _U_at_mu(average_dir, chem_pots, mu_target, n_atoms, n_last):
    """Return mean PE/atom from the SGCMC file closest to mu_target."""
    chem_pots = np.asarray(chem_pots)
    idx = int(np.argmin(np.abs(chem_pots - mu_target)))
    fpath = os.path.join(average_dir, f"average_{idx}.dat")
    df = read_average_dat(fpath, n_last=n_last)
    return df["pe"].mean() / n_atoms


def _U_single(average_dir, n_atoms, n_last):
    """Return mean PE/atom from average_0.dat (single-mu run)."""
    fpath = os.path.join(average_dir, "average_0.dat")
    df = read_average_dat(fpath, n_last=n_last)
    return df["pe"].mean() / n_atoms


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def trace_coexistence(config=None):
    """Trace the Cu_fcc / lqd_Cu coexistence line from T_start to T_end.

    Parameters
    ----------
    config : dict or None
        Workflow configuration dict.  ``None`` uses :data:`DEFAULT_CONFIG`.

    Returns
    -------
    list of dict
        One record per temperature step with keys:
        ``T``, ``delta_mu_coex``, ``x_solid``, ``x_liquid``,
        ``phi_coex``, ``S_solid``, ``S_liquid``, ``d_delta_mu``.
    """
    if config is None:
        config = DEFAULT_CONFIG

    # ── Logging: console + file ──────────────────────────────────────────
    _fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root_log = logging.getLogger()
    root_log.setLevel(logging.INFO)

    # Console handler (always present)
    if not any(isinstance(h, logging.StreamHandler) and
               not isinstance(h, logging.FileHandler)
               for h in root_log.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(_fmt)
        root_log.addHandler(sh)

    # File handler — added after output_dir is known
    os.makedirs(config["output_dir"], exist_ok=True)
    log_path = os.path.join(config["output_dir"], "coexistence.log")
    fh = logging.FileHandler(log_path, mode="a")   # append so reruns don't overwrite
    fh.setFormatter(_fmt)
    root_log.addHandler(fh)

    log.info("=" * 60)
    log.info("Coexistence tracing started")
    log.info("Log file: %s", log_path)
    log.info("=" * 60)

    T_start  = config["T_start"]
    T_end    = config["T_end"]
    dT       = config["dT"]
    out_dir  = config["output_dir"]
    n_atoms  = config["n_atoms"]
    n_last   = config["n_last"]
    n_grid   = config["n_grid"]

    os.makedirs(out_dir, exist_ok=True)

    chem_pots = np.linspace(
        config["chem_pot_start"],
        config["chem_pot_stop"],
        config["n_chem_pots"],
    )

    solid_lmp_cfg  = _lammps_config(config["phase_solid"],  config)
    liquid_lmp_cfg = _lammps_config(config["phase_liquid"], config)

    # ── CSV output & Resume Check ───────────────────────────────────────
    csv_path = os.path.join(out_dir, "coexistence_line.csv")
    csv_fields = ["T", "delta_mu_coex", "x_solid", "x_liquid",
                  "phi_coex", "S_solid", "S_liquid", "d_delta_mu"]

    results = []
    T_current = T_start
    delta_mu_new = None
    step_idx = 0
    resume = False

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        log.info("Found existing CSV: %s — checking for resume", csv_path)
        try:
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    for r in rows:
                        results.append({k: float(v) for k, v in r.items()})

                    last = results[-1]
                    T_current = last["T"] + dT
                    delta_mu_new = last["delta_mu_coex"] + last["d_delta_mu"]
                    step_idx = len(results)
                    
                    # Check if we've already reached the end temperature
                    if dT > 0:
                        finished = (T_current > T_end + 1e-6)
                    else:
                        finished = (T_current < T_end - 1e-6)
                    
                    if not finished:
                        resume = True
                        log.info("Resuming from T=%.1f K (step %d), predicted δμ=%.4f eV",
                                 T_current, step_idx, delta_mu_new)
                    else:
                        log.info("CSV already contains full tracing (T_current=%.1f K reaches T_end=%.1f K).",
                                 T_current, T_end)
        except Exception as e:
            log.warning("Could not resume from CSV: %s. Starting fresh.", e)

    if resume:
        csv_fh = open(csv_path, "a", newline="")
        writer = csv.DictWriter(csv_fh, fieldnames=csv_fields)
    else:
        csv_fh = open(csv_path, "w", newline="")
        writer = csv.DictWriter(csv_fh, fieldnames=csv_fields)
        writer.writeheader()

    try:
        def loop_condition(T, Tend, dT):
            if dT > 0:
                return T <= Tend + 1e-6
            else:
                return T >= Tend - 1e-6

        while loop_condition(T_current, T_end, dT):
            log.info("=" * 60)
            log.info("T = %.1f K  (step %d)", T_current, step_idx)

            solid_dir  = os.path.join(out_dir, f"solid_T{T_current:.0f}_scan")
            liquid_dir = os.path.join(out_dir, f"liquid_T{T_current:.0f}_scan")

            phi0_method = config.get("phi0_method", "propagate")

            # ── SGCMC runs ───────────────────────────────────────────
            if step_idx == 0:
                log.info("SGCMC scan — solid  @ T=%.0f K (%d points)",
                         T_current, len(chem_pots))
                run_sgcmc_scan(solid_lmp_cfg,  T_current, chem_pots,
                               solid_dir,  cores=config["cores"])
                log.info("SGCMC scan — liquid @ T=%.0f K", T_current)
                run_sgcmc_scan(liquid_lmp_cfg, T_current, chem_pots,
                               liquid_dir, cores=config["cores"])

                # ── Reference free energies from Calphy ──────────────────
                phi0_s, phi0_l = _run_calphy_both(config, T_current, out_dir)
                log.info("phi0_solid=%.6f eV/atom   phi0_liquid=%.6f eV/atom",
                         phi0_s, phi0_l)

                # ── Integrate and find initial coexistence ───────────────
                _, mu_s_scan, x_s_scan, phi_s_scan = _phi_from_scan(
                    solid_dir, chem_pots, phi0_s, config)
                _, mu_l_scan, x_l_scan, phi_l_scan = _phi_from_scan(
                    liquid_dir, chem_pots, phi0_l, config)

                mu_coex, x_s_coex, x_l_coex, phi_coex = find_coexistence(
                    mu_s_scan, x_s_scan, phi_s_scan,
                    mu_l_scan, x_l_scan, phi_l_scan)

                U_solid  = _U_at_mu(solid_dir,  chem_pots, mu_coex, n_atoms, n_last)
                U_liquid = _U_at_mu(liquid_dir, chem_pots, mu_coex, n_atoms, n_last)
            else:
                # ── Build scan grid ───────────────────────────────────
                n_local = config.get("n_local_points", 5)
                spacing = config.get("local_spacing", 0.02)
                n_half = n_local // 2
                mu_local = delta_mu_new + np.arange(-n_half, n_half + 1) * spacing

                if phi0_method == "calphy":
                    # Anchor (chem_pot_start) + optional bridge + local
                    anchor = config["chem_pot_start"]
                    n_bridge = config.get("n_bridge_points", 0)
                    if n_bridge > 0:
                        bridge = np.linspace(anchor, mu_local[0],
                                             n_bridge + 2)[1:-1]
                    else:
                        bridge = np.array([])
                    mu_scan = np.sort(np.unique(
                        np.concatenate([[anchor], bridge, mu_local])))
                    mu_ref_val = None   # integrate from anchor (first pt)
                    log.info(
                        "SGCMC grid (calphy): anchor=%.4f + %d bridge + %d local "
                        "= %d total pts, range [%.4f, %.4f]",
                        anchor, n_bridge, len(mu_local), len(mu_scan),
                        mu_scan[0], mu_scan[-1])
                else:
                    mu_scan = mu_local
                    mu_ref_val = delta_mu_new
                    log.info(
                        "SGCMC grid (propagate): %d local pts, "
                        "range [%.4f, %.4f]",
                        len(mu_scan), mu_scan[0], mu_scan[-1])

                # ── Run SGCMC ─────────────────────────────────────────
                log.info("SGCMC scan — solid  @ T=%.0f K", T_current)
                run_sgcmc_scan(solid_lmp_cfg,  T_current, mu_scan,
                               solid_dir,  cores=config["cores"])
                log.info("SGCMC scan — liquid @ T=%.0f K", T_current)
                run_sgcmc_scan(liquid_lmp_cfg, T_current, mu_scan,
                               liquid_dir, cores=config["cores"])

                # ── Compute phi0 ──────────────────────────────────────
                if phi0_method == "calphy":
                    phi0_s, phi0_l = _run_calphy_both(config, T_current, out_dir)
                    log.info("phi0 (calphy): phi0_solid=%.6f  phi0_liquid=%.6f eV/atom",
                             phi0_s, phi0_l)
                else:
                    last = results[-1]
                    phi0_s = last["phi_coex"] - last["S_solid"] * dT
                    phi0_l = last["phi_coex"] - last["S_liquid"] * dT
                    log.info("phi0 (propagated): phi0_solid=%.6f  phi0_liquid=%.6f eV/atom",
                             phi0_s, phi0_l)

                # ── Integrate and find coexistence ────────────────────
                df_s_scan = compute_sgcmc_averages(solid_dir,  mu_scan, n_atoms, n_last)
                df_l_scan = compute_sgcmc_averages(liquid_dir, mu_scan, n_atoms, n_last)

                mu_s, x_s_fine, phi_s = compute_semi_grand_fe(
                    df_s_scan, phi0_s, n_grid=n_grid, mu_ref=mu_ref_val)
                mu_l, x_l_fine, phi_l = compute_semi_grand_fe(
                    df_l_scan, phi0_l, n_grid=n_grid, mu_ref=mu_ref_val)

                mu_coex, x_s_coex, x_l_coex, phi_coex = find_coexistence(
                    mu_s, x_s_fine, phi_s,
                    mu_l, x_l_fine, phi_l)

                U_solid  = _U_at_mu(solid_dir,  mu_scan, mu_coex, n_atoms, n_last)
                U_liquid = _U_at_mu(liquid_dir, mu_scan, mu_coex, n_atoms, n_last)

            log.info("Coexistence: δμ=%.4f eV  x_solid=%.4f  x_liquid=%.4f",
                     mu_coex, x_s_coex, x_l_coex)

            # ── Entropy ──────────────────────────────────────────────
            S_solid  = compute_entropy(
                U_solid,  mu_coex, x_s_coex, phi_coex, T_current)
            S_liquid = compute_entropy(
                U_liquid, mu_coex, x_l_coex, phi_coex, T_current)
            log.info("S_solid=%.6f  S_liquid=%.6f  eV/(atom·K)",
                     S_solid, S_liquid)

            # ── Prediction ────────────────────────────────────────────
            pred_method = config.get("prediction_method", "clausius-clapeyron")
            
            if pred_method == "clausius-clapeyron":
                try:
                    d_mu = clausius_clapeyron_step(
                        S_solid, x_s_coex, S_liquid, x_l_coex, dT)
                except ZeroDivisionError as exc:
                    log.error("Clausius-Clapeyron failed: %s — stopping.", exc)
                    break
            elif pred_method == "tau":
                T_start_val = config["T_start"]
                tau_current = T_start_val / T_current
                tau_new     = T_start_val / (T_current + dT)
                d_tau       = tau_new - tau_current
                try:
                    # Step in scaled space: Δμ̃ = τ * Δμ
                    d_mu_tilde = tau_based_prediction(
                        U_solid, x_s_coex, U_liquid, x_l_coex, d_tau)
                    mu_tilde_coex = tau_current * mu_coex
                    mu_tilde_new  = mu_tilde_coex + d_mu_tilde
                    delta_mu_new  = mu_tilde_new / tau_new   # convert back
                    d_mu = delta_mu_new - mu_coex
                    log.info(
                        "tau prediction: τ_cur=%.6f  τ_new=%.6f  δτ=%.6f",
                        tau_current, tau_new, d_tau)
                    log.info(
                        "  Δμ̃_coex=%.6f  δΔμ̃=%.6f  Δμ̃_new=%.6f  → Δμ_new=%.6f eV",
                        mu_tilde_coex, d_mu_tilde, mu_tilde_new, delta_mu_new)
                except ZeroDivisionError as exc:
                    log.error("Tau-based prediction failed: %s — stopping.", exc)
                    break
            else:
                log.error("Unknown prediction_method: %s", pred_method)
                break

            if pred_method != "tau":
                # CC already set d_mu; compute delta_mu_new
                delta_mu_new = mu_coex + d_mu
            log.info("d_δμ=%.4f eV  →  δμ_new=%.4f eV @ T=%.1f K",
                     d_mu, delta_mu_new, T_current + dT)

            # ── Record ────────────────────────────────────────────────
            row = {
                "T":             T_current,
                "delta_mu_coex": mu_coex,
                "x_solid":       x_s_coex,
                "x_liquid":      x_l_coex,
                "phi_coex":      phi_coex,
                "S_solid":       S_solid,
                "S_liquid":      S_liquid,
                "d_delta_mu":    d_mu,
            }
            results.append(row)
            writer.writerow(row)
            csv_fh.flush()

            T_current += dT
            step_idx  += 1

    finally:
        csv_fh.close()

    log.info("Done.  Results: %s", csv_path)
    return results
