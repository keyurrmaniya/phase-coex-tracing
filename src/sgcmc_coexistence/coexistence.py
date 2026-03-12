"""
sgcmc_coexistence.coexistence
=============================
Find the coexistence point from fine-grid phi curves, compute entropy,
and apply the Clausius-Clapeyron equation for the next temperature step.
"""

import warnings
import numpy as np
from scipy.interpolate import interp1d


def find_coexistence(mu_solid, x_solid, phi_solid,
                     mu_liquid, x_liquid, phi_liquid):
    """Find the coexistence delta_mu where phi_solid == phi_liquid.

    Both phi curves are already on a fine interpolated grid
    (output of :func:`~sgcmc_coexistence.sgcmc.compute_semi_grand_fe`).
    We re-interpolate them onto a common grid that spans the overlapping
    delta_mu range and locate the sign change of (phi_solid - phi_liquid).

    Parameters
    ----------
    mu_solid, mu_liquid : np.ndarray
        Fine-grid delta_mu values for solid and liquid phases.
    x_solid, x_liquid : np.ndarray
        Interpolated compositions on those grids.
    phi_solid, phi_liquid : np.ndarray
        Semi-grand free energy on those grids (eV/atom).

    Returns
    -------
    delta_mu_coex : float
        Coexistence chemical potential difference (eV).
    x_solid_coex : float
        Solid Ag fraction at coexistence.
    x_liquid_coex : float
        Liquid Ag fraction at coexistence.
    phi_coex : float
        Common semi-grand free energy at coexistence (eV/atom).

    Raises
    ------
    RuntimeError
        If no crossing is found in the overlapping delta_mu range.
    """
    # Overlapping range
    mu_lo = max(mu_solid[0],  mu_liquid[0])
    mu_hi = min(mu_solid[-1], mu_liquid[-1])
    if mu_lo >= mu_hi:
        raise RuntimeError(
            "No overlapping delta_mu range between solid and liquid data.\n"
            f"  solid:  [{mu_solid[0]:.4f}, {mu_solid[-1]:.4f}]\n"
            f"  liquid: [{mu_liquid[0]:.4f}, {mu_liquid[-1]:.4f}]"
        )

    # Common fine grid over the overlap
    n_common = max(len(mu_solid), len(mu_liquid))
    mu_common = np.linspace(mu_lo, mu_hi, n_common)

    phi_s = interp1d(mu_solid,  phi_solid,  kind="cubic",
                     bounds_error=False, fill_value="extrapolate")(mu_common)
    phi_l = interp1d(mu_liquid, phi_liquid, kind="cubic",
                     bounds_error=False, fill_value="extrapolate")(mu_common)

    diff = phi_s - phi_l
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_changes) == 0:
        raise RuntimeError(
            "No coexistence crossing found in delta_mu range "
            f"[{mu_lo:.4f}, {mu_hi:.4f}] eV.\n"
            "Check that phi_solid and phi_liquid actually cross."
        )
    if len(sign_changes) > 1:
        warnings.warn(
            f"Multiple crossings found ({len(sign_changes)}); using the first."
        )

    idx = sign_changes[0]
    # Linear interpolation between the two bracketing points
    mu_a, mu_b = mu_common[idx], mu_common[idx + 1]
    d_a,  d_b  = diff[idx],      diff[idx + 1]
    mu_coex = mu_a - d_a * (mu_b - mu_a) / (d_b - d_a)

    # Interpolate x and phi at the coexistence mu
    x_s_at_coex  = float(interp1d(mu_solid,  x_solid,
                                   kind="cubic", bounds_error=False,
                                   fill_value="extrapolate")(mu_coex))
    x_l_at_coex  = float(interp1d(mu_liquid, x_liquid,
                                   kind="cubic", bounds_error=False,
                                   fill_value="extrapolate")(mu_coex))
    phi_coex     = float(interp1d(mu_solid,  phi_solid,
                                   kind="cubic", bounds_error=False,
                                   fill_value="extrapolate")(mu_coex))

    return mu_coex, x_s_at_coex, x_l_at_coex, phi_coex


def compute_entropy(U_per_atom, delta_mu, x, phi, T):
    """Compute entropy per atom.

    Uses::

        S = (U - delta_mu * x - phi) / T

    Parameters
    ----------
    U_per_atom : float
        Mean potential energy per atom (eV/atom).
    delta_mu : float
        Chemical potential difference at coexistence (eV).
    x : float
        Mean Ag mole fraction at coexistence.
    phi : float
        Semi-grand free energy at coexistence (eV/atom).
    T : float
        Temperature (K).

    Returns
    -------
    float
        Entropy per atom in eV/(atom·K).
    """
    return (U_per_atom - delta_mu * x - phi) / T


def clausius_clapeyron_step(S_solid, x_solid, S_liquid, x_liquid, dT):
    """Compute d_delta_mu to step from T to T+dT along the coexistence line.

    Clausius-Clapeyron for the semi-grand canonical ensemble::

        d_delta_mu = -(S_solid - S_liquid) / (x_solid - x_liquid) * dT

    Parameters
    ----------
    S_solid : float
        Entropy per atom of solid at coexistence (eV/(atom·K)).
    x_solid : float
        Ag mole fraction of solid at coexistence.
    S_liquid : float
        Entropy per atom of liquid at coexistence (eV/(atom·K)).
    x_liquid : float
        Ag mole fraction of liquid at coexistence.
    dT : float
        Temperature step (K).

    Returns
    -------
    float
        Change in delta_mu (eV).
    """
    delta_x = x_solid - x_liquid
    if abs(delta_x) < 1e-12:
        raise ZeroDivisionError(
            "x_solid ≈ x_liquid; Clausius-Clapeyron is ill-defined."
        )
    return -(S_solid - S_liquid) / delta_x * dT
