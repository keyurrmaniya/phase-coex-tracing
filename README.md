# Phase Coexistence Tracing Methodology

This document details the thermodynamic framework and workflow used to determine the solid-liquid coexistence line in the Cu-Ag alloy system.

## 1. Thermodynamic Ensemble
The simulations are performed in the **Semi-Grand Canonical Ensemble** (constant $T, P, N, \Delta\mu$), where:
- $T$: Temperature
- $P$: Pressure (usually 0 bar)
- $N$: Total number of atoms
- $\delta\mu$: Chemical potential difference between species ($\mu_{Ag} - \mu_{Cu}$)

The characteristic state function for this ensemble is the **Semi-Grand Free Energy** ($\Phi$), defined per atom as:
$$\Phi = U - TS - \delta\mu x$$
where $U$ is the potential energy per atom, $S$ is the entropy per atom, and $x$ is the mole fraction of the solute species (Ag).

The fundamental differential relation for $\Phi$ is:
$$d\Phi = -S dT + v dP - x d\delta\mu$$
At constant pressure ($v dP = 0$):
$$d\Phi = -S dT - x d\delta\mu$$

## 2. Determining Coexistence at $T_{start}$
Finding the initial coexistence point requires calculating the full $\Phi(\delta\mu)$ curve for both phases at a fixed starting temperature.

### 2.1 Pure-Phase Anchoring
We calculate the free energy of the pure phases ($\Phi_0$) at a reference chemical potential $\delta\mu_0$ (usually corresponding to pure Cu) using the `calphy` library, which employs thermodynamic integration from a reference state (Einstein crystal for solid, Ideal Gas for liquid).

### 2.2 Thermodynamic Integration
To obtain $\Phi$ at any $\delta\mu$, we integrate the composition $x$ from the SGCMC simulations:
$$\Phi(T, \delta\mu) = \Phi(T, \delta\mu_0) - \int_{\delta\mu_0}^{\delta\mu} x(\delta\mu') d\delta\mu'$$
The integral is performed numerically using the trapezoidal rule on a fine grid (1000 points) interpolated from the SGCMC simulation data.

### 2.3 Coexistence Condition
Phase coexistence occurs when the semi-grand free energies of the solid ($s$) and liquid ($l$) phases are equal:
$$\Phi_s(T, \delta\mu_{coex}) = \Phi_l(T, \delta\mu_{coex})$$
We solve for $\delta\mu_{coex}$ by finding the intersection of the two $\Phi$ curves.

## 3. Tracing the Coexistence Line
Once the first point is found, we step in temperature using the **Clausius-Clapeyron equation** for the semi-grand canonical ensemble.

### 3.1 Entropy Calculation
The entropy per atom at the coexistence point is calculated using the relation:
$$S = \frac{U - \delta\mu_{coex} x - \Phi}{T}$$
where all quantities on the right are measured or calculated at the current temperature $T$.

### 3.2 Clausius-Clapeyron Prediction
Along the coexistence line, $d\Phi_s = d\Phi_l$. Substituting the differential relation:
$$-S_s dT - x_s d\delta\mu = -S_l dT - x_l d\delta\mu$$
Rearranging gives the slope of the coexistence line:
$$\frac{d\delta\mu_{coex}}{dT} = - \frac{S_s - S_l}{x_s - x_l}$$
For a small temperature step $\Delta T$, the new coexistence chemical potential is predicted as:
$$\delta\mu_{coex}(T + \Delta T) \approx \delta\mu_{coex}(T) + \left( \frac{d\delta\mu_{coex}}{dT} \right) \Delta T$$

### 3.3 Alternative $\tau$-based Prediction
An alternative prediction method uses the reciprocal temperature $\tau = T_{start} / T$. The change in coexistence chemical potential is predicted as:
$$d\delta\mu = \frac{(U_s - U_l) d\tau}{2 (x_s - x_l)}$$
where $d\tau = \tau_{new} - \tau_{old}$. The user can select the desired prediction method in the configuration.

## 4. Refinement Loop
1. **Prediction**: Estimate $\delta\mu_{new}$ for $T + \Delta T$ using the Clausius-Clapeyron slope.
2. **Measurement**: Perform a single-point SGCMC simulation at the new $(T, \delta\mu)$ to obtain updated $U$ and $x$.
3. **Update**: Recalculate pure-phase $\Phi_0$ at the new $T$ using `calphy`.
4. **Iterate**: Repeat the entropy calculation and slope prediction for the next step.

This methodology allows for efficient tracing of the entire phase envelope starting from a single scan.
