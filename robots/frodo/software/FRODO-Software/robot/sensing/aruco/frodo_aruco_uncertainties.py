# aruco_uncertainty.py
# -----------------------------------------------------------------------------
# Tunable, knee-shaped covariance model for ArUco pose measurements.
# - Full 3x3 covariance for [x, y, psi]
# - Scalars compatible with your current fields (pos_std, psi_std)
# - Includes 3D surface plotting at fixed psi
# - NEW: Off-axis "knee" shaping so errors stay low across most of the FOV
#        and rise sharply only near the edges.

# -----------------------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D proj)


# -----------------------------------------------------------------------------
# Tunable parameters
# -----------------------------------------------------------------------------

@dataclass
class ArUcoUncertaintyParams:
    # Geometry / limits
    FOV_deg: float = 90.0
    r_max: float = 2.0  # meters

    # --- Position error (meters) ---
    sigma_r_center: float = 0.01  # ~1 cm near center/close
    sigma_r_cap: float = 0.10     # ~10 cm cap at outer edges

    # --- Angle error (radians) ---
    sigma_psi_center: float = math.radians(1.0)  # ~1°
    sigma_psi_max: float    = math.radians(8.0)  # ~8° at edge & r_max

    # --- Distance growth exponents ---
    # How fast errors grow with range (normalized r_hat in [0,1])
    p_r: float = 1.3    # position vs distance
    p_psi: float = 1.0  # angle vs distance

    # --- NEW: Off-axis knee shaping (beta in [0,1] across half-FOV) ---
    # Fraction of half-FOV where ramping starts, e.g. 0.75 → "deep valley" across 75%
    beta_knee_frac: float = 0.75
    # Sharpness of the ramp beyond the knee; larger → steeper rise near edge
    beta_sharpness: float = 6.0
    # Magnitude of off-axis boost (multiplicative) once fully at the edge
    offaxis_mag_r: float = 1.0   # position extra factor (so center→1, edge→1+offaxis_mag_r)
    offaxis_mag_psi: float = 1.0 # angle extra factor

    # Anisotropy & correlation in x/y due to projective geometry
    tangential_gain_at_edge: float = 1.8  # tangential worse near edges
    rho_at_edge: float = 0.35             # x/y correlation magnitude near edge

    # Angle growth with |psi|
    s_yaw: float = 1.0
    psi_max_for_scale: float = math.radians(60.0)


# -----------------------------------------------------------------------------
# Helper: knee-shaped edge curve in [0,1] → [0,1]
# -----------------------------------------------------------------------------

def _edge_ramp(beta_hat: float, knee: float, sharpness: float) -> float:
    """
    A smooth "knee" function:
      - ~0 for beta_hat <= knee
      - rises smoothly to 1 as beta_hat -> 1
    Uses a smoothstep on the post-knee fraction, then raises to a power for sharpness.

    Args:
        beta_hat: normalized off-axis (0 at centerline, 1 at edge)
        knee: in [0,1); where the ramp begins
        sharpness: >= 1; larger = steeper near the edge
    """
    beta_hat = max(0.0, min(1.0, beta_hat))
    knee = max(0.0, min(0.99, knee))  # keep <1
    if beta_hat <= knee:
        return 0.0
    u = (beta_hat - knee) / (1.0 - knee)  # map knee..1 -> 0..1
    # smoothstep(u) = 3u^2 - 2u^3 (C1 continuous)
    s = (3.0 * u**2) - (2.0 * u**3)
    # sharpen it
    return s ** max(1.0, sharpness)


# -----------------------------------------------------------------------------
# Core model
# -----------------------------------------------------------------------------

def aruco_uncertainty_cov(
    tvec_xy: np.ndarray,
    psi: float,
    params: Optional[ArUcoUncertaintyParams] = None,
) -> Tuple[np.ndarray, float, float, Dict[str, float]]:
    """
    Compute covariance for [x, y, psi] and convenient scalar stds.

    Args:
        tvec_xy: np.array([x, y]) in meters (x forward, y left)
        psi: yaw angle (radians)
        params: ArUcoUncertaintyParams

    Returns:
        Sigma: (3,3) covariance matrix over [x, y, psi]
        pos_std_rms: sqrt(0.5*(Var[x]+Var[y]))  -- convenient position std (meters)
        psi_std: sqrt(Var[psi])  -- angle std (radians)
        meta: dict with auxiliaries (r, beta, sigma_radial, sigma_tangential, ...)
    """
    if params is None:
        params = ArUcoUncertaintyParams()

    x, y = float(tvec_xy[0]), float(tvec_xy[1])
    r = math.hypot(x, y) + 1e-12
    r_hat = min(max(r / params.r_max, 0.0), 1.0)

    # Off-axis measure beta_hat in [0,1] across the half-FOV
    half_fov = math.radians(params.FOV_deg / 2.0)
    beta = abs(math.atan2(y, max(x, 1e-9)))
    beta_hat = min(beta / half_fov, 1.0)

    # Knee-shaped ramp: near 0 for central region, rises hard near edge
    edge = _edge_ramp(beta_hat, params.beta_knee_frac, params.beta_sharpness)

    # --- Position noise: radial/tangential model ---
    base_r = params.sigma_r_center + (params.sigma_r_cap - params.sigma_r_center) * (r_hat ** params.p_r)
    # Off-axis multiplicative boost: 1 at center, (1 + offaxis_mag_r) at edge
    off_axis_boost_r = 1.0 + params.offaxis_mag_r * edge

    sigma_radial = min(base_r * off_axis_boost_r, params.sigma_r_cap)

    # Tangential is worse near edges; also modulated by the same "edge" ramp
    tangential_gain = 1.0 + (params.tangential_gain_at_edge - 1.0) * edge
    sigma_tangential = min(base_r * off_axis_boost_r * tangential_gain, params.sigma_r_cap)

    # Rotate (radial, tangential) -> (x,y)
    Sig_rt = np.diag([sigma_radial**2, sigma_tangential**2])
    theta = math.atan2(y, x)
    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta),  math.cos(theta)]])
    Sig_xy = R @ Sig_rt @ R.T

    # Add x/y correlation increasing toward edge
    rho = params.rho_at_edge * (edge ** 0.8)
    sx = math.sqrt(Sig_xy[0, 0])
    sy = math.sqrt(Sig_xy[1, 1])
    Sig_xy[0, 1] = Sig_xy[1, 0] = rho * sx * sy

    # --- Angle noise ---
    base_psi = params.sigma_psi_center + (params.sigma_psi_max - params.sigma_psi_center) * (r_hat ** params.p_psi)
    off_axis_boost_psi = 1.0 + params.offaxis_mag_psi * edge
    yaw_boost = 1.0 + (min(abs(psi), params.psi_max_for_scale) / params.psi_max_for_scale) ** params.s_yaw
    sigma_psi = base_psi * off_axis_boost_psi * yaw_boost

    # Assemble full covariance
    Sigma = np.zeros((3, 3), dtype=float)
    Sigma[0:2, 0:2] = Sig_xy
    Sigma[2, 2] = sigma_psi**2

    pos_std_rms = math.sqrt(0.5 * (Sig_xy[0, 0] + Sig_xy[1, 1]))

    meta = {
        "r": r,
        "r_hat": r_hat,
        "beta": beta,
        "beta_hat": beta_hat,
        "edge": edge,
        "sigma_radial": sigma_radial,
        "sigma_tangential": sigma_tangential,
        "sx": sx,
        "sy": sy,
        "rho_xy": rho,
    }
    return Sigma, pos_std_rms, sigma_psi, meta


def aruco_uncertainty_scalars(
    tvec_xy: np.ndarray,
    psi: float,
    params: Optional[ArUcoUncertaintyParams] = None,
) -> Tuple[float, float]:
    """
    Shim to match your _dummy_uncertainty signature:
    returns (uncertainty_position, uncertainty_psi).
    """
    _, pos_std, psi_std, _ = aruco_uncertainty_cov(tvec_xy, psi, params)
    return float(pos_std), float(psi_std)


# -----------------------------------------------------------------------------
# Grid sampling utilities
# -----------------------------------------------------------------------------

def fov_y_limit(x: float, FOV_deg: float) -> float:
    """Return max |y| allowed by the FOV wedge for a given x (>0)."""
    return x * math.tan(math.radians(FOV_deg / 2.0))


def sample_grid(
    params: Optional[ArUcoUncertaintyParams] = None,
    psi_fixed: float = 0.0,
    x_min: float = 0.1,
    x_max: Optional[float] = None,
    x_step: float = 0.05,
    y_step: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample (x,y) within the FOV wedge and compute position/angle std for fixed psi.

    Returns:
        X, Y: meshgrids (NaN outside FOV wedge)
        Z_pos: position std RMS (meters)
        Z_ang: angle std (radians)
    """
    params = params or ArUcoUncertaintyParams()
    if x_max is None:
        x_max = params.r_max

    x_vals = np.arange(x_min, x_max + 1e-9, x_step)
    y_max_global = fov_y_limit(x_max, params.FOV_deg)
    y_vals = np.arange(-y_max_global, y_max_global + 1e-9, y_step)

    X, Y = np.meshgrid(x_vals, y_vals)
    Z_pos = np.full_like(X, np.nan, dtype=float)
    Z_ang = np.full_like(X, np.nan, dtype=float)

    for j in range(Y.shape[0]):
        for i in range(X.shape[1]):
            x, y = float(X[j, i]), float(Y[j, i])
            if x <= 0.0:
                continue
            if abs(y) <= fov_y_limit(x, params.FOV_deg):
                _, pstd, astd, _ = aruco_uncertainty_cov(np.array([x, y]), psi_fixed, params)
                Z_pos[j, i] = pstd
                Z_ang[j, i] = astd

    return X, Y, Z_pos, Z_ang


# -----------------------------------------------------------------------------
# Plotting (3D surfaces for a fixed psi)
# -----------------------------------------------------------------------------

def plot_surface3d(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    title: str,
    zlabel: str,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot a 3D surface for Z(X,Y). NaNs are skipped to preserve the FOV wedge."""
    mask = ~np.isnan(Z)
    x = X[mask]; y = Y[mask]; z = Z[mask]

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

    import matplotlib.tri as mtri
    triang = mtri.Triangulation(x, y)
    ax.plot_trisurf(triang, z, linewidth=0.2, antialiased=True, alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("x (m) forward")
    ax.set_ylabel("y (m) left (+)")
    ax.set_zlabel(zlabel)
    return ax


def plot_std_surfaces_for_fixed_psi(
    psi_fixed_deg: float = 0.0,
    params: Optional[ArUcoUncertaintyParams] = None,
    x_min: float = 0.1,
    x_max: Optional[float] = None,
    x_step: float = 0.05,
    y_step: float = 0.05,
    show: bool = True,
) -> Tuple[Axes, Axes]:
    """
    Compute a grid and plot two 3D surfaces (position std & angle std)
    for a fixed psi (in degrees).
    """
    params = params or ArUcoUncertaintyParams()
    if x_max is None:
        x_max = params.r_max

    psi_fixed = math.radians(psi_fixed_deg)
    X, Y, Z_pos, Z_ang = sample_grid(
        params=params,
        psi_fixed=psi_fixed,
        x_min=x_min,
        x_max=x_max,
        x_step=x_step,
        y_step=y_step,
    )

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    plot_surface3d(X, Y, Z_pos, f"Position std (RMS) [m], psi={psi_fixed_deg:.1f}°", "pos std [m]", ax=ax1)
    plot_surface3d(X, Y, Z_ang, f"Angle std [rad], psi={psi_fixed_deg:.1f}°", "angle std [rad]", ax=ax2)

    fig.tight_layout()
    if show:
        plt.show()
    return ax1, ax2


# -----------------------------------------------------------------------------
# Optional: visualize the knee curve itself
# -----------------------------------------------------------------------------

def plot_offaxis_curve(params: Optional[ArUcoUncertaintyParams] = None, show: bool = True) -> Axes:
    """Plot the knee-shaped ramp vs normalized off-axis beta_hat."""
    params = params or ArUcoUncertaintyParams()
    bs = np.linspace(0, 1, 400)
    edge = np.array([_edge_ramp(b, params.beta_knee_frac, params.beta_sharpness) for b in bs])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bs, edge, label="edge ramp")
    ax.axvline(params.beta_knee_frac, ls="--", label="knee")
    ax.set_xlabel(r"normalized off-axis $\beta_{\hat{}}$  (0=center, 1=edge)")
    ax.set_ylabel("ramp (0..1)")
    ax.set_title("Off-axis knee-shaped ramp")
    ax.legend()
    if show:
        plt.show()
    return ax


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------

def _example():
    # Defaults: wide valley across ~75% of the half-FOV, sharp rise near edge
    params = ArUcoUncertaintyParams(
        FOV_deg=90.0,
        r_max=2.0,
        beta_knee_frac=0.75,
        beta_sharpness=6.0,
        offaxis_mag_r=1.0,
        offaxis_mag_psi=1.0,
    )

    # Quick sanity prints at a few points
    for (x, y, psi_deg) in [(0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (1.5, 0.3, 20.0), (2.0, 0.8, 40.0)]:
        _, pstd, astd, meta = aruco_uncertainty_cov(np.array([x, y]), math.radians(psi_deg), params)
        print(f"(x={x:.2f}, y={y:.2f}, psi={psi_deg:>4.1f}°)  "
              f"pos_std={pstd*100:5.1f} cm,  psi_std={math.degrees(astd):5.2f}°,  edge={meta['edge']:.2f}")

    # See the knee curve
    plot_offaxis_curve(params, show=True)

    # 3D surfaces at a fixed psi
    plot_std_surfaces_for_fixed_psi(psi_fixed_deg=20.0, params=params, x_step=0.06, y_step=0.06, show=True)


if __name__ == "__main__":
    _example()