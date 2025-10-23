# covariance_model.py
from __future__ import annotations

import dataclasses
from typing import Sequence, Tuple, List

import numpy as np


# ---------------------------
# Domain dataclasses (as given)
# ---------------------------

@dataclasses.dataclass
class Position:
    x: float
    y: float


@dataclasses.dataclass
class DataPoint:
    position_measured: Position
    position_true: Position
    psi_measured: float
    psi_true: float
    v: float          # forward speed
    psi_dot: float    # turning rate r


@dataclasses.dataclass
class MeasurementModel:
    # Placeholder for extensibility
    pass


# ---------------------------
# Covariance model definition
# ---------------------------

@dataclasses.dataclass
class CovarianceModel:
    """
    Diagonal covariance model with normalized linear features:

        Var_axis = k_axis * exp( w_axis · z ),
        z = (ϕ - μ) / σ, but with bias un-normalized.

    We store per-axis weights in the normalized space, per-feature normalization,
    and a per-axis calibration scalar k_axis.
    """
    w_x: np.ndarray
    w_y: np.ndarray
    w_psi: np.ndarray
    feature_names: List[str]
    mu: np.ndarray
    sigma: np.ndarray
    calib_scale: np.ndarray
    eps: float = 1e-12


# ---------------------------
# Feature engineering
# ---------------------------

def _features_from_state(
    x: float, y: float, psi: float, v: float, r: float
) -> Tuple[np.ndarray, List[str]]:
    dist = np.hypot(x, y)
    psi_wrapped = (psi + np.pi) % (2 * np.pi) - np.pi

    feats = np.array([
        1.0,                       # bias
        x, y,
        dist,
        abs(x), abs(y),
        psi_wrapped,
        abs(psi_wrapped),
        v, abs(v), v * v,
        r, abs(r), r * r,
        dist * abs(v),
        dist * abs(r),
        abs(psi_wrapped) * abs(v),
        abs(psi_wrapped) * abs(r),
    ], dtype=float)

    names = [
        "bias",
        "x", "y",
        "dist",
        "|x|", "|y|",
        "psi",
        "|psi|",
        "v", "|v|", "v^2",
        "r", "|r|", "r^2",
        "dist*|v|",
        "dist*|r|",
        "|psi|*|v|",
        "|psi|*|r|",
    ]
    return feats, names


def _stack_training(
    data: Sequence[DataPoint],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    Phi = []
    rx, ry, rpsi = [], [], []
    names_ref: List[str] | None = None

    for d in data:
        dx = d.position_measured.x - d.position_true.x
        dy = d.position_measured.y - d.position_true.y
        dpsi = d.psi_measured - d.psi_true
        dpsi = (dpsi + np.pi) % (2 * np.pi) - np.pi

        feats, names = _features_from_state(
            d.position_measured.x, d.position_measured.y,
            d.psi_measured, d.v, d.psi_dot
        )
        if names_ref is None:
            names_ref = names

        Phi.append(feats)
        rx.append(dx)
        ry.append(dy)
        rpsi.append(dpsi)

    assert names_ref is not None
    return np.vstack(Phi), np.asarray(rx), np.asarray(ry), np.asarray(rpsi), names_ref


# ---------------------------
# Normalization
# ---------------------------

def _normalize_features(Phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    z = (Phi - mu)/sigma, except the bias column stays un-normalized:
    mu[0]=0, sigma[0]=1.
    """
    mu = Phi.mean(axis=0)
    sigma = Phi.std(axis=0)
    sigma = np.maximum(sigma, 1e-8)

    # Keep bias raw
    mu[0] = 0.0
    sigma[0] = 1.0

    Z = (Phi - mu) / sigma
    return Z, mu, sigma


# ---------------------------
# Helpers: ridge warm-start and Adam
# ---------------------------

def _ridge_fit(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    F = X.shape[1]
    A = X.T @ X + l2 * np.eye(F)
    b = X.T @ y
    return np.linalg.solve(A, b)


class _Adam:
    def __init__(self, fdim: int, lr: float = 1e-2, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.m = np.zeros(fdim)
        self.v = np.zeros(fdim)
        self.t = 0

    def step(self, w: np.ndarray, g: np.ndarray):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * (g * g)
        mhat = self.m / (1 - self.b1 ** self.t)
        vhat = self.v / (1 - self.b2 ** self.t)
        w -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
        return w


# ---------------------------
# Training via Gaussian NLL (stable)
# ---------------------------

def _fit_axis_gaussian_nll(
    Z: np.ndarray,
    residuals: np.ndarray,
    l2: float = 1e-4,
    lr: float = 5e-3,
    steps: int = 3000,
    warmstart: bool = True,
) -> np.ndarray:
    """
    Minimize sum_i 0.5*(η_i + r_i^2 * exp(-η_i)) + 0.5*l2*||w||^2
    with Adam, gradient clipping, and stable exponentials.

    - Warm start: ridge on log(r^2 + eps) to initialize w.
    - Stability: clip η to [-CLIP, CLIP] before exp(-η).
    """
    n, f = Z.shape
    y2 = residuals * residuals
    eps = 1e-12
    CLIP = 30.0  # exp(±30) is within float range and numerically stable enough

    # Warm start
    if warmstart:
        w = _ridge_fit(Z, np.log(y2 + eps), l2=max(l2, 1e-4))
    else:
        w = np.zeros(f, dtype=float)

    opt = _Adam(f, lr=lr)

    for _ in range(steps):
        eta = Z @ w
        eta_clipped = np.clip(eta, -CLIP, CLIP)
        inv_exp_eta = np.exp(-eta_clipped)  # = exp(-η)

        # Gradient of data term: 0.5 * Z^T (1 - y2 * exp(-η))
        g_data = 0.5 * (Z.T @ (1.0 - y2 * inv_exp_eta))
        g_reg = l2 * w
        grad = g_data + g_reg

        # Gradient clipping (by norm)
        gnorm = np.linalg.norm(grad)
        if np.isfinite(gnorm) and gnorm > 5.0:
            grad = grad * (5.0 / gnorm)

        # Adam step
        w = opt.step(w, grad)

    return w


def _predict_var_axis(Z: np.ndarray, w: np.ndarray) -> np.ndarray:
    CLIP = 30.0
    eta = Z @ w
    return np.exp(np.clip(eta, -CLIP, CLIP))


def estimate_covariance_model(
    data: list[DataPoint],
    l2: float = 5e-4,
    lr: float = 7e-3,
    steps: int = 3000,
    calib_split: float = 0.2,
) -> CovarianceModel:
    """
    Fit weights by minimizing Gaussian NLL in normalized feature space
    (bias un-normalized). Then compute a per-axis calibration scale on
    a held-out split.
    """
    if len(data) == 0:
        raise ValueError("No data provided to estimate_covariance_model")

    Phi, rx, ry, rpsi, names = _stack_training(data)
    Z, mu, sigma = _normalize_features(Phi)

    # Shuffle + split
    n = Z.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(123)
    rng.shuffle(idx)
    split = int((1.0 - calib_split) * n)
    tr_idx, cal_idx = idx[:split], idx[split:]

    Ztr, Zcal = Z[tr_idx], Z[cal_idx]
    rx_tr, ry_tr, rpsi_tr = rx[tr_idx], ry[tr_idx], rpsi[tr_idx]
    rx_cal, ry_cal, rpsi_cal = rx[cal_idx], ry[cal_idx], rpsi[cal_idx]

    # Fit per-axis (stable)
    w_x = _fit_axis_gaussian_nll(Ztr, rx_tr, l2=l2, lr=lr, steps=steps, warmstart=True)
    w_y = _fit_axis_gaussian_nll(Ztr, ry_tr, l2=l2, lr=lr, steps=steps, warmstart=True)
    w_psi = _fit_axis_gaussian_nll(Ztr, rpsi_tr, l2=l2, lr=lr, steps=steps, warmstart=True)

    # Calibration on held-out split: k = mean(empirical / predicted)
    def calib_k(Zc, rc, w):
        pred = _predict_var_axis(Zc, w)
        emp = rc * rc + 1e-12
        k = float(np.mean(emp / pred))
        return float(np.clip(k, 1e-6, 1e6))

    kx = calib_k(Zcal, rx_cal, w_x)
    ky = calib_k(Zcal, ry_cal, w_y)
    kpsi = calib_k(Zcal, rpsi_cal, w_psi)

    return CovarianceModel(
        w_x=w_x, w_y=w_y, w_psi=w_psi,
        feature_names=list(names),
        mu=mu, sigma=sigma,
        calib_scale=np.array([kx, ky, kpsi], dtype=float),
        eps=1e-12,
    )


# ---------------------------
# Inference
# ---------------------------

def _featurize_and_normalize(
    position: Position, psi: float, v: float, r: float, mu: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    feats, _ = _features_from_state(position.x, position.y, psi, v, r)
    Z = (feats - mu) / sigma
    # keep bias raw: undo normalization for index 0
    Z[0] = feats[0]
    return Z


def get_covariance(
    position: Position,
    psi: float,
    v: float,
    r: float,
    covariance_model: CovarianceModel,
) -> np.ndarray:
    """
    Return 3x3 diagonal covariance diag([σ_x^2, σ_y^2, σ_psi^2]).
    """
    Z = _featurize_and_normalize(position, psi, v, r, covariance_model.mu, covariance_model.sigma)

    # Stable exponentials via _predict_var_axis
    var_x = float(_predict_var_axis(Z[None, :], covariance_model.w_x)[0] * covariance_model.calib_scale[0])
    var_y = float(_predict_var_axis(Z[None, :], covariance_model.w_y)[0] * covariance_model.calib_scale[1])
    var_psi = float(_predict_var_axis(Z[None, :], covariance_model.w_psi)[0] * covariance_model.calib_scale[2])

    tiny = 1e-15
    var_x = max(var_x, tiny)
    var_y = max(var_y, tiny)
    var_psi = max(var_psi, tiny)

    return np.diag([var_x, var_y, var_psi])


# ---------------------------
# Debug helpers
# ---------------------------

def describe_model(cm: CovarianceModel) -> str:
    lines = []
    for name, w in [("x", cm.w_x), ("y", cm.w_y), ("psi", cm.w_psi)]:
        lines.append(f"Axis {name}:")
        for fname, coeff in zip(cm.feature_names, w):
            lines.append(f"  {fname:>12s}: {coeff:+.5f}")
    lines.append(f"Calibration k: [x={cm.calib_scale[0]:.3f}, y={cm.calib_scale[1]:.3f}, psi={cm.calib_scale[2]:.3f}]")
    return "\n".join(lines)