# covariance_model.py
from __future__ import annotations

import dataclasses
from typing import Callable, Sequence

import numpy as np


# ---------------------------
# Domain dataclasses (given)
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
    v: float  # forward speed
    psi_dot: float  # turning rate r


# (Kept for extensibility; unused in this simple example.)
@dataclasses.dataclass
class MeasurementModel:
    pass


# ---------------------------
# Covariance model definition
# ---------------------------

@dataclasses.dataclass
class CovarianceModel:
    """
    Diagonal covariance model:

        Var(axis) = exp( w_axis · ϕ(x, y, ψ, v, r) )

    where ϕ is a feature vector and w_axis is a learned weight vector
    for each of x, y, and ψ axes, respectively.

    Attributes
    ----------
    w_x, w_y, w_psi : np.ndarray
        Weight vectors for x, y, psi variances.
    feature_names : list[str]
        Names for each feature (for debugging/introspection).
    eps : float
        Small constant used when forming log(residual^2 + eps).
    """
    w_x: np.ndarray
    w_y: np.ndarray
    w_psi: np.ndarray
    feature_names: list[str]
    eps: float = 1e-12


# ---------------------------
# Feature engineering
# ---------------------------

def _features_from_state(
        x: float, y: float, psi: float, v: float, r: float
) -> tuple[np.ndarray, list[str]]:
    """
    Construct a compact, well-behaved feature vector. Keep it simple but expressive.
    You can tweak this without changing the trainer or predictor.

    We avoid raw large magnitudes by using distances and magnitudes where sensible.
    """
    dist = np.hypot(x, y)
    # Clamp angle into [-pi, pi] like many systems do (safe even if already in range).
    psi_wrapped = (psi + np.pi) % (2 * np.pi) - np.pi

    feats = np.array([
        1.0,  # bias
        x, y,  # linear position
        dist,  # range
        abs(x), abs(y),  # L1-like terms
        psi_wrapped,  # signed angle
        abs(psi_wrapped),  # magnitude of angle
        v, abs(v), v * v,  # speed terms
        r, abs(r), r * r,  # turn rate terms
                   dist * abs(v),  # simple interaction
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
    assert feats.shape[0] == len(names)
    return feats, names


def _stack_features(
        data: Sequence[DataPoint],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Build design matrix X and log-residual^2 targets y_x, y_y, y_psi.

    Targets are:
        t_axis[i] = log( (residual_axis[i])^2 + eps )

    where residuals come from measured - true.
    """
    X_list: list[np.ndarray] = []
    rx2_log: list[float] = []
    ry2_log: list[float] = []
    rpsi2_log: list[float] = []
    eps = 1e-12

    feat_names_ref: list[str] | None = None

    for d in data:
        # residuals
        rx = d.position_measured.x - d.position_true.x
        ry = d.position_measured.y - d.position_true.y
        rpsi = d.psi_measured - d.psi_true
        # normalize angle residual to [-pi, pi] for stability
        rpsi = (rpsi + np.pi) % (2 * np.pi) - np.pi

        feats, names = _features_from_state(
            x=d.position_measured.x,  # you could also use true position; using measured keeps it online-usable
            y=d.position_measured.y,
            psi=d.psi_measured,
            v=d.v,
            r=d.psi_dot,
        )
        if feat_names_ref is None:
            feat_names_ref = names
        X_list.append(feats)
        rx2_log.append(np.log(rx * rx + eps))
        ry2_log.append(np.log(ry * ry + eps))
        rpsi2_log.append(np.log(rpsi * rpsi + eps))

    X = np.vstack(X_list)  # [N, F]
    yx = np.asarray(rx2_log)  # [N]
    yy = np.asarray(ry2_log)  # [N]
    ypsi = np.asarray(rpsi2_log)  # [N]
    assert feat_names_ref is not None
    return X, yx, yy, ypsi, feat_names_ref


# ---------------------------
# Ridge regression fit
# ---------------------------

def _ridge_fit(X: np.ndarray, y: np.ndarray, l2: float = 1e-4) -> np.ndarray:
    """
    Solve (X^T X + λI) w = X^T y
    """
    F = X.shape[1]
    A = X.T @ X + l2 * np.eye(F)
    b = X.T @ y
    w = np.linalg.solve(A, b)
    return w


# ---------------------------
# Public API
# ---------------------------

def estimate_covariance_model(
        data: list[DataPoint],
        l2: float = 1e-4,
) -> CovarianceModel:
    """
    Estimate weights w_x, w_y, w_psi by linear regression on log(residual^2 + eps).
    Returns a CovarianceModel, which predicts variances via exp(w · ϕ).
    """
    if len(data) == 0:
        raise ValueError("No data provided to estimate_covariance_model")

    X, yx, yy, ypsi, feat_names = _stack_features(data)

    w_x = _ridge_fit(X, yx, l2=l2)
    w_y = _ridge_fit(X, yy, l2=l2)
    w_psi = _ridge_fit(X, ypsi, l2=l2)

    return CovarianceModel(
        w_x=w_x,
        w_y=w_y,
        w_psi=w_psi,
        feature_names=list(feat_names),
        eps=1e-12,
    )


def get_covariance(
        position: Position,
        psi: float,
        v: float,
        r: float,
        covariance_model: CovarianceModel,
) -> np.ndarray:
    """
    Compute 3x3 diagonal covariance matrix diag([σ_x^2, σ_y^2, σ_psi^2]).

    Variances are predicted as:
        σ_axis^2 = exp( w_axis · ϕ(x, y, ψ, v, r) )
    """
    feats, _ = _features_from_state(position.x, position.y, psi, v, r)
    var_x = float(np.exp(covariance_model.w_x @ feats))
    var_y = float(np.exp(covariance_model.w_y @ feats))
    var_psi = float(np.exp(covariance_model.w_psi @ feats))

    # Safety: avoid exact zeros or NaNs
    tiny = 1e-15
    var_x = max(var_x, tiny)
    var_y = max(var_y, tiny)
    var_psi = max(var_psi, tiny)

    return np.diag([var_x, var_y, var_psi])


# ---------------------------
# Convenience: debugging helpers
# ---------------------------

def describe_model(cm: CovarianceModel) -> str:
    """
    Pretty-print weights with feature names.
    """
    lines = []
    for name, w in [("x", cm.w_x), ("y", cm.w_y), ("psi", cm.w_psi)]:
        lines.append(f"Axis {name}:")
        for fname, coeff in zip(cm.feature_names, w):
            lines.append(f"  {fname:>12s}: {coeff:+.5f}")
    return "\n".join(lines)
