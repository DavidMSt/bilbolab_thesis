# import dataclasses
# from typing import TypeAlias
#
# import numpy as np
# from numpy.typing import NDArray
# from typing_extensions import Annotated
#
#
# @dataclasses.dataclass
# class Position:
#     x: float
#     y: float
#
#
# @dataclasses.dataclass
# class DataPoint:
#     position_measured: Position
#     position_true: Position
#     psi_measured: float
#     psi_true: float
#     v: float
#     psi_dot: float
#
#
# @dataclasses.dataclass
# class MeasurementModel:
#     ...
#
#
# @dataclasses.dataclass
# class CovarianceModel:
#     ...
#
#
# def estimate_covariance_model(data: list[DataPoint]):
#     ...
#
#
# def get_covariance(position: Position, psi: float, v: float, r: float, covariance_model: CovarianceModel) -> np.ndarray:
#     ...

import dataclasses
import math
from typing import List

import numpy as np
from numpy.typing import NDArray


# ============================================================
# === Data containers ========================================
# ============================================================

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
    v: float
    psi_dot: float


@dataclasses.dataclass
class MeasurementModel:
    # Placeholder (for future use if needed)
    ...


@dataclasses.dataclass
class CovarianceModel:
    feature_mean: NDArray[np.float64]
    feature_std: NDArray[np.float64]

    w_var_xx: NDArray[np.float64]
    w_var_yy: NDArray[np.float64]
    w_var_pp: NDArray[np.float64]

    w_cov_xy: NDArray[np.float64]
    w_cov_xp: NDArray[np.float64]
    w_cov_yp: NDArray[np.float64]

    reg_lambda: float = 1e-4
    jitter: float = 1e-9
    alpha: float = 1.0  # global covariance scale (calibrated later)


# ============================================================
# === Helpers ================================================
# ============================================================

def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _distance(x: float, y: float) -> float:
    return float(np.hypot(x, y))


def _features(position: Position, psi: float, v: float, r: float) -> NDArray[np.float64]:
    """Feature vector capturing geometry + dynamics."""
    x, y = position.x, position.y
    d = _distance(x, y)
    cpsi = math.cos(psi)
    spsi = math.sin(psi)
    v_abs = abs(v)
    r_abs = abs(r)

    phi = np.array([
        1.0,
        d,
        d ** 2,
        cpsi,
        spsi,
        x,
        y,
        v_abs,
        v_abs ** 2,
        r_abs,
        r_abs ** 2,
        v_abs * d,
        r_abs * d,
        v_abs * r_abs,
    ], dtype=np.float64)
    return phi


def _standardize(X: NDArray[np.float64]):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xn = (X - mu) / sigma
    return Xn, mu, sigma


def _ridge_fit(X: NDArray[np.float64], y: NDArray[np.float64], reg_lambda: float) -> NDArray[np.float64]:
    F = X.shape[1]
    A = X.T @ X + reg_lambda * np.eye(F)
    b = X.T @ y
    return np.linalg.solve(A, b)


def _nearest_psd(A: NDArray[np.float64], eps: float = 0.0) -> NDArray[np.float64]:
    A = 0.5 * (A + A.T)
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, eps)
    return (vecs * vals) @ vecs.T


# ============================================================
# === Main estimation ========================================
# ============================================================

def estimate_covariance_model(data: List[DataPoint],
                              reg_lambda: float = 1e-4,
                              jitter: float = 1e-9) -> CovarianceModel:
    if len(data) == 0:
        raise ValueError("No data provided.")

    rows, res = [], []
    for dp in data:
        dx = dp.position_measured.x - dp.position_true.x
        dy = dp.position_measured.y - dp.position_true.y
        dpsi = _wrap_angle(dp.psi_measured - dp.psi_true)
        phi = _features(dp.position_true, dp.psi_true, dp.v, dp.psi_dot)
        rows.append(phi)
        res.append([dx, dy, dpsi])

    X = np.vstack(rows)
    R = np.asarray(res)
    Xn, mu, sigma = _standardize(X)

    tiny = 1e-12
    y_var_xx = np.log(R[:, 0] ** 2 + tiny)
    y_var_yy = np.log(R[:, 1] ** 2 + tiny)
    y_var_pp = np.log(R[:, 2] ** 2 + tiny)

    y_cov_xy = R[:, 0] * R[:, 1]
    y_cov_xp = R[:, 0] * R[:, 2]
    y_cov_yp = R[:, 1] * R[:, 2]

    w_xx = _ridge_fit(Xn, y_var_xx, reg_lambda)
    w_yy = _ridge_fit(Xn, y_var_yy, reg_lambda)
    w_pp = _ridge_fit(Xn, y_var_pp, reg_lambda)

    w_xy = _ridge_fit(Xn, y_cov_xy, reg_lambda)
    w_xp = _ridge_fit(Xn, y_cov_xp, reg_lambda)
    w_yp = _ridge_fit(Xn, y_cov_yp, reg_lambda)

    return CovarianceModel(
        feature_mean=mu,
        feature_std=sigma,
        w_var_xx=w_xx,
        w_var_yy=w_yy,
        w_var_pp=w_pp,
        w_cov_xy=w_xy,
        w_cov_xp=w_xp,
        w_cov_yp=w_yp,
        reg_lambda=reg_lambda,
        jitter=jitter,
        alpha=1.0,
    )


def get_covariance(position: Position,
                   psi: float,
                   v: float,
                   r: float,
                   covariance_model: CovarianceModel) -> np.ndarray:
    cm = covariance_model
    phi = _features(position, psi, v, r)
    phi_n = (phi - cm.feature_mean) / cm.feature_std

    var_x = float(np.exp(phi_n @ cm.w_var_xx))
    var_y = float(np.exp(phi_n @ cm.w_var_yy))
    var_p = float(np.exp(phi_n @ cm.w_var_pp))

    cov_xy = float(phi_n @ cm.w_cov_xy)
    cov_xp = float(phi_n @ cm.w_cov_xp)
    cov_yp = float(phi_n @ cm.w_cov_yp)

    Sigma = np.array([
        [var_x, cov_xy, cov_xp],
        [cov_xy, var_y, cov_yp],
        [cov_xp, cov_yp, var_p],
    ], dtype=np.float64)

    Sigma[np.diag_indices_from(Sigma)] += cm.jitter
    Sigma *= cm.alpha  # global scaling
    return _nearest_psd(Sigma, eps=1e-12)


# ============================================================
# === Calibration helper =====================================
# ============================================================

def calibrate_alpha(data: List[DataPoint], model: CovarianceModel) -> float:
    D2s = []
    for dp in data:
        r = np.array([
            dp.position_measured.x - dp.position_true.x,
            dp.position_measured.y - dp.position_true.y,
            _wrap_angle(dp.psi_measured - dp.psi_true),
        ], dtype=float)

        # Temporarily force alpha = 1 for calibration
        alpha_old = model.alpha
        model.alpha = 1.0
        S = get_covariance(dp.position_true, dp.psi_true, dp.v, dp.psi_dot, model)
        model.alpha = alpha_old

        Sinv = np.linalg.pinv(S)
        D2s.append(float(r @ Sinv @ r))

    mean_D2 = float(np.mean(D2s)) if D2s else 3.0
    alpha = max(mean_D2 / 3.0, 1e-6)
    model.alpha = alpha
    return alpha
