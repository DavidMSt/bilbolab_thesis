# simulate_and_test.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from aruco_covariance import (
    Position,
    DataPoint,
    CovarianceModel,
    estimate_covariance_model,
    get_covariance,
    _features_from_state,  # only for simulation (to create ground-truth)
    describe_model,
)

rng = np.random.default_rng(42)
random.seed(42)


def make_ground_truth_model() -> CovarianceModel:
    """
    Construct a synthetic 'true' covariance model with fixed weights.
    We'll keep it sparse-ish and reasonable.
    """
    feats_dummy, names = _features_from_state(0, 0, 0, 0, 0)
    F = feats_dummy.shape[0]

    w_x = np.zeros(F)
    w_y = np.zeros(F)
    w_psi = np.zeros(F)

    # Bias (base variance)
    w_x[names.index("bias")] = -4.0  # exp(-4) ≈ 0.018
    w_y[names.index("bias")] = -4.2  # exp(-4.2) ≈ 0.015
    w_psi[names.index("bias")] = -2.5  # exp(-2.5) ≈ 0.082

    # Distance & angle magnitude increase uncertainty
    for w in (w_x, w_y):
        w[names.index("dist")] = +0.15
        w[names.index("|psi|")] = +0.10
        w[names.index("dist*|v|")] = +0.08
        w[names.index("dist*|r|")] = +0.05

    # Speed/turn effects
    for w in (w_x, w_y):
        w[names.index("v^2")] = +0.06
        w[names.index("|r|")] = +0.12

    # Heading variance especially sensitive to turn rate & angle
    w_psi[names.index("|psi|")] = +0.20
    w_psi[names.index("r^2")] = +0.20
    w_psi[names.index("|r|")] = +0.10
    w_psi[names.index("v^2")] = +0.05

    return CovarianceModel(
        w_x=w_x, w_y=w_y, w_psi=w_psi, feature_names=names, eps=1e-12
    )


def sample_state(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample (x,y,psi,v,r) states. These are the *measured* states used to predict covariance.
    """
    # Positions within a 5m square, angle within [-pi, pi], speeds within reasonable robot bounds
    x = rng.uniform(-2.5, 2.5, size=n)
    y = rng.uniform(-2.5, 2.5, size=n)
    psi = rng.uniform(-math.pi, math.pi, size=n)
    v = rng.uniform(-1.0, 1.5, size=n)  # m/s
    r = rng.uniform(-1.0, 1.0, size=n)  # rad/s
    return x, y, psi, v, r


def generate_dataset(n: int, gt: CovarianceModel) -> list[DataPoint]:
    """
    Generate synthetic DataPoints by:
      1) sampling a *true* state (x*, y*, psi*)
      2) computing true covariance from the *measured* conditions (we can choose either;
         here we add noise based on the measured to match online usage)
      3) drawing noise ~ N(0, diag(σ^2)) and forming measured = true + noise
    """
    data: list[DataPoint] = []

    # true positions: draw around origin so residuals are controlled
    xt, yt, psit, v, r = sample_state(n)
    # use the *true* states as a base, then compute measured by adding noise per model

    for i in range(n):
        pos_true = Position(float(xt[i]), float(yt[i]))
        psi_true = float(psit[i])

        # For realism, use the *same* state to compute covariance (could also corrupt it first)
        cov = get_covariance(pos_true, psi_true, float(v[i]), float(r[i]), gt)
        stds = np.sqrt(np.diag(cov))

        # Draw measurement noise
        noise_x = float(rng.normal(0.0, stds[0]))
        noise_y = float(rng.normal(0.0, stds[1]))
        noise_psi = float(rng.normal(0.0, stds[2]))

        pos_meas = Position(pos_true.x + noise_x, pos_true.y + noise_y)
        psi_meas = psi_true + noise_psi

        data.append(
            DataPoint(
                position_measured=pos_meas,
                position_true=pos_true,
                psi_measured=psi_meas,
                psi_true=psi_true,
                v=float(v[i]),
                psi_dot=float(r[i]),
            )
        )
    return data


def evaluate_model(cm: CovarianceModel, data: list[DataPoint]) -> dict:
    """
    Compare predicted variances vs empirical residual^2 on a set.
    Report RMSE on log-variances, and also calibration ratios.
    """
    preds = []
    targets = []

    eps = 1e-12

    for d in data:
        cov = get_covariance(d.position_measured, d.psi_measured, d.v, d.psi_dot, cm)
        var_pred = np.diag(cov)  # [var_x, var_y, var_psi]

        # empirical residuals
        rx = d.position_measured.x - d.position_true.x
        ry = d.position_measured.y - d.position_true.y
        rpsi = d.psi_measured - d.psi_true
        rpsi = (rpsi + np.pi) % (2 * np.pi) - np.pi

        var_emp = np.array([rx * rx, ry * ry, rpsi * rpsi]) + eps

        preds.append(var_pred)
        targets.append(var_emp)

    P = np.vstack(preds)  # [N, 3]
    T = np.vstack(targets)  # [N, 3]

    # RMSE in log-space
    rmse_log = np.sqrt(np.mean((np.log(P) - np.log(T)) ** 2, axis=0))

    # Calibration ratio: mean(empirical / predicted) ~ 1 if calibrated
    calib = np.mean(T / P, axis=0)

    return {
        "rmse_log_x": float(rmse_log[0]),
        "rmse_log_y": float(rmse_log[1]),
        "rmse_log_psi": float(rmse_log[2]),
        "calib_x": float(calib[0]),
        "calib_y": float(calib[1]),
        "calib_psi": float(calib[2]),
    }


def main():
    # 1) Make a ground-truth covariance model and synthesize data
    gt = make_ground_truth_model()
    print("=== Ground-truth weights (for reference) ===")
    print(describe_model(gt))
    print()

    n_train = 20000
    n_test = 5000

    print(f"Generating {n_train} training and {n_test} test samples...")
    train = generate_dataset(n_train, gt)
    test = generate_dataset(n_test, gt)

    # 2) Fit model on the synthetic training data
    print("Fitting covariance model...")
    learned = estimate_covariance_model(train, l2=1e-4)

    print("\n=== Learned weights ===")
    print(describe_model(learned))
    print()

    # 3) Evaluate on test set
    metrics = evaluate_model(learned, test)
    print("=== Test metrics ===")
    for k, v in metrics.items():
        print(f"{k:>14s}: {v:.4f}")

    # 4) Show a few example predictions
    print("\nExamples (first 5 test points):")
    for i in range(5):
        d = test[i]
        cov_pred = get_covariance(d.position_measured, d.psi_measured, d.v, d.psi_dot, learned)
        rx = d.position_measured.x - d.position_true.x
        ry = d.position_measured.y - d.position_true.y
        rpsi = d.psi_measured - d.psi_true
        rpsi = (rpsi + np.pi) % (2 * np.pi) - np.pi
        emp = np.array([rx * rx, ry * ry, rpsi * rpsi])
        print(f"  Sample {i}:")
        print(f"    predicted var: {np.diag(cov_pred)}")
        print(f"    empirical var: {emp}")

    # 5) Compare to ground-truth model on the same test set (optional sanity check)
    gt_metrics = evaluate_model(gt, test)
    print("\n=== Ground-truth model (oracle) metrics on test set ===")
    for k, v in gt_metrics.items():
        print(f"{k:>14s}: {v:.4f}")


if __name__ == "__main__":
    main()
