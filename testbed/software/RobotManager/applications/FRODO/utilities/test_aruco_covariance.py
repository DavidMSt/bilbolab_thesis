# simulate_and_test.py
from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt

from aruco_covariance import (
    Position,
    DataPoint,
    CovarianceModel,
    estimate_covariance_model,
    get_covariance,
    _features_from_state,   # only to define the ground-truth weights
    describe_model,
)

rng = np.random.default_rng(7)


# ---------------------------
# Ground-truth model (same structure)
# ---------------------------

def make_ground_truth_model() -> CovarianceModel:
    feats_dummy, names = _features_from_state(0, 0, 0, 0, 0)
    F = feats_dummy.shape[0]

    w_x = np.zeros(F)
    w_y = np.zeros(F)
    w_psi = np.zeros(F)

    # Base levels
    w_x[names.index("bias")] = -4.0
    w_y[names.index("bias")] = -4.2
    w_psi[names.index("bias")] = -2.5

    # Distance & angle magnitude -> more noise (pos)
    for w in (w_x, w_y):
        w[names.index("dist")] = +0.15
        w[names.index("|psi|")] = +0.10
        w[names.index("v^2")] = +0.06
        w[names.index("|r|")] = +0.12
        w[names.index("dist*|v|")] = +0.08
        w[names.index("dist*|r|")] = +0.05

    # Heading is especially sensitive to |psi| and r dynamics
    w_psi[names.index("|psi|")] = +0.20
    w_psi[names.index("r^2")] = +0.20
    w_psi[names.index("|r|")] = +0.10
    w_psi[names.index("v^2")] = +0.05

    # Wrap in CovarianceModel shape; GT uses raw feats (mu=0, sigma=1), calib=1
    mu = np.zeros(F)
    sigma = np.ones(F)
    calib = np.ones(3)

    return CovarianceModel(
        w_x=w_x, w_y=w_y, w_psi=w_psi,
        feature_names=names, mu=mu, sigma=sigma,
        calib_scale=calib, eps=1e-12
    )


# ---------------------------
# Data generation
# ---------------------------

def sample_states(n: int):
    x = rng.uniform(-3.0, 3.0, size=n)
    y = rng.uniform(-3.0, 3.0, size=n)
    psi = rng.uniform(-math.pi, math.pi, size=n)
    v = rng.uniform(-1.2, 1.8, size=n)    # m/s
    r = rng.uniform(-1.2, 1.2, size=n)    # rad/s
    return x, y, psi, v, r


def get_covariance_gt(pos: Position, psi: float, v: float, r: float, gt: CovarianceModel) -> np.ndarray:
    feats, _ = _features_from_state(pos.x, pos.y, psi, v, r)
    var_x = float(np.exp(feats @ gt.w_x))
    var_y = float(np.exp(feats @ gt.w_y))
    var_psi = float(np.exp(feats @ gt.w_psi))
    return np.diag([var_x, var_y, var_psi])


def generate_dataset(n: int, gt: CovarianceModel) -> list[DataPoint]:
    data: list[DataPoint] = []
    x, y, psi, v, r = sample_states(n)
    for i in range(n):
        pos_true = Position(float(x[i]), float(y[i]))
        psi_true = float(psi[i])

        cov = get_covariance_gt(pos_true, psi_true, float(v[i]), float(r[i]), gt)
        stds = np.sqrt(np.diag(cov))

        noise = rng.normal(0.0, stds, size=3)
        pos_meas = Position(pos_true.x + float(noise[0]), pos_true.y + float(noise[1]))
        psi_meas = psi_true + float(noise[2])

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


# ---------------------------
# Evaluation helpers
# ---------------------------

def evaluate_model(cm: CovarianceModel, data: list[DataPoint]) -> dict:
    eps = 1e-12
    preds = []
    emps = []
    for d in data:
        cov = get_covariance(d.position_measured, d.psi_measured, d.v, d.psi_dot, cm)
        var_pred = np.diag(cov)

        rx = d.position_measured.x - d.position_true.x
        ry = d.position_measured.y - d.position_true.y
        rpsi = d.psi_measured - d.psi_true
        rpsi = (rpsi + np.pi) % (2*np.pi) - np.pi
        var_emp = np.array([rx*rx, ry*ry, rpsi*rpsi]) + eps

        preds.append(var_pred)
        emps.append(var_emp)

    P = np.vstack(preds)
    T = np.vstack(emps)

    rmse_log = np.sqrt(np.mean((np.log(P) - np.log(T))**2, axis=0))
    calib = np.mean(T / P, axis=0)
    return {
        "rmse_log_x": float(rmse_log[0]),
        "rmse_log_y": float(rmse_log[1]),
        "rmse_log_psi": float(rmse_log[2]),
        "calib_x": float(calib[0]),
        "calib_y": float(calib[1]),
        "calib_psi": float(calib[2]),
    }


# ---------------------------
# Plotting: slices & empirical bins
# ---------------------------

def _bin_stats(x: np.ndarray, y: np.ndarray, bins: int = 20):
    edges = np.linspace(x.min(), x.max(), bins+1)
    idx = np.digitize(x, edges) - 1
    centers = 0.5*(edges[1:] + edges[:-1])
    means = np.zeros(bins)
    counts = np.zeros(bins)
    for b in range(bins):
        mask = idx == b
        if np.any(mask):
            means[b] = np.mean(y[mask])
            counts[b] = np.sum(mask)
        else:
            means[b] = np.nan
    return centers, means, counts


def plot_slices(gt: CovarianceModel, learned: CovarianceModel, test: list[DataPoint], out_prefix: str = "slice"):
    xs = np.array([d.position_measured.x for d in test])
    ys = np.array([d.position_measured.y for d in test])
    psis = np.array([d.psi_measured for d in test])
    vs = np.array([d.v for d in test])
    rs = np.array([d.psi_dot for d in test])

    x0 = float(np.median(xs))
    y0 = float(np.median(ys))
    psi0 = float(np.median(psis))
    v0 = float(np.median(vs))
    r0 = float(np.median(rs))
    pos0 = Position(x0, y0)

    def pred_all(pos: Position, psi: float, v: float, r: float):
        # GT uses raw feature model
        cov_gt = get_covariance_gt(pos, psi, v, r, gt)
        # learned uses normalized + calibrated model
        cov_learned = get_covariance(pos, psi, v, r, learned)
        return np.diag(cov_gt), np.diag(cov_learned)

    def plot_one(variable: str, grid: np.ndarray):
        gt_vars = []
        lr_vars = []
        for val in grid:
            if variable == "dist":
                pos = Position(val, 0.0)  # move along x to control distance
                psi, v, r = psi0, v0, r0
            elif variable == "|psi|":
                pos = pos0
                psi = float(np.sign(psi0) or 1.0) * float(val)
                v, r = v0, r0
            elif variable == "v":
                pos = pos0
                psi = psi0
                v = float(val)
                r = r0
            elif variable == "r":
                pos = pos0
                psi = psi0
                v = v0
                r = float(val)
            else:
                raise ValueError("unknown variable")

            g, l = pred_all(pos, psi, v, r)
            gt_vars.append(g)
            lr_vars.append(l)

        gt_vars = np.vstack(gt_vars)
        lr_vars = np.vstack(lr_vars)

        # Empirical bins from test set for this variable
        if variable == "dist":
            dist_test = np.hypot(xs, ys)
            emp_x = (np.array([d.position_measured.x - d.position_true.x for d in test])**2)
            emp_y = (np.array([d.position_measured.y - d.position_true.y for d in test])**2)
            emp_p = (np.array([
                ((d.psi_measured - d.psi_true + np.pi) % (2*np.pi) - np.pi) for d in test
            ])**2)
            bx, mx, _ = _bin_stats(dist_test, emp_x)
            by, my, _ = _bin_stats(dist_test, emp_y)
            bp, mp, _ = _bin_stats(dist_test, emp_p)
            xs_emp, ys_emp, ps_emp = bx, by, bp
        elif variable == "|psi|":
            abspsi = np.abs(psis)
            emp_x = (np.array([d.position_measured.x - d.position_true.x for d in test])**2)
            emp_y = (np.array([d.position_measured.y - d.position_true.y for d in test])**2)
            emp_p = (np.array([
                ((d.psi_measured - d.psi_true + np.pi) % (2*np.pi) - np.pi) for d in test
            ])**2)
            bx, mx, _ = _bin_stats(abspsi, emp_x)
            by, my, _ = _bin_stats(abspsi, emp_y)
            bp, mp, _ = _bin_stats(abspsi, emp_p)
            xs_emp, ys_emp, ps_emp = bx, by, bp
        elif variable == "v":
            emp_x = (np.array([d.position_measured.x - d.position_true.x for d in test])**2)
            emp_y = (np.array([d.position_measured.y - d.position_true.y for d in test])**2)
            emp_p = (np.array([
                ((d.psi_measured - d.psi_true + np.pi) % (2*np.pi) - np.pi) for d in test
            ])**2)
            bx, mx, _ = _bin_stats(vs, emp_x)
            by, my, _ = _bin_stats(vs, emp_y)
            bp, mp, _ = _bin_stats(vs, emp_p)
            xs_emp, ys_emp, ps_emp = bx, by, bp
        elif variable == "r":
            emp_x = (np.array([d.position_measured.x - d.position_true.x for d in test])**2)
            emp_y = (np.array([d.position_measured.y - d.position_true.y for d in test])**2)
            emp_p = (np.array([
                ((d.psi_measured - d.psi_true + np.pi) % (2*np.pi) - np.pi) for d in test
            ])**2)
            bx, mx, _ = _bin_stats(rs, emp_x)
            by, my, _ = _bin_stats(rs, emp_y)
            bp, mp, _ = _bin_stats(rs, emp_p)
            xs_emp, ys_emp, ps_emp = bx, by, bp
        else:
            raise ValueError

        # Plot three separate figures (one per axis)
        for axis_name, emp_xs, emp_mean, gt_y, lr_y in [
            ("x", xs_emp, mx, gt_vars[:, 0], lr_vars[:, 0]),
            ("y", ys_emp, my, gt_vars[:, 1], lr_vars[:, 1]),
            ("psi", ps_emp, mp, gt_vars[:, 2], lr_vars[:, 2]),
        ]:
            plt.figure(figsize=(6, 4))
            plt.plot(grid, gt_y, label="Ground truth", linewidth=2)
            plt.plot(grid, lr_y, label="Learned", linewidth=2)
            plt.scatter(emp_xs, emp_mean, s=20, alpha=0.7, label="Empirical (binned)")
            plt.xlabel(variable)
            plt.ylabel(f"Var({axis_name})")
            plt.title(f"Slice: Var({axis_name}) vs {variable}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{out_prefix}_{axis_name}_vs_{variable}.png", dpi=140)
            plt.close()

    # Grids
    dist_grid = np.linspace(0.0, 5.0, 100)
    psi_abs_grid = np.linspace(0.0, np.pi, 100)
    v_grid = np.linspace(-1.2, 1.8, 120)
    r_grid = np.linspace(-1.2, 1.2, 120)

    plot_one("dist", dist_grid)
    plot_one("|psi|", psi_abs_grid)
    plot_one("v", v_grid)
    plot_one("r", r_grid)


# ---------------------------
# Main
# ---------------------------

def main():
    gt = make_ground_truth_model()
    print("=== Ground-truth weights (reference) ===")
    print(describe_model(gt))
    print()

    # Bigger data by default; adjust if needed
    n_train = 100_000
    n_test = 20_000

    print(f"Generating {n_train} training and {n_test} test samples...")
    train = generate_dataset(n_train, gt)
    test = generate_dataset(n_test, gt)

    # Fit improved model (stable NLL + Adam + clipping + warm start)
    print("Fitting improved covariance model...")
    learned = estimate_covariance_model(
        train,
        l2=5e-4,
        lr=7e-3,
        steps=3000,
        calib_split=0.2,
    )

    print("\n=== Learned weights (normalized space) & calibration ===")
    print(describe_model(learned))
    print()

    # Evaluate on test set
    metrics = evaluate_model(learned, test)
    print("=== Test metrics (improved model) ===")
    for k, v in metrics.items():
        print(f"{k:>14s}: {v:.4f}")

    # Oracle baseline on same test set
    def evaluate_oracle(gt_model: CovarianceModel, data: list[DataPoint]) -> dict:
        eps = 1e-12
        preds, emps = [], []
        for d in data:
            cov = get_covariance_gt(d.position_measured, d.psi_measured, d.v, d.psi_dot, gt_model)
            var_pred = np.diag(cov)
            rx = d.position_measured.x - d.position_true.x
            ry = d.position_measured.y - d.position_true.y
            rpsi = d.psi_measured - d.psi_true
            rpsi = (rpsi + np.pi) % (2*np.pi) - np.pi
            var_emp = np.array([rx*rx, ry*ry, rpsi*rpsi]) + eps
            preds.append(var_pred)
            emps.append(var_emp)
        P = np.vstack(preds)
        T = np.vstack(emps)
        rmse_log = np.sqrt(np.mean((np.log(P) - np.log(T))**2, axis=0))
        calib = np.mean(T / P, axis=0)
        return {
            "rmse_log_x": float(rmse_log[0]),
            "rmse_log_y": float(rmse_log[1]),
            "rmse_log_psi": float(rmse_log[2]),
            "calib_x": float(calib[0]),
            "calib_y": float(calib[1]),
            "calib_psi": float(calib[2]),
        }

    gt_metrics = evaluate_oracle(gt, test)
    print("\n=== Ground-truth model (oracle) metrics on test set ===")
    for k, v in gt_metrics.items():
        print(f"{k:>14s}: {v:.4f}")

    # Plots
    print("\nGenerating slice plots (PNG files)...")
    plot_slices(gt, learned, test, out_prefix="slice")
    print("Saved files: slice_x_vs_dist.png, slice_y_vs_dist.png, slice_psi_vs_dist.png, etc.")


if __name__ == "__main__":
    main()