# simulate_covariance_recovery.py
# ============================================================
# Simulates noisy 2D+angle measurements from a known covariance
# model, fits covariance_model.py, evaluates recovery, and
# produces simple visual diagnostics.
# ============================================================

import math
import dataclasses
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from aruco_covariance import (
    Position,
    DataPoint,
    estimate_covariance_model,
    get_covariance,
    calibrate_alpha,  # scalar alpha (optional)
    calibrate_alpha_function,  # functional alpha(phi)
)


# ============================================================
# === Ground-truth model =====================================
# ============================================================

def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _features(position: Position, psi: float, v: float, r: float) -> np.ndarray:
    x, y = position.x, position.y
    d = float(np.hypot(x, y))
    cpsi, spsi = math.cos(psi), math.sin(psi)
    v_abs, r_abs = abs(v), abs(r)
    return np.array([
        1.0, d, d ** 2, cpsi, spsi, x, y,
        v_abs, v_abs ** 2, r_abs, r_abs ** 2,
                v_abs * d, r_abs * d, v_abs * r_abs,
    ], dtype=float)


def _nearest_psd(A: np.ndarray, eps: float = 0.0) -> np.ndarray:
    A = 0.5 * (A + A.T)
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, eps)
    return (vecs * vals) @ vecs.T


@dataclasses.dataclass
class TrueParams:
    w_var_xx: np.ndarray
    w_var_yy: np.ndarray
    w_var_pp: np.ndarray
    w_cov_xy: np.ndarray
    w_cov_xp: np.ndarray
    w_cov_yp: np.ndarray
    jitter: float = 1e-10


def make_default_true_params() -> TrueParams:
    F = 14

    def w(*pairs):
        v = np.zeros(F, dtype=float)
        for idx, val in pairs:
            v[idx] = val
        return v

    # Feature indices
    BIAS = 0;
    D = 1;
    D2 = 2;
    CPSI = 3;
    SPSI = 4;
    X = 5;
    Y = 6;
    V = 7;
    V2 = 8;
    R = 9;
    R2 = 10;
    VxD = 11;
    RxD = 12;
    VxR = 13

    # Modest dependencies; turning rate drives psi variance strongly
    w_xx = w((BIAS, -7.3), (D, 0.25), (D2, 0.10), (V, 0.15), (R, 0.35))
    w_yy = w((BIAS, -7.1), (D, 0.20), (D2, 0.08), (V, 0.12), (R, 0.30))
    w_pp = w((BIAS, -6.5), (D, 0.10), (R, 0.50), (R2, 0.20))

    # Small cross-covariances
    w_xy = w((BIAS, 0.00), (SPSI, 0.01), (R, 0.03))
    w_xp = w((BIAS, 0.00), (CPSI, -0.01), (R, 0.015))
    w_yp = w((BIAS, 0.00), (SPSI, 0.01), (R, -0.015))

    return TrueParams(w_xx, w_yy, w_pp, w_xy, w_xp, w_yp)


def true_covariance(position: Position, psi: float, v: float, r: float, p: TrueParams) -> np.ndarray:
    phi = _features(position, psi, v, r)
    var_x = math.exp(phi @ p.w_var_xx)
    var_y = math.exp(phi @ p.w_var_yy)
    var_p = math.exp(phi @ p.w_var_pp)
    cov_xy = float(phi @ p.w_cov_xy)
    cov_xp = float(phi @ p.w_cov_xp)
    cov_yp = float(phi @ p.w_cov_yp)
    S = np.array([
        [var_x, cov_xy, cov_xp],
        [cov_xy, var_y, cov_yp],
        [cov_xp, cov_yp, var_p],
    ], dtype=float)
    S[np.diag_indices_from(S)] += p.jitter
    return _nearest_psd(S, eps=1e-12)


# ============================================================
# === Simulation =============================================
# ============================================================

def simulate_dataset(N: int, params: TrueParams, seed: int = 0) -> List[DataPoint]:
    rng = np.random.default_rng(seed)
    data: List[DataPoint] = []

    for _ in range(N):
        x = rng.uniform(-1.2, 1.2)
        y = rng.uniform(-1.5, 1.5)
        psi = rng.uniform(-math.pi, math.pi)
        v = rng.uniform(-1.0, 1.0)
        r = rng.uniform(-2.0, 2.0)

        pos_true = Position(x, y)
        Sigma = true_covariance(pos_true, psi, v, r, params)
        noise = rng.multivariate_normal(np.zeros(3), Sigma)
        dx, dy, dpsi = noise
        pos_meas = Position(x + dx, y + dy)
        psi_meas = _wrap_angle(psi + dpsi)

        data.append(DataPoint(pos_meas, pos_true, psi_meas, psi, v, r))
    return data


def elementwise_errors(S_true: np.ndarray, S_pred: np.ndarray) -> Tuple[float, float, float]:
    var_idx = [(0, 0), (1, 1), (2, 2)]
    cov_idx = [(0, 1), (0, 2), (1, 2)]
    var_mae = np.mean([abs(S_true[i, j] - S_pred[i, j]) for (i, j) in var_idx])
    cov_mae = np.mean([abs(S_true[i, j] - S_pred[i, j]) for (i, j) in cov_idx])
    frob = np.linalg.norm(S_true - S_pred, ord='fro')
    return var_mae, cov_mae, frob


# ============================================================
# === Diagnostics / Visualization ============================
# ============================================================

def heatmap_mean_D2_vs_d_r(val_data: List[DataPoint],
                           cov_model,
                           title: str = "Mean D^2 over (distance d, |r|)"):
    """
    Bin the validation set by distance d and |r|; compute mean D^2 using the
    *predicted* covariance, to see where we're under/over confident.
    """
    # Binning
    d_vals = []
    r_vals = []
    D2_vals = []

    for dp in val_data:
        # residual
        rvec = np.array([
            dp.position_measured.x - dp.position_true.x,
            dp.position_measured.y - dp.position_true.y,
            _wrap_angle(dp.psi_measured - dp.psi_true)
        ], dtype=float)

        # features
        d = float(np.hypot(dp.position_true.x, dp.position_true.y))
        rabs = abs(dp.psi_dot)

        S_pred = get_covariance(dp.position_true, dp.psi_true, dp.v, dp.psi_dot, cov_model)
        Sinv = np.linalg.pinv(S_pred)
        D2 = float(rvec @ Sinv @ rvec)

        d_vals.append(d)
        r_vals.append(rabs)
        D2_vals.append(D2)

    d_bins = np.linspace(0.0, 2.0, 21)  # 0..2m
    r_bins = np.linspace(0.0, 2.0, 21)  # 0..2 rad/s
    H = np.zeros((len(d_bins) - 1, len(r_bins) - 1), dtype=float)
    C = np.zeros_like(H)

    d_vals = np.asarray(d_vals)
    r_vals = np.asarray(r_vals)
    D2_vals = np.asarray(D2_vals)

    di = np.digitize(d_vals, d_bins) - 1
    ri = np.digitize(r_vals, r_bins) - 1

    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            mask = (di == i) & (ri == j)
            if np.any(mask):
                H[i, j] = float(np.mean(D2_vals[mask]))
                C[i, j] = np.sum(mask)
            else:
                H[i, j] = np.nan
                C[i, j] = 0

    # plot
    fig = plt.figure(figsize=(7, 5))
    im = plt.imshow(H.T, origin="lower",
                    extent=[d_bins[0], d_bins[-1], r_bins[0], r_bins[-1]],
                    aspect="auto")
    plt.colorbar(im, label="Mean D^2 (pred Σ)")
    plt.xlabel("distance d [m]")
    plt.ylabel("|r| [rad/s]")
    plt.title(title + " (white = no data)")
    # overlay low-count cells
    yy, xx = np.meshgrid(0.5 * (r_bins[:-1] + r_bins[1:]), 0.5 * (d_bins[:-1] + d_bins[1:]))
    low = C.T < 10  # highlight sparse bins
    plt.scatter(xx[low], yy[low], marker="x", s=10, alpha=0.6, label="low count")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("diagnostic_heatmap_meanD2.png", dpi=140)
    plt.close(fig)


def plot_covariance_slices(params_true: TrueParams,
                           cov_model,
                           base_position: Position = Position(0.5, 0.0),
                           base_psi: float = 0.0,
                           base_v: float = 0.0,
                           base_r: float = 0.0,
                           out_prefix: str = "slice"):
    """
    Plot 1D slices of the covariance elements versus a chosen variable, holding
    others fixed. We do four slices: vs distance d, vs psi, vs v, vs r.
    For each slice, we plot diag variances (σ_x^2, σ_y^2, σ_ψ^2) and off-diagonals.
    """

    def pred_cov(p: Position, psi: float, v: float, r: float):
        return get_covariance(p, psi, v, r, cov_model)

    def true_cov(p: Position, psi: float, v: float, r: float):
        return true_covariance(p, psi, v, r, params_true)

    # ---- 1) Slice vs distance d (along +x with y=0) ----
    ds = np.linspace(0.05, 2.0, 100)
    pred_diag = [];
    true_diag = []
    pred_off = [];
    true_off = []
    for d in ds:
        p = Position(d, 0.0)
        S_pred = pred_cov(p, base_psi, base_v, base_r)
        S_true = true_cov(p, base_psi, base_v, base_r)
        pred_diag.append([S_pred[0, 0], S_pred[1, 1], S_pred[2, 2]])
        true_diag.append([S_true[0, 0], S_true[1, 1], S_true[2, 2]])
        pred_off.append([S_pred[0, 1], S_pred[0, 2], S_pred[1, 2]])
        true_off.append([S_true[0, 1], S_true[0, 2], S_true[1, 2]])
    pred_diag = np.array(pred_diag);
    true_diag = np.array(true_diag)
    pred_off = np.array(pred_off);
    true_off = np.array(true_off)

    fig = plt.figure(figsize=(8, 5))
    for k, lbl in enumerate([r"$\sigma_x^2$", r"$\sigma_y^2$", r"$\sigma_\psi^2$"]):
        plt.plot(ds, true_diag[:, k], linestyle="--", label=f"true {lbl}")
        plt.plot(ds, pred_diag[:, k], label=f"pred {lbl}")
    plt.xlabel("distance d [m]")
    plt.ylabel("variance")
    plt.title("Covariance slice vs distance d (y=0, ψ=0, v=0, r=0)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_var_vs_d.png", dpi=140);
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    for k, lbl in enumerate([r"cov(x,y)", r"cov(x,ψ)", r"cov(y,ψ)"]):
        plt.plot(ds, true_off[:, k], linestyle="--", label=f"true {lbl}")
        plt.plot(ds, pred_off[:, k], label=f"pred {lbl}")
    plt.xlabel("distance d [m]")
    plt.ylabel("covariance")
    plt.title("Off-diagonal slice vs distance d")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_cov_vs_d.png", dpi=140);
    plt.close(fig)

    # ---- 2) Slice vs ψ (angle) ----
    psis = np.linspace(-math.pi, math.pi, 200)
    pred_diag = [];
    true_diag = []
    pred_off = [];
    true_off = []
    p = base_position
    for psi in psis:
        S_pred = pred_cov(p, psi, base_v, base_r)
        S_true = true_cov(p, psi, base_v, base_r)
        pred_diag.append([S_pred[0, 0], S_pred[1, 1], S_pred[2, 2]])
        true_diag.append([S_true[0, 0], S_true[1, 1], S_true[2, 2]])
        pred_off.append([S_pred[0, 1], S_pred[0, 2], S_pred[1, 2]])
        true_off.append([S_true[0, 1], S_true[0, 2], S_true[1, 2]])
    pred_diag = np.array(pred_diag);
    true_diag = np.array(true_diag)
    pred_off = np.array(pred_off);
    true_off = np.array(true_off)

    fig = plt.figure(figsize=(8, 5))
    for k, lbl in enumerate([r"$\sigma_x^2$", r"$\sigma_y^2$", r"$\sigma_\psi^2$"]):
        plt.plot(psis, true_diag[:, k], linestyle="--", label=f"true {lbl}")
        plt.plot(psis, pred_diag[:, k], label=f"pred {lbl}")
    plt.xlabel("ψ [rad]")
    plt.ylabel("variance")
    plt.title(f"Covariance slice vs ψ (pos={base_position}, v={base_v}, r={base_r})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_var_vs_psi.png", dpi=140);
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    for k, lbl in enumerate([r"cov(x,y)", r"cov(x,ψ)", r"cov(y,ψ)"]):
        plt.plot(psis, true_off[:, k], linestyle="--", label=f"true {lbl}")
        plt.plot(psis, pred_off[:, k], label=f"pred {lbl}")
    plt.xlabel("ψ [rad]")
    plt.ylabel("covariance")
    plt.title("Off-diagonal slice vs ψ")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_cov_vs_psi.png", dpi=140);
    plt.close(fig)

    # ---- 3) Slice vs v (forward speed) ----
    vs = np.linspace(-1.0, 1.0, 120)
    pred_diag = [];
    true_diag = []
    pred_off = [];
    true_off = []
    p = base_position;
    psi = base_psi
    for v in vs:
        S_pred = pred_cov(p, psi, v, base_r)
        S_true = true_cov(p, psi, v, base_r)
        pred_diag.append([S_pred[0, 0], S_pred[1, 1], S_pred[2, 2]])
        true_diag.append([S_true[0, 0], S_true[1, 1], S_true[2, 2]])
        pred_off.append([S_pred[0, 1], S_pred[0, 2], S_pred[1, 2]])
        true_off.append([S_true[0, 1], S_true[0, 2], S_true[1, 2]])
    pred_diag = np.array(pred_diag);
    true_diag = np.array(true_diag)
    pred_off = np.array(pred_off);
    true_off = np.array(true_off)

    fig = plt.figure(figsize=(8, 5))
    for k, lbl in enumerate([r"$\sigma_x^2$", r"$\sigma_y^2$", r"$\sigma_\psi^2$"]):
        plt.plot(vs, true_diag[:, k], linestyle="--", label=f"true {lbl}")
        plt.plot(vs, pred_diag[:, k], label=f"pred {lbl}")
    plt.xlabel("v [m/s]")
    plt.ylabel("variance")
    plt.title(f"Covariance slice vs v (pos={base_position}, ψ={base_psi}, r={base_r})")
    plt.legend();
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_var_vs_v.png", dpi=140);
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    for k, lbl in enumerate([r"cov(x,y)", r"cov(x,ψ)", r"cov(y,ψ)"]):
        plt.plot(vs, true_off[:, k], linestyle="--", label=f"true {lbl}")
        plt.plot(vs, pred_off[:, k], label=f"pred {lbl}")
    plt.xlabel("v [m/s]")
    plt.ylabel("covariance")
    plt.title("Off-diagonal slice vs v")
    plt.legend();
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_cov_vs_v.png", dpi=140);
    plt.close(fig)

    # ---- 4) Slice vs r (turning rate) ----
    rs = np.linspace(-2.0, 2.0, 160)
    pred_diag = [];
    true_diag = []
    pred_off = [];
    true_off = []
    p = base_position;
    psi = base_psi
    for r in rs:
        S_pred = pred_cov(p, psi, base_v, r)
        S_true = true_cov(p, psi, base_v, r)
        pred_diag.append([S_pred[0, 0], S_pred[1, 1], S_pred[2, 2]])
        true_diag.append([S_true[0, 0], S_true[1, 1], S_true[2, 2]])
        pred_off.append([S_pred[0, 1], S_pred[0, 2], S_pred[1, 2]])
        true_off.append([S_true[0, 1], S_true[0, 2], S_true[1, 2]])
    pred_diag = np.array(pred_diag);
    true_diag = np.array(true_diag)
    pred_off = np.array(pred_off);
    true_off = np.array(true_off)

    fig = plt.figure(figsize=(8, 5))
    for k, lbl in enumerate([r"$\sigma_x^2$", r"$\sigma_y^2$", r"$\sigma_\psi^2$"]):
        plt.plot(rs, true_diag[:, k], linestyle="--", label=f"true {lbl}")
        plt.plot(rs, pred_diag[:, k], label=f"pred {lbl}")
    plt.xlabel("r [rad/s]")
    plt.ylabel("variance")
    plt.title(f"Covariance slice vs r (pos={base_position}, ψ={base_psi}, v={base_v})")
    plt.legend();
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_var_vs_r.png", dpi=140);
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    for k, lbl in enumerate([r"cov(x,y)", r"cov(x,ψ)", r"cov(y,ψ)"]):
        plt.plot(rs, true_off[:, k], linestyle="--", label=f"true {lbl}")
        plt.plot(rs, pred_off[:, k], label=f"pred {lbl}")
    plt.xlabel("r [rad/s]")
    plt.ylabel("covariance")
    plt.title("Off-diagonal slice vs r")
    plt.legend();
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_cov_vs_r.png", dpi=140);
    plt.close(fig)


# ============================================================
# === Main experiment ========================================
# ============================================================

def main():
    params = make_default_true_params()
    train = simulate_dataset(6000, params, seed=1)
    val = simulate_dataset(2000, params, seed=11)
    test = simulate_dataset(2000, params, seed=2)

    cov_model = estimate_covariance_model(train)

    # --- Calibration ---
    # Option A: global scalar alpha
    # alpha = calibrate_alpha(val, cov_model)
    # print(f"[Calibration] Set global alpha = {alpha:.4f}")

    # Option B: functional alpha(phi) for tighter tails
    w_alpha = calibrate_alpha_function(val, cov_model)
    print(f"[Calibration] Learned alpha(ϕ) with {w_alpha.size} weights")

    # --- Evaluation ---
    var_maes, cov_maes, frobs, D2_list = [], [], [], []
    for dp in test:
        S_true = true_covariance(dp.position_true, dp.psi_true, dp.v, dp.psi_dot, params)
        S_pred = get_covariance(dp.position_true, dp.psi_true, dp.v, dp.psi_dot, cov_model)

        vm, cm, f = elementwise_errors(S_true, S_pred)
        var_maes.append(vm);
        cov_maes.append(cm);
        frobs.append(f)

        r_vec = np.array([
            dp.position_measured.x - dp.position_true.x,
            dp.position_measured.y - dp.position_true.y,
            _wrap_angle(dp.psi_measured - dp.psi_true)
        ], dtype=float)
        Sinv = np.linalg.pinv(S_pred)
        D2_list.append(float(r_vec @ Sinv @ r_vec))

    print("=== Recovery quality on held-out set ===")
    print(f"Var MAE (avg of diag terms): {np.mean(var_maes):.4e}")
    print(f"Cov MAE (avg of off-diag):   {np.mean(cov_maes):.4e}")
    print(f"Frobenius error (mean):       {np.mean(frobs):.4e}")

    D2_arr = np.array(D2_list)
    print("\n=== Mahalanobis consistency (pred Σ after calibration) ===")
    print(f"Mean D^2: {D2_arr.mean():.3f}  (ideal ~ 3.0)")
    print(f"Std  D^2: {D2_arr.std():.3f}  (ideal ~ sqrt(6) ≈ 2.45)")
    print(
        f"Quantiles D^2: 5%={np.quantile(D2_arr, 0.05):.2f}, 50%={np.quantile(D2_arr, 0.50):.2f}, 95%={np.quantile(D2_arr, 0.95):.2f}")

    # --- Visual diagnostics ---
    heatmap_mean_D2_vs_d_r(val, cov_model,
                           title="Mean D^2 over (distance, |r|) — validation set")
    print("Saved: diagnostic_heatmap_meanD2.png")

    plot_covariance_slices(params, cov_model,
                           base_position=Position(0.6, 0.0),
                           base_psi=0.0, base_v=0.0, base_r=0.0,
                           out_prefix="slice")
    print(
        "Saved: slice_var_vs_d.png, slice_cov_vs_d.png, slice_var_vs_psi.png, slice_cov_vs_psi.png, slice_var_vs_v.png, slice_cov_vs_v.png, slice_var_vs_r.png, slice_cov_vs_r.png")


if __name__ == "__main__":
    main()
