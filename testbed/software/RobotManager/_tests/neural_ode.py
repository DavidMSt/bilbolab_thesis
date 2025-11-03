#!/usr/bin/env python3
"""
Neural ODE for a forced mass–spring–damper (m=1):
    x'' + c x' + k x = u(t)

We learn the parameters k, c by simulating the ODE with a differentiable
RK4 integrator in PyTorch and minimizing data mismatch. At test time,
we demonstrate generalization to a new input u_test(t) and a new IC.

Author: you
"""

import math
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# For stable gradients through integration
torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cpu")


# -----------------------------
# Forcing functions u(t)
# -----------------------------
def u_train_np(t: np.ndarray) -> np.ndarray:
    """Training forcing (NumPy) for data generation."""
    return (
            np.sin(1.0 * t)
            + 0.7 * np.cos(1.7 * t)
            + 0.4 * np.sin(2.3 * t + 0.5)
    )


def u_train_torch(t: torch.Tensor) -> torch.Tensor:
    """Training forcing (Torch) used during training/integration."""
    return (
            torch.sin(1.0 * t)
            + 0.7 * torch.cos(1.7 * t)
            + 0.4 * torch.sin(2.3 * t + 0.5)
    )


def u_test_np(t: np.ndarray) -> np.ndarray:
    """A different input for testing generalization."""
    return (
            0.8 * np.sin(0.8 * t + 0.3)
            + 0.6 * np.cos(2.2 * t + 0.1)
            + 0.3 * np.sin(3.1 * t - 0.2)
    )


def u_test_torch(t: torch.Tensor) -> torch.Tensor:
    """Torch version of the test input."""
    return (
            0.8 * torch.sin(0.8 * t + 0.3)
            + 0.6 * torch.cos(2.2 * t + 0.1)
            + 0.3 * torch.sin(3.1 * t - 0.2)
    )


# -----------------------------
# Ground-truth simulator (NumPy RK4, for synthetic data)
# -----------------------------
def simulate_np(
        k: float, c: float,
        x0: float, v0: float,
        t_max: float, n_steps: int,
        u_func_np
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate x'' = u(t) - c x' - k x via RK4 (NumPy) on a uniform grid.
    Returns t, x, v (1D arrays of len n_steps).
    """
    t = np.linspace(0.0, t_max, n_steps)
    dt = t[1] - t[0]
    x = np.zeros_like(t)
    v = np.zeros_like(t)
    x[0] = x0
    v[0] = v0

    def f_x(x_, v_, tt):
        return v_

    def f_v(x_, v_, tt):
        return u_func_np(np.array([tt]))[0] - c * v_ - k * x_

    for i in range(n_steps - 1):
        ti = t[i]

        k1_x = f_x(x[i], v[i], ti)
        k1_v = f_v(x[i], v[i], ti)

        k2_x = f_x(x[i] + 0.5 * dt * k1_x, v[i] + 0.5 * dt * k1_v, ti + 0.5 * dt)
        k2_v = f_v(x[i] + 0.5 * dt * k1_x, v[i] + 0.5 * dt * k1_v, ti + 0.5 * dt)

        k3_x = f_x(x[i] + 0.5 * dt * k2_x, v[i] + 0.5 * dt * k2_v, ti + 0.5 * dt)
        k3_v = f_v(x[i] + 0.5 * dt * k2_x, v[i] + 0.5 * dt * k2_v, ti + 0.5 * dt)

        k4_x = f_x(x[i] + dt * k3_x, v[i] + dt * k3_v, ti + dt)
        k4_v = f_v(x[i] + dt * k3_x, v[i] + dt * k3_v, ti + dt)

        x[i + 1] = x[i] + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        v[i + 1] = v[i] + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    return t, x, v


# -----------------------------
# Differentiable Torch RK4 integrator for training/inference
# -----------------------------
def simulate_torch(
        k_pos: torch.Tensor, c_pos: torch.Tensor,
        x0: torch.Tensor, v0: torch.Tensor,
        t: torch.Tensor,
        u_func_torch
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable RK4 without in-place writes to graph-tracked tensors.
    Returns x, v with shapes (T, B).
    """
    assert t.ndim == 2 and t.shape[1] == 1, "t must be (T,1)"
    T = t.shape[0]
    dt = t[1, 0] - t[0, 0]
    B = x0.shape[0]

    # Precompute u(t) once (keeps the graph intact)
    u_all = u_func_torch(t).squeeze(1)  # (T,)

    def f_x(x_, v_, _ti):
        return v_

    def f_v(x_, v_, ti_idx):
        # broadcast u_i to batch
        ui = u_all[ti_idx].expand(B)  # (B,)
        return ui - c_pos * v_ - k_pos * x_

    # Start from ICs; keep separate tensors per step (no in-place into a history tensor)
    x_i = x0  # (B,)
    v_i = v0  # (B,)
    xs = [x_i]
    vs = [v_i]

    for i in range(T - 1):
        ti_idx = i
        # k1
        k1_x = f_x(x_i, v_i, ti_idx)
        k1_v = f_v(x_i, v_i, ti_idx)

        # k2
        x_k2 = x_i + 0.5 * dt * k1_x
        v_k2 = v_i + 0.5 * dt * k1_v
        k2_x = f_x(x_k2, v_k2, ti_idx)
        k2_v = f_v(x_k2, v_k2, ti_idx)

        # k3
        x_k3 = x_i + 0.5 * dt * k2_x
        v_k3 = v_i + 0.5 * dt * k2_v
        k3_x = f_x(x_k3, v_k3, ti_idx)
        k3_v = f_v(x_k3, v_k3, ti_idx)

        # k4 (use next time index for u(t+dt))
        x_k4 = x_i + dt * k3_x
        v_k4 = v_i + dt * k3_v
        k4_x = f_x(x_k4, v_k4, ti_idx + 1)
        k4_v = f_v(x_k4, v_k4, ti_idx + 1)

        # Next state (new tensors; no in-place into a stored history)
        x_next = x_i + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        v_next = v_i + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

        x_i, v_i = x_next, v_next
        xs.append(x_i)
        vs.append(v_i)

    x = torch.stack(xs, dim=0)  # (T, B)
    v = torch.stack(vs, dim=0)  # (T, B)
    return x, v


# -----------------------------
# Learnable gray-box model (learn k, c)
# -----------------------------
class MSDNeuralODE(nn.Module):
    """
    Parameterized physical model:
      dx/dt = v
      dv/dt = u(t) - c v - k x
    Learnable: k, c (kept positive with Softplus).
    """

    def __init__(self, k_init=1.0, c_init=1.0):
        super().__init__()
        self.softplus = nn.Softplus()
        # choose unconstrained so softplus(unconstr) ~ init
        self.k_unconstr = nn.Parameter(torch.tensor([math.log(math.exp(k_init) - 1.0)]))
        self.c_unconstr = nn.Parameter(torch.tensor([math.log(math.exp(c_init) - 1.0)]))

    @property
    def k(self):
        return self.softplus(self.k_unconstr)[0]

    @property
    def c(self):
        return self.softplus(self.c_unconstr)[0]

    def forward(self, t: torch.Tensor, x0: torch.Tensor, v0: torch.Tensor, u_func_torch):
        """
        Simulate and return x(t), v(t) with differentiable RK4.
        t: (T,1), x0,v0: (B,)
        """
        x, v = simulate_torch(self.k, self.c, x0, v0, t, u_func_torch)
        return x, v


# -----------------------------
# Dataset (training trajectory)
# -----------------------------
def make_training_data(
        n_data=400, t_max=10.0, noise_std=0.01, seed=0
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    rng = np.random.default_rng(seed)
    # True physical parameters (unknown to learner)
    k_true = 2.0
    c_true = 0.5
    # Initial condition used for training
    x0_true = 0.8
    v0_true = -0.2

    t_np, x_np, v_np = simulate_np(
        k_true, c_true, x0_true, v0_true, t_max, n_data, u_train_np
    )
    x_noisy = x_np + rng.normal(0.0, noise_std, size=x_np.shape)

    t = torch.tensor(t_np, device=DEVICE).reshape(-1, 1)
    x = torch.tensor(x_noisy, device=DEVICE).reshape(-1, 1)

    meta = dict(
        k_true=k_true, c_true=c_true,
        x0=x0_true, v0=v0_true,
        t_max=t_max, n_data=n_data, noise_std=noise_std
    )
    return t, x, meta


# -----------------------------
# Training
# -----------------------------
def train_model(
        epochs=4000, lr=1e-3, seed=0, print_every=400
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data
    t, x_meas, meta = make_training_data(n_data=500, t_max=12.0, noise_std=0.01, seed=seed)
    T = t.shape[0]

    # Batch size 1 trajectory; we still keep a batch dim for ICs
    x0 = torch.tensor([meta["x0"]], device=DEVICE)
    v0 = torch.tensor([meta["v0"]], device=DEVICE)

    # Model + optimizer
    model = MSDNeuralODE(k_init=1.0, c_init=1.0).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    mse = nn.MSELoss()

    for ep in range(1, epochs + 1):
        opt.zero_grad()
        x_sim, v_sim = model(t, x0, v0, u_train_torch)  # (T,1) each
        loss = mse(x_sim, x_meas)  # only position is measured
        loss.backward()
        opt.step()
        sched.step()

        if ep % print_every == 0 or ep == 1:
            print(f"Epoch {ep:5d} | loss={loss.item():.4e} "
                  f"| k={model.k.item():.4f} | c={model.c.item():.4f}")

    return model, (t, x_meas, meta)


# -----------------------------
# Evaluation & Generalization
# -----------------------------
def evaluate(model: MSDNeuralODE, dataset):
    t, x_meas, meta = dataset
    x0_train = torch.tensor([meta["x0"]], device=DEVICE)
    v0_train = torch.tensor([meta["v0"]], device=DEVICE)

    # Simulate on training input/IC
    with torch.no_grad():
        x_sim_train, v_sim_train = model(t, x0_train, v0_train, u_train_torch)

    # Clean ground truth for comparison (same u & IC)
    x_clean = simulate_np(
        meta["k_true"], meta["c_true"],
        meta["x0"], meta["v0"],
        meta["t_max"], meta["n_data"], u_train_np
    )[1]

    mse_noisy = torch.mean((x_sim_train - x_meas) ** 2).item()
    mse_clean = np.mean((x_sim_train.cpu().numpy().squeeze() - x_clean) ** 2)

    print("\n=== In-sample evaluation ===")
    print(f"True params:    k={meta['k_true']:.4f}, c={meta['c_true']:.4f}")
    print(f"Learned params: k={model.k.item():.4f}, c={model.c.item():.4f}")
    print(f"MSE vs noisy data:  {mse_noisy:.4e}")
    print(f"MSE vs clean truth: {mse_clean:.4e}")

    # Generalization: new input and new IC
    t_test = t.clone()  # same grid, different u and IC
    x0_test = torch.tensor([0.3], device=DEVICE)
    v0_test = torch.tensor([0.6], device=DEVICE)

    with torch.no_grad():
        x_sim_test, v_sim_test = model(t_test, x0_test, v0_test, u_test_torch)

    x_clean_test = simulate_np(
        meta["k_true"], meta["c_true"],
        x0_test.item(), v0_test.item(),
        meta["t_max"], meta["n_data"], u_test_np
    )[1]
    mse_test = np.mean((x_sim_test.cpu().numpy().squeeze() - x_clean_test) ** 2)

    print("\n=== Out-of-sample generalization ===")
    print(f"Test ICs: x0={x0_test.item():.3f}, v0={v0_test.item():.3f} | Different u(t)")
    print(f"MSE vs clean truth (test): {mse_test:.4e}")

    # --------- Plots ----------
    t_np = t.cpu().numpy().squeeze()

    plt.figure(figsize=(10, 6))
    plt.plot(t_np, x_clean, lw=2, label="Ground truth (train, clean)")
    plt.scatter(t_np, x_meas.cpu().numpy().squeeze(), s=10, alpha=0.5, label="Training data (noisy)")
    plt.plot(t_np, x_sim_train.cpu().numpy().squeeze(), lw=2, ls="--", label="Neural ODE (train)")
    plt.title(f"Neural ODE fit (train)\nLearned k={model.k.item():.3f}, c={model.c.item():.3f}")
    plt.xlabel("t");
    plt.ylabel("x(t)");
    plt.grid(True);
    plt.legend()
    plt.tight_layout();
    plt.savefig("neural_ode_train_fit.png", dpi=160)
    print("Saved: neural_ode_train_fit.png")

    plt.figure(figsize=(10, 6))
    plt.plot(t_np, x_clean_test, lw=2, label="Ground truth (test, clean)")
    plt.plot(t_np, x_sim_test.cpu().numpy().squeeze(), lw=2, ls="--", label="Neural ODE (test)")
    plt.title("Generalization to new u(t) and new IC")
    plt.xlabel("t");
    plt.ylabel("x(t)");
    plt.grid(True);
    plt.legend()
    plt.tight_layout();
    plt.savefig("neural_ode_test_generalization.png", dpi=160)
    print("Saved: neural_ode_test_generalization.png")

    # Plot inputs
    plt.figure(figsize=(10, 3.2))
    plt.plot(t_np, u_train_np(t_np), lw=1.5, label="u_train(t)")
    plt.plot(t_np, u_test_np(t_np), lw=1.5, label="u_test(t)")
    plt.title("Inputs");
    plt.xlabel("t");
    plt.ylabel("u(t)");
    plt.grid(True);
    plt.legend()
    plt.tight_layout();
    plt.savefig("neural_ode_inputs.png", dpi=160)
    print("Saved: neural_ode_inputs.png")

    plt.show()


# -----------------------------
# Main
# -----------------------------
def main():
    model, dataset = train_model(
        epochs=4000, lr=1e-3, seed=42, print_every=400
    )
    evaluate(model, dataset)


if __name__ == "__main__":
    main()
