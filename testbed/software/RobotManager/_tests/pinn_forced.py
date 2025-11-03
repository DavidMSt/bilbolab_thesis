#!/usr/bin/env python3
"""
Extended PINN for a forced mass-spring-damper system (m=1):
    x'' + c x' + k x = u(t)

Network input: [t, u(t)]  -> output: x(t)
Learnable physical parameters: k, c (kept positive via Softplus)

We:
  - simulate ground truth with RK4 under a rich input u(t),
  - add noise to obtain training data,
  - train a PINN with data + physics + initial condition losses,
  - evaluate and plot results.

Author: you + PINN-RNN fan club
"""

import math
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Prefer double precision for higher-order autograd stability
torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cpu")


# -----------------------------
# Forcing: differentiable u(t)
# -----------------------------
def forcing_torch(t: torch.Tensor) -> torch.Tensor:
    """
    Rich, smooth forcing as a function of t (PyTorch ops so autograd works):
        u(t) = 1.0*sin(1.0 t) + 0.7*cos(1.7 t) + 0.4*sin(2.3 t + 0.5)
    Shape: (N, 1) -> (N, 1)
    """
    return (
        torch.sin(1.0 * t) +
        0.7 * torch.cos(1.7 * t) +
        0.4 * torch.sin(2.3 * t + 0.5)
    )


def forcing_numpy(t: np.ndarray) -> np.ndarray:
    """NumPy version for simulation/plotting."""
    return (
        np.sin(1.0 * t) +
        0.7 * np.cos(1.7 * t) +
        0.4 * np.sin(2.3 * t + 0.5)
    )


# -----------------------------
# Ground-truth simulator (RK4)
# -----------------------------
def simulate_forced_msd_rk4(
    k: float, c: float,
    x0: float, v0: float,
    t_max: float, n_steps: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate:
        x'' = u(t) - c x' - k x
    via RK4 on a uniform grid.

    Returns:
      t: (N,) time
      x: (N,) position
      v: (N,) velocity
    """
    t = np.linspace(0.0, t_max, n_steps)
    dt = t[1] - t[0]
    x = np.zeros_like(t)
    v = np.zeros_like(t)
    x[0] = x0
    v[0] = v0

    def f_x(x_, v_, tt):
        # x' = v
        return v_

    def f_v(x_, v_, tt):
        # v' = u(t) - c v - k x
        return forcing_numpy(np.array([tt]))[0] - c * v_ - k * x_

    for i in range(len(t) - 1):
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
# Dataset
# -----------------------------
def make_dataset(
    n_data: int = 400,
    t_max: float = 10.0,
    noise_std: float = 0.01,
    seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Generate synthetic data from the true system using RK4, then add noise.
    """
    rng = np.random.default_rng(seed)

    # True physical parameters (unknown to the model)
    k_true = 2.0
    c_true = 0.5

    # Initial conditions
    x0_true = 0.8
    v0_true = -0.2

    # High-res sim; we will use the same grid as training points for simplicity
    t_np, x_np, v_np = simulate_forced_msd_rk4(
        k=k_true, c=c_true, x0=x0_true, v0=v0_true, t_max=t_max, n_steps=n_data
    )

    # Add noise to position measurements (common in experiments)
    x_noisy = x_np + rng.normal(0.0, noise_std, size=x_np.shape)

    # Torch tensors
    t = torch.tensor(t_np, device=DEVICE).reshape(-1, 1)
    x = torch.tensor(x_noisy, device=DEVICE).reshape(-1, 1)

    meta = dict(
        k_true=k_true, c_true=c_true,
        x0=x0_true, v0=v0_true,
        t_max=t_max, n_data=n_data,
        noise_std=noise_std
    )
    return t, x, meta


# -----------------------------
# Model: MLP and Extended PINN
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=1, hidden=128, depth=4, act=nn.Tanh):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden] * depth + [out_dim]
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), act()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.net(z)


class ExtendedPINN(nn.Module):
    """
    x_theta(t) = NN([t, u(t)])
    Learnable positive parameters k, c via Softplus.
    """
    def __init__(self, k_init: float = 1.0, c_init: float = 1.0):
        super().__init__()
        self.model = MLP(in_dim=2, out_dim=1, hidden=128, depth=4, act=nn.Tanh)
        # Unconstrained params -> positive via Softplus
        self.softplus = nn.Softplus()
        # initialize so softplus(k_u) ~ k_init
        self.k_unconstrained = nn.Parameter(
            torch.tensor([math.log(math.exp(k_init) - 1.0)], dtype=torch.get_default_dtype())
        )
        self.c_unconstrained = nn.Parameter(
            torch.tensor([math.log(math.exp(c_init) - 1.0)], dtype=torch.get_default_dtype())
        )

    @property
    def k(self):
        return self.softplus(self.k_unconstrained)

    @property
    def c(self):
        return self.softplus(self.c_unconstrained)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward prediction x(t) given time t.
        We compute u(t) inside to keep the graph intact for autograd through u(t).
        """
        u = forcing_torch(t)
        z = torch.cat([t, u], dim=1)  # (N, 2)
        return self.model(z)

    def physics_residual(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        r(t) = x_tt + c * x_t + k * x - u(t)
        Autograd wrt t handles chain rule through u(t).
        """
        t = t.clone().requires_grad_(True)
        x = self.forward(t)

        # First derivative
        x_t = torch.autograd.grad(
            x, t, grad_outputs=torch.ones_like(x), create_graph=True, retain_graph=True
        )[0]

        # Second derivative
        x_tt = torch.autograd.grad(
            x_t, t, grad_outputs=torch.ones_like(x_t), create_graph=True, retain_graph=True
        )[0]

        u = forcing_torch(t)
        r = x_tt + self.c * x_t + self.k * x - u
        return r, x, x_t


# -----------------------------
# Training
# -----------------------------
def train_extended_pinn(
    epochs: int = 6000,
    lr: float = 1e-3,
    n_colloc: int = 2000,
    weight_data: float = 1.0,
    weight_phys: float = 1.0,
    weight_ic: float = 5.0,
    seed: int = 0
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data
    t_data, x_data, meta = make_dataset(n_data=400, t_max=10.0, noise_std=0.01, seed=seed)
    x0, v0 = meta["x0"], meta["v0"]

    # Collocation points over [0, t_max]
    t_colloc = torch.rand(n_colloc, 1, device=DEVICE) * meta["t_max"]
    t_colloc.requires_grad_(True)

    # t=0 for ICs
    t0 = torch.zeros((1, 1), device=DEVICE, requires_grad=True)

    # Model + optim
    model = ExtendedPINN(k_init=1.0, c_init=1.0).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    mse = nn.MSELoss()

    for ep in range(1, epochs + 1):
        opt.zero_grad()

        # Data loss
        x_pred = model(t_data)
        loss_data = mse(x_pred, x_data)

        # Physics residual at collocation points
        r_colloc, _, _ = model.physics_residual(t_colloc)
        loss_phys = torch.mean(r_colloc ** 2)

        # Initial condition loss: x(0)=x0, x'(0)=v0
        r0, x0_pred, x0_t_pred = model.physics_residual(t0)
        loss_ic = torch.mean((x0_pred - x0) ** 2 + (x0_t_pred - v0) ** 2)

        # Total loss
        loss = weight_data * loss_data + weight_phys * loss_phys + weight_ic * loss_ic
        loss.backward()
        opt.step()
        sched.step()

        if ep % 500 == 0 or ep == 1:
            print(f"Epoch {ep:5d} | loss={loss.item():.4e} | "
                  f"data={loss_data.item():.4e} | phys={loss_phys.item():.4e} | ic={loss_ic.item():.4e} | "
                  f"k={model.k.item():.4f} | c={model.c.item():.4f}")

    # Optional LBFGS refinement (often helps PINNs). Uncomment to try:
    # def closure():
    #     opt_lbfgs.zero_grad()
    #     x_pred = model(t_data)
    #     loss_data = mse(x_pred, x_data)
    #     r_colloc, _, _ = model.physics_residual(t_colloc)
    #     loss_phys = torch.mean(r_colloc ** 2)
    #     r0, x0_pred, x0_t_pred = model.physics_residual(t0)
    #     loss_ic = torch.mean((x0_pred - x0) ** 2 + (x0_t_pred - v0) ** 2)
    #     loss = weight_data * loss_data + weight_phys * loss_phys + weight_ic * loss_ic
    #     loss.backward()
    #     return loss
    # opt_lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=200,
    #                               tolerance_grad=1e-8, tolerance_change=1e-12, line_search_fn="strong_wolfe")
    # opt_lbfgs.step(closure)

    return model, (t_data, x_data, meta)


# -----------------------------
# Evaluation / Visualization
# -----------------------------
def evaluate_and_plot(model: ExtendedPINN, dataset, save_fig: bool = True):
    model.eval()
    t_data, x_data, meta = dataset
    with torch.no_grad():
        x_pred = model(t_data)

    # Physics residual visualization
    t_vis = torch.linspace(0.0, meta["t_max"], 800, device=DEVICE).reshape(-1, 1).requires_grad_(True)
    r_vis, x_vis, _ = model.physics_residual(t_vis)

    # Ground truth (clean) via simulator with the same grid as t_data
    t_np = t_data.detach().cpu().numpy().squeeze()
    x_clean_np = simulate_forced_msd_rk4(
        k=meta["k_true"], c=meta["c_true"],
        x0=meta["x0"], v0=meta["v0"],
        t_max=meta["t_max"], n_steps=meta["n_data"]
    )[1]

    # Learned parameters
    k_hat = float(model.k.detach().cpu())
    c_hat = float(model.c.detach().cpu())

    # Metrics
    mse_noisy = torch.mean((x_pred - x_data) ** 2).item()
    mse_clean = np.mean((x_pred.detach().cpu().numpy().squeeze() - x_clean_np) ** 2)

    print("\n=== Evaluation ===")
    print(f"True params:    k = {meta['k_true']:.4f}, c = {meta['c_true']:.4f}")
    print(f"Learned params: k = {k_hat:.4f}, c = {c_hat:.4f}")
    print(f"MSE to noisy data:  {mse_noisy:.4e}")
    print(f"MSE to clean truth: {mse_clean:.4e}")

    # Plots
    plt.figure(figsize=(10, 6))
    plt.plot(t_np, x_clean_np, lw=2, label="Ground truth (clean)")
    plt.scatter(t_np, x_data.detach().cpu().numpy().squeeze(), s=10, alpha=0.5, label="Training data (noisy)")
    plt.plot(t_np, x_pred.detach().cpu().numpy().squeeze(), lw=2, ls="--", label="Extended PINN prediction")
    plt.title(f"Forced MSD (Extended PINN)\nLearned k={k_hat:.3f}, c={c_hat:.3f} | True k={meta['k_true']:.3f}, c={meta['c_true']:.3f}")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.grid(True)
    plt.legend()
    if save_fig:
        plt.tight_layout()
        plt.savefig("extended_pinn_forced_fit.png", dpi=160)
        print("Saved: extended_pinn_forced_fit.png")

    plt.figure(figsize=(10, 3.5))
    plt.plot(t_vis.detach().cpu().numpy().squeeze(),
             np.abs(r_vis.detach().cpu().numpy().squeeze()), lw=1.5)
    plt.title("Physics residual |x'' + c x' + k x - u(t)|")
    plt.xlabel("t")
    plt.ylabel("|residual|")
    plt.grid(True)
    if save_fig:
        plt.tight_layout()
        plt.savefig("extended_pinn_forced_residual.png", dpi=160)
        print("Saved: extended_pinn_forced_residual.png")

    # Optional: plot input u(t)
    plt.figure(figsize=(10, 3.5))
    u_np = forcing_numpy(t_np)
    plt.plot(t_np, u_np, lw=1.5)
    plt.title("Forcing input u(t)")
    plt.xlabel("t")
    plt.ylabel("u")
    plt.grid(True)
    if save_fig:
        plt.tight_layout()
        plt.savefig("extended_pinn_forced_input.png", dpi=160)
        print("Saved: extended_pinn_forced_input.png")

    plt.show()


# -----------------------------
# Main
# -----------------------------
def main():
    model, dataset = train_extended_pinn(
        epochs=6000,
        lr=1e-3,
        n_colloc=2000,
        weight_data=1.0,
        weight_phys=1.0,
        weight_ic=5.0,
        seed=42
    )
    evaluate_and_plot(model, dataset, save_fig=True)


if __name__ == "__main__":
    main()