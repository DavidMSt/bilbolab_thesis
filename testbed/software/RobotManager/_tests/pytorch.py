#!/usr/bin/env python3
"""
PINN for an unactuated mass-spring-damper (m=1) system:
    x'' + c * x' + k * x = 0
The network takes time t and outputs x(t). We learn x(t) AND the system
parameters k and c jointly from data + physics residuals.

- Synthetic ground truth is generated from an underdamped closed-form solution.
- Loss = data loss + physics loss + initial condition loss.
- Optimizer: Adam, followed by optional LBFGS (commented; enable if desired).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Tuple

# Use double precision for better higher-order autograd stability
torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cpu")


# -----------------------------
# Utilities: ground truth signal
# -----------------------------
def underdamped_solution(t: np.ndarray, k: float, c: float,
                         x0: float, v0: float) -> np.ndarray:
    """
    Closed form for underdamped case (m=1):
      x(t) = e^{-c t / 2} ( A cos(omega_d t) + B sin(omega_d t) )
      with omega_d = sqrt(k - c^2/4), A = x0, B = (v0 + (c/2) * x0)/omega_d
    """
    assert k - (c**2) / 4.0 > 0.0, "Choose underdamped params: c < 2*sqrt(k)."
    omega_d = math.sqrt(k - (c**2) / 4.0)
    A = x0
    B = (v0 + 0.5 * c * x0) / omega_d
    exp_term = np.exp(-0.5 * c * t)
    return exp_term * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))


# -----------------------------
# PINN model
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, hidden=4*(32), depth=4, act=nn.Tanh):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden] * depth + [out_dim]
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i+1]), act()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, t):
        # expects t of shape (N, 1)
        return self.net(t)


class PINN(nn.Module):
    """
    PINN where k and c are learned Parameters.
    """
    def __init__(self, k_init: float = 1.0, c_init: float = 1.0):
        super().__init__()
        self.model = MLP(in_dim=1, out_dim=1, hidden=128, depth=4, act=nn.Tanh)
        # Learnable physical parameters (constrained to be positive via softplus)
        self.k_unconstrained = nn.Parameter(torch.tensor([math.log(math.exp(k_init) - 1.0)]))
        self.c_unconstrained = nn.Parameter(torch.tensor([math.log(math.exp(c_init) - 1.0)]))
        self.softplus = nn.Softplus()

    @property
    def k(self):
        return self.softplus(self.k_unconstrained)

    @property
    def c(self):
        return self.softplus(self.c_unconstrained)

    def forward(self, t):
        return self.model(t)

    def physics_residual(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute r(t) = x_tt + c * x_t + k * x at collocation points t.
        """
        t.requires_grad_(True)
        x = self.forward(t)                      # (N,1)
        x_t = torch.autograd.grad(
            x, t, grad_outputs=torch.ones_like(x), create_graph=True, retain_graph=True
        )[0]
        x_tt = torch.autograd.grad(
            x_t, t, grad_outputs=torch.ones_like(x_t), create_graph=True, retain_graph=True
        )[0]
        r = x_tt + self.c * x_t + self.k * x
        return r, x, x_t


# -----------------------------
# Data generation
# -----------------------------
def make_dataset(n_data: int = 200,
                 t_max: float = 10.0,
                 noise_std: float = 0.01,
                 seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Generate synthetic data from underdamped ground truth.
    """
    rng = np.random.default_rng(seed)
    k_true = 2.0
    c_true = 0.4
    x0_true = 1.0
    v0_true = 0.0

    t = np.linspace(0.0, t_max, n_data)
    x = underdamped_solution(t, k=k_true, c=c_true, x0=x0_true, v0=v0_true)
    if noise_std > 0.0:
        x_noisy = x + rng.normal(0.0, noise_std, size=x.shape)
    else:
        x_noisy = x

    t_torch = torch.tensor(t, dtype=torch.get_default_dtype()).reshape(-1, 1).to(DEVICE)
    x_torch = torch.tensor(x_noisy, dtype=torch.get_default_dtype()).reshape(-1, 1).to(DEVICE)

    meta = dict(k_true=k_true, c_true=c_true, x0=x0_true, v0=v0_true, t_max=t_max, n_data=n_data)
    return t_torch, x_torch, meta


# -----------------------------
# Training
# -----------------------------
def train_pinn(epochs: int = 6000,
               lr: float = 1e-3,
               n_colloc: int = 2000,
               weight_data: float = 1.0,
               weight_phys: float = 1.0,
               weight_ic: float = 5.0,
               seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data
    t_data, x_data, meta = make_dataset()
    x0, v0 = meta["x0"], meta["v0"]
    t0 = torch.zeros((1, 1), dtype=t_data.dtype, device=DEVICE, requires_grad=True)

    # Collocation points over [0, t_max]
    t_colloc = torch.rand(n_colloc, 1, dtype=t_data.dtype, device=DEVICE) * meta["t_max"]
    t_colloc.requires_grad_(True)

    # Model
    model = PINN(k_init=1.0, c_init=1.0).to(DEVICE)
    params = list(model.parameters())  # includes k,c since they are nn.Parameters on the module
    opt = torch.optim.Adam(params, lr=lr)
    # Optionally add a scheduler for a gentle cosine decay
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    mse = nn.MSELoss()

    losses_history = {"total": [], "data": [], "phys": [], "ic": []}

    for ep in range(1, epochs + 1):
        opt.zero_grad()

        # Data loss
        x_pred = model(t_data)
        loss_data = mse(x_pred, x_data)

        # Physics residual loss at collocation points
        r_colloc, _, _ = model.physics_residual(t_colloc)
        loss_phys = torch.mean(r_colloc**2)

        # Initial condition loss at t=0 (x(0)=x0, x_t(0)=v0)
        r0, x0_pred, x0_t_pred = model.physics_residual(t0)
        loss_ic = (x0_pred - x0)**2 + (x0_t_pred - v0)**2
        loss_ic = torch.mean(loss_ic)

        loss = weight_data * loss_data + weight_phys * loss_phys + weight_ic * loss_ic
        loss.backward()
        opt.step()
        sched.step()

        # Track
        if ep % 50 == 0 or ep == 1:
            losses_history["total"].append(float(loss.detach().cpu()))
            losses_history["data"].append(float(loss_data.detach().cpu()))
            losses_history["phys"].append(float(loss_phys.detach().cpu()))
            losses_history["ic"].append(float(loss_ic.detach().cpu()))

        if ep % 500 == 0 or ep == 1:
            print(f"Epoch {ep:5d} | loss={loss.item():.4e} | "
                  f"data={loss_data.item():.4e} | phys={loss_phys.item():.4e} | ic={loss_ic.item():.4e} | "
                  f"k={model.k.item():.4f} | c={model.c.item():.4f}")

    # Optionally refine with LBFGS (often helpful for PINNs). Uncomment to use.
    # def closure():
    #     opt_lbfgs.zero_grad()
    #     x_pred = model(t_data)
    #     loss_data = mse(x_pred, x_data)
    #     r_colloc, _, _ = model.physics_residual(t_colloc)
    #     loss_phys = torch.mean(r_colloc**2)
    #     r0, x0_pred, x0_t_pred = model.physics_residual(t0)
    #     loss_ic = torch.mean((x0_pred - x0)**2 + (x0_t_pred - v0)**2)
    #     loss = weight_data * loss_data + weight_phys * loss_phys + weight_ic * loss_ic
    #     loss.backward()
    #     return loss
    # opt_lbfgs = torch.optim.LBFGS(params, lr=1.0, max_iter=200, tolerance_grad=1e-8, tolerance_change=1e-12)
    # opt_lbfgs.step(closure)

    return model, (t_data, x_data, meta), losses_history


# -----------------------------
# Evaluation / Visualization
# -----------------------------
def evaluate_and_plot(model: PINN, dataset, save_fig: bool = True):
    model.eval()
    t_data, x_data, meta = dataset
    with torch.no_grad():
        x_pred = model(t_data)

    # Compute physics residual on a dense grid for visualization
    t_vis = torch.linspace(0.0, meta["t_max"], 800, dtype=t_data.dtype, device=DEVICE).reshape(-1, 1)
    t_vis.requires_grad_(True)
    r_vis, x_vis, _ = model.physics_residual(t_vis)
    r_vis = r_vis.detach().cpu().numpy()
    t_vis_np = t_vis.detach().cpu().numpy().squeeze()
    x_vis_np = x_vis.detach().cpu().numpy().squeeze()

    # Ground truth (noise-free) for comparison
    t_np = t_data.detach().cpu().numpy().squeeze()
    k_true, c_true = meta["k_true"], meta["c_true"]
    x0_true, v0_true = meta["x0"], meta["v0"]
    x_true_np = underdamped_solution(t_np, k_true, c_true, x0_true, v0_true)

    # Learned parameters
    k_hat = float(model.k.detach().cpu())
    c_hat = float(model.c.detach().cpu())

    # Metrics
    with torch.no_grad():
        mse_data = torch.mean((x_pred - x_data)**2).item()
        mse_to_true = np.mean((x_pred.detach().cpu().numpy().squeeze() - x_true_np)**2)

    print("\n=== Evaluation ===")
    print(f"True params:   k = {k_true:.4f}, c = {c_true:.4f}")
    print(f"Learned params: k = {k_hat:.4f}, c = {c_hat:.4f}")
    print(f"MSE to training data (noisy): {mse_data:.4e}")
    print(f"MSE to ground truth (clean):  {mse_to_true:.4e}")

    # Plots
    plt.figure(figsize=(10, 6))
    plt.plot(t_np, x_true_np, lw=2, label="Ground truth (clean)")
    plt.scatter(t_np, x_data.detach().cpu().numpy().squeeze(),
                s=12, alpha=0.5, label="Training data (noisy)")
    plt.plot(t_np, x_pred.detach().cpu().numpy().squeeze(),
             lw=2, linestyle="--", label="PINN prediction")
    plt.title(f"PINN fit for mass–spring–damper (m=1)\nLearned k={k_hat:.3f}, c={c_hat:.3f} | True k={k_true:.3f}, c={c_true:.3f}")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend()
    plt.grid(True)
    if save_fig:
        plt.tight_layout()
        plt.savefig("pinn_msd_fit.png", dpi=160)
        print("Saved figure: pinn_msd_fit.png")

    plt.figure(figsize=(10, 3.5))
    plt.plot(t_vis_np, np.abs(r_vis).squeeze(), lw=1.5)
    plt.title("Physics residual |x'' + c x' + k x| over time (PINN)")
    plt.xlabel("t")
    plt.ylabel("|residual|")
    plt.grid(True)
    if save_fig:
        plt.tight_layout()
        plt.savefig("pinn_msd_residual.png", dpi=160)
        print("Saved figure: pinn_msd_residual.png")

    plt.show()


# -----------------------------
# Main
# -----------------------------
def main():
    model, dataset, _ = train_pinn(
        epochs=6000,          # increase/decrease as needed
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