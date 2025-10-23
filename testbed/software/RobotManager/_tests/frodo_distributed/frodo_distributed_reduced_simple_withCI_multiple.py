from __future__ import annotations

import dataclasses
from typing import Tuple, Optional, List, Dict

import numpy as np
import matplotlib.pyplot as plt


# ===========================
# Toggle mitigation strategies
# ===========================
USE_CI = True             # Use Covariance Intersection for fusion
USE_GAUGE_OFFSET = False   # Add a fixed gauge/offset covariance to represent unobservable global translation (when unanchored)
OFFSET_STD = 50.0         # std dev per axis for the global translation (only used if USE_GAUGE_OFFSET)

np.set_printoptions(precision=3, suppress=True)


# ===========================
# Utility: Covariance Intersection (information form)
# ===========================

def ci_fuse(mean1: np.ndarray, cov1: np.ndarray,
            mean2: np.ndarray, cov2: np.ndarray,
            omega: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse two Gaussians N(mean1, cov1) and N(mean2, cov2) using Covariance Intersection.
    If omega is None, choose omega in [0,1] by a simple 1D grid search minimizing trace of the fused covariance.
    Returns (fused_mean, fused_cov).
    """
    cov1 = 0.5 * (cov1 + cov1.T)
    cov2 = 0.5 * (cov2 + cov2.T)

    Lam1 = np.linalg.inv(cov1)
    Lam2 = np.linalg.inv(cov2)
    eta1 = Lam1 @ mean1
    eta2 = Lam2 @ mean2

    if omega is None:
        best_omega = 0.0
        best_trace = np.inf
        for w in np.linspace(0.0, 1.0, 51):
            Lam = w * Lam1 + (1.0 - w) * Lam2
            try:
                P = np.linalg.inv(Lam)
            except np.linalg.LinAlgError:
                continue
            tr = np.trace(P)
            if tr < best_trace:
                best_trace = tr
                best_omega = w
        omega = best_omega

    Lam = omega * Lam1 + (1.0 - omega) * Lam2
    P = np.linalg.inv(Lam)
    eta = omega * eta1 + (1.0 - omega) * eta2
    m = P @ eta
    return m, P


# ----------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Agent:
    id: str
    state_true: np.ndarray             # true (unknown to estimator)
    state_est: np.ndarray              # current estimate
    cov_local: np.ndarray              # covariance of *local* (relative) state
    is_anchor: bool = False            # anchor with perfectly known state (we'll simply freeze its estimate)
    cov_offset: Optional[np.ndarray] = None  # unobservable offset covariance (gauge); not reduced by relatives

    def absolute_covariance(self) -> np.ndarray:
        if USE_GAUGE_OFFSET and self.cov_offset is not None:
            return self.cov_local + self.cov_offset
        return self.cov_local

    def update_from_measured_state(self, measured_state: np.ndarray, measured_covariance: np.ndarray):
        if self.is_anchor:
            # Perfect anchor: do not update (it is already "ground truth")
            return
        if USE_CI:
            new_mean, new_cov = ci_fuse(self.state_est, self.cov_local,
                                        measured_state, measured_covariance,
                                        omega=None)
            self.state_est = new_mean
            self.cov_local = new_cov
        else:
            # Vanilla (unsafe) fusion
            K = self.cov_local @ np.linalg.inv(self.cov_local + measured_covariance)
            self.state_est = self.state_est + K @ (measured_state - self.state_est)
            I = np.eye(self.cov_local.shape[0])
            self.cov_local = (I - K) @ self.cov_local


# ----------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Measurement:
    agent_from: Agent
    agent_to: Agent
    covariance: np.ndarray             # measurement noise covariance R (2x2)
    # In this simplified linear setting, the measured vector is global difference: z = x_to - x_from + v

    def vector(self) -> np.ndarray:
        # Use *true* states to generate a measurement vector (noisy or noise-free)
        # For simulation, we can add a fresh noise sample each call if desired.
        return self.agent_to.state_true - self.agent_from.state_true


# ----------------------------------------------------------------------------------------------------------------------
def get_measured_state(agent: Agent, measurement: Measurement, z_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the 'parcel' for the target agent from a single relative measurement.
    Linear, 2D: z = x_to - x_from + v, with known z and R.
    The parcel includes propagation of the neighbor's local covariance (not the gauge offset).
    """
    if measurement.agent_from.id == agent.id:
        # Agent is the 'from' end; we infer our own state using the 'to' side
        # x_from = x_to - z
        state = measurement.agent_to.state_est - z_vec
        cov = measurement.agent_to.cov_local + measurement.covariance
    elif measurement.agent_to.id == agent.id:
        # Agent is the 'to' end; infer own state from 'from' side
        # x_to = x_from + z
        state = measurement.agent_from.state_est + z_vec
        cov = measurement.agent_from.cov_local + measurement.covariance
    else:
        raise ValueError("Invalid measurement agent pairing.")
    return state, cov


# ----------------------------------------------------------------------------------------------------------------------
def run_two_cluster_sim(T=80, t_connect=30, noisy=True, seed=7):
    rng = np.random.default_rng(seed)
    dim = 2

    # Gauge offset covariance (kept constant unless anchored)
    offset_cov = np.eye(dim) * (OFFSET_STD ** 2) if USE_GAUGE_OFFSET else None

    # -------- Cluster A: anchor + random agent --------
    anchor = Agent(
        id="A1",
        state_true=np.array([5.0, -2.0]),
        state_est=np.array([5.0, -2.0]),   # perfect knowledge
        cov_local=np.zeros((dim, dim)),    # exactly known
        is_anchor=True,
        cov_offset=np.zeros((dim, dim)) if USE_GAUGE_OFFSET else None,
    )

    A2_init = np.array([10.0, -15.0])  # arbitrary wrong initial guess
    agent_A2 = Agent(
        id="A2",
        state_true=np.array([7.5, 4.0]),
        state_est=A2_init.copy(),
        cov_local=np.eye(dim) * 100.0,    # very uncertain initially
        is_anchor=False,
        cov_offset=offset_cov.copy() if offset_cov is not None else None,
    )

    # Relative measurement inside Cluster A
    R_A = np.eye(dim) * 0.05  # cluster A measurement noise
    meas_A = Measurement(agent_from=anchor, agent_to=agent_A2, covariance=R_A)

    # -------- Cluster B: two random agents, no anchor --------
    B1_init = np.array([-12.0, 20.0])
    B2_init = np.array([-8.0, 18.0])

    agent_B1 = Agent(
        id="B1",
        state_true=np.array([0.0, 8.0]),
        state_est=B1_init.copy(),
        cov_local=np.eye(dim) * 80.0,
        is_anchor=False,
        cov_offset=offset_cov.copy() if offset_cov is not None else None,
    )
    agent_B2 = Agent(
        id="B2",
        state_true=np.array([3.0, 9.0]),
        state_est=B2_init.copy(),
        cov_local=np.eye(dim) * 80.0,
        is_anchor=False,
        cov_offset=offset_cov.copy() if offset_cov is not None else None,
    )

    # Relative measurement inside Cluster B
    R_B = np.eye(dim) * 0.05
    meas_B = Measurement(agent_from=agent_B1, agent_to=agent_B2, covariance=R_B)

    # -------- Bridge measurement to be added at t_connect --------
    # Connect A2 <-> B1 (arbitrary choice)
    R_bridge = np.eye(dim) * 0.05
    bridge = Measurement(agent_from=agent_A2, agent_to=agent_B1, covariance=R_bridge)

    # Storage for plotting
    agents = [anchor, agent_A2, agent_B1, agent_B2]
    names = [a.id for a in agents]

    est_hist: Dict[str, List[np.ndarray]] = {a.id: [] for a in agents}
    cov_hist: Dict[str, List[float]] = {a.id: [] for a in agents}

    # Simulation loop
    for t in range(T):
        # Build active measurement list
        active_meas = [meas_A, meas_B]
        if t >= t_connect:
            active_meas.append(bridge)

        # Optionally draw fresh noise for each measurement (independent samples)
        def measure_vec(m: Measurement):
            z = m.vector()
            if noisy:
                noise = rng.multivariate_normal(np.zeros(dim), m.covariance)
                z = z + noise
            return z

        # Perform symmetric updates for each active measurement
        for m in active_meas:
            z_vec = measure_vec(m)

            # Update "from" agent using parcel derived from "to"
            state_from, cov_from = get_measured_state(m.agent_from, m, z_vec)
            m.agent_from.update_from_measured_state(state_from, cov_from)

            # Update "to" agent using parcel derived from "from"
            state_to, cov_to = get_measured_state(m.agent_to, m, z_vec)
            m.agent_to.update_from_measured_state(state_to, cov_to)

        # Log
        for a in agents:
            est_hist[a.id].append(a.state_est.copy())
            cov_hist[a.id].append(np.trace(a.absolute_covariance()))

    # Convert logs to arrays
    for k in est_hist:
        est_hist[k] = np.vstack(est_hist[k])  # T x 2

    # Plot estimates (x and y) and covariance norms
    # Plot 1: x coordinate over time
    plt.figure()
    for a in agents:
        plt.plot(est_hist[a.id][:, 0], label=f"{a.id} x")
    plt.axvline(t_connect, linestyle="--")
    plt.title("Estimated x over time")
    plt.xlabel("timestep")
    plt.ylabel("x estimate")
    plt.legend()
    plt.show()

    # Plot 2: y coordinate over time
    plt.figure()
    for a in agents:
        plt.plot(est_hist[a.id][:, 1], label=f"{a.id} y")
    plt.axvline(t_connect, linestyle="--")
    plt.title("Estimated y over time")
    plt.xlabel("timestep")
    plt.ylabel("y estimate")
    plt.legend()
    plt.show()

    # Plot 3: trace of absolute covariance over time
    plt.figure()
    for a in agents:
        plt.plot(cov_hist[a.id], label=f"{a.id} tr(P_abs)")
    plt.axvline(t_connect, linestyle="--")
    plt.title("Trace of absolute covariance over time")
    plt.xlabel("timestep")
    plt.ylabel("trace(P_abs)")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    run_two_cluster_sim(T=80, t_connect=30, noisy=False, seed=7)