from __future__ import annotations

import dataclasses
import numpy as np
from typing import Tuple, Optional

# ===========================
# Toggle mitigation strategies
# ===========================
USE_CI = True  # if True, use Covariance Intersection instead of vanilla Kalman


# ===========================
# Utility: Covariance Intersection (information form)
# ===========================

def ci_fuse(mean1: np.ndarray, cov1: np.ndarray,
            mean2: np.ndarray, cov2: np.ndarray,
            omega: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse two Gaussians N(mean1, cov1) and N(mean2, cov2) using Covariance Intersection.
    If omega is None, choose omega in [0,1] by a simple 1D line search minimizing trace of the fused covariance.
    Returns (fused_mean, fused_cov).
    """
    # Safeguards
    cov1 = 0.5 * (cov1 + cov1.T)
    cov2 = 0.5 * (cov2 + cov2.T)

    # Precompute information parameters
    Lam1 = np.linalg.inv(cov1)
    Lam2 = np.linalg.inv(cov2)
    eta1 = Lam1 @ mean1
    eta2 = Lam2 @ mean2

    if omega is None:
        # Coarse-but-robust 1D search over omega in [0,1]
        best_omega = 0.0
        best_trace = np.inf
        # Try a small grid; 0 and 1 are included
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
    state: np.ndarray  # true (unknown) state, for simulation/printing
    state_estimated: np.ndarray  # current estimate
    covariance: np.ndarray  # covariance of *local* (relative) state
    # Optional: unobservable global translation covariance that is *not* reduced by relative updates.
    # This represents the gauge (global shift); it keeps "absolute" covariance from collapsing without an anchor.

    def update_from_measured_state(self, measured_state: np.ndarray, measured_covariance: np.ndarray):
        """
        Update this agent's estimate using either standard Kalman fusion (your original)
        or Covariance Intersection (default here). The 'measured_state' is the parcel
        produced from a neighbor via a relative measurement.
        """
        if USE_CI:
            # CI between our current belief and the incoming parcel
            new_mean, new_cov = ci_fuse(self.state_estimated, self.covariance,
                                        measured_state, measured_covariance,
                                        omega=None)  # pick omega by trace-min
            self.state_estimated = new_mean
            self.covariance = new_cov
        else:
            # Original Kalman-style fusion (unsafe under unknown correlations / repeated info)
            K = self.covariance @ np.linalg.inv(self.covariance + measured_covariance)
            self.state_estimated = self.state_estimated + K @ (measured_state - self.state_estimated)
            self.covariance = (np.eye(self.covariance.shape[0]) - K) @ self.covariance


# ----------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Measurement:
    vector: np.ndarray
    agent_from: Agent
    agent_to: Agent
    covariance: np.ndarray


# ----------------------------------------------------------------------------------------------------------------------
def get_measured_state(agent: Agent, measurement: Measurement) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the 'parcel' for the target agent from a single relative measurement.
    Linear, 2D: z = x_to - x_from + v, with known z and R.
    The parcel includes propagation of the sender's local covariance (not the gauge offset).
    """
    if measurement.agent_from.id == agent.id:
        # Agent is the 'from' end; we need a measurement of our own state based on 'to'
        # x_from = x_to - z  => mean and covariance:
        state = measurement.agent_to.state_estimated - measurement.vector
        cov = measurement.agent_to.covariance + measurement.covariance

    elif measurement.agent_to.id == agent.id:
        # Agent is the 'to' end; parcel derived from 'from'
        # x_to = x_from + z
        state = measurement.agent_from.state_estimated + measurement.vector
        cov = measurement.agent_from.covariance + measurement.covariance
    else:
        raise ValueError("Invalid measurement")

    # NOTE: We intentionally do *not* add any gauge/offset covariance into the parcel,
    # because relative measurements don't inform global translation.
    return state, cov


def calculate_relative_measurement(agent_from: Agent, agent_to: Agent) -> np.ndarray:
    return agent_to.state - agent_from.state


# ----------------------------------------------------------------------------------------------------------------------
def example_1():
    dim = 2

    agent_a = Agent(
        id="a",
        state=np.array([1.0, 2.0]),
        state_estimated=np.array([0.0, 0.0]),
        covariance=np.array([[10.0, 0.0], [0.0, 10.0]]),
    )

    agent_b = Agent(
        id="b",
        state=np.array([3.0, 5.0]),
        state_estimated=np.array([10.0, 12.0]),
        covariance=np.array([[6.0, 0.0], [0.0, 10.0]]),
    )

    # Fixed measurement (global difference vector), noisy
    R = np.array([[0.01, 0.0], [0.0, 0.01]])
    measurement_a_2_b = Measurement(
        agent_from=agent_a,
        agent_to=agent_b,
        vector=calculate_relative_measurement(agent_from=agent_a, agent_to=agent_b),
        covariance=R,
    )

    for i in range(50):

        # if i == 25:
        #     agent_b.state_estimated = np.asarray([3.0, 5.0])
        #     agent_b.covariance = np.diag([5, 2])

        state_meas_a, cov_meas_a = get_measured_state(agent=agent_a, measurement=measurement_a_2_b)
        state_meas_b, cov_meas_b = get_measured_state(agent=agent_b, measurement=measurement_a_2_b)

        agent_a.update_from_measured_state(state_meas_a, cov_meas_a)
        agent_b.update_from_measured_state(state_meas_b, cov_meas_b)

        print(f"--------------- Step {i + 1} ---------------")
        print(f"Agent A: est {agent_a.state_estimated}, true {agent_a.state}")
        print(f"    Cov:\n{np.linalg.norm(agent_a.covariance, 'fro'):.1f}")
        print(f"Agent B: est {agent_b.state_estimated}, true {agent_b.state}")
        print(f"    Cov:\n{np.linalg.norm(agent_b.covariance, 'fro'):.1f}")


if __name__ == '__main__':
    example_1()
