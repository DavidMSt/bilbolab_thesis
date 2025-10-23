import dataclasses
from typing import Optional, Tuple

import numpy as np

USE_CI = True
USE_FEJ = False  # freeze psi_from the first time a measurement edge is used

INDEX_X = 0
INDEX_Y = 1
INDEX_PSI = 2


def wrap_angle(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def align_angle(ref: float, a: float) -> float:
    """Shift a to the nearest 2π branch relative to ref."""
    d = wrap_angle(a - ref)
    return ref + d


def ci_fuse(mean1: np.ndarray, cov1: np.ndarray,
            mean2: np.ndarray, cov2: np.ndarray,
            omega: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Covariance Intersection in information form.
    Before forming information, we assume angle alignment has been done by caller.
    """
    cov1 = 0.5 * (cov1 + cov1.T)
    cov2 = 0.5 * (cov2 + cov2.T)

    Lam1 = np.linalg.inv(cov1)
    Lam2 = np.linalg.inv(cov2)
    eta1 = Lam1 @ mean1
    eta2 = Lam2 @ mean2

    if omega is None:
        best_w, best_tr = 0.0, np.inf
        for w in np.linspace(0.0, 1.0, 51):
            Lam = w * Lam1 + (1.0 - w) * Lam2
            try:
                P = np.linalg.inv(Lam)
            except np.linalg.LinAlgError:
                continue
            tr = np.trace(P)
            if tr < best_tr:
                best_tr, best_w = tr, w
        omega = best_w

    Lam = omega * Lam1 + (1.0 - omega) * Lam2
    P = np.linalg.inv(Lam)
    eta = omega * eta1 + (1.0 - omega) * eta2
    m = P @ eta
    return m, P


@dataclasses.dataclass
class Agent:
    id: str
    state: np.ndarray  # true state [x, y, psi] (for simulation only)
    state_estimated: np.ndarray  # estimate [x, y, psi]
    covariance: np.ndarray  # 3x3 covariance

    def update_from_measured_state(self, measured_state: np.ndarray, measured_covariance: np.ndarray):
        # Align the incoming angle to our angle branch before fusion
        measured_state = measured_state.copy()
        measured_state[INDEX_PSI] = align_angle(self.state_estimated[INDEX_PSI], measured_state[INDEX_PSI])

        if USE_CI:
            new_mean, new_cov = ci_fuse(self.state_estimated, self.covariance,
                                        measured_state, measured_covariance,
                                        omega=None)
            # Wrap the resulting angle
            new_mean[INDEX_PSI] = wrap_angle(new_mean[INDEX_PSI])
            self.state_estimated = new_mean
            self.covariance = new_cov
        else:
            # (Unsafe under loops) – EKF-style linear fusion
            K = self.covariance @ np.linalg.inv(self.covariance + measured_covariance)
            innov = measured_state - self.state_estimated
            innov[INDEX_PSI] = wrap_angle(innov[INDEX_PSI])
            self.state_estimated = self.state_estimated + K @ innov
            self.state_estimated[INDEX_PSI] = wrap_angle(self.state_estimated[INDEX_PSI])
            self.covariance = (np.eye(3) - K) @ self.covariance


@dataclasses.dataclass
class Measurement:
    """
    Relative pose measurement z_{i->j} in i's local frame:
      z = R(psi_i) * ([x_j - x_i, y_j - y_i, psi_j - psi_i])
    For simulation we precompute vector from *true* states, but the estimator must
    rotate with *estimated* (or FEJ-frozen) psi_i.
    """
    vector: np.ndarray  # 3x1 local measurement (from simulator)
    agent_from: Agent
    agent_to: Agent
    covariance: np.ndarray  # 3x3 measurement covariance in i's local frame
    psi_from_frozen: Optional[float] = None  # FEJ: freeze psi_from at first use


def get_rotation_matrix(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    R = np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])
    return R


def R_to_local(psi: float) -> np.ndarray:
    return get_rotation_matrix(-psi)


def R_to_global(psi: float) -> np.ndarray:
    return get_rotation_matrix(psi)


def calculate_measurement(agent_from: Agent, agent_to: Agent) -> np.ndarray:
    """Simulator-side: build local measurement from true states."""
    dx = agent_to.state[INDEX_X] - agent_from.state[INDEX_X]
    dy = agent_to.state[INDEX_Y] - agent_from.state[INDEX_Y]
    dpsi = wrap_angle(agent_to.state[INDEX_PSI] - agent_from.state[INDEX_PSI])
    measurement_global = np.array([dx, dy, dpsi])
    measurement_local = R_to_local(agent_from.state[INDEX_PSI]) @ measurement_global
    return measurement_local


def dR_dpsi_times_vec(psi: float, z: np.ndarray) -> np.ndarray:
    """
    Compute (d/dpsi) [ R(psi) * z ] for 3x1 z = [z_x, z_y, z_psi].
    Only x,y depend on psi; psi row is zero. For 2x2 R(psi) = [[c,-s],[s,c]]:
      dR/dpsi = [[-s, -c],
                 [ c, -s]]
    """
    c, s = np.cos(psi), np.sin(psi)
    J2 = np.array([[-s, -c],
                   [c, -s]])
    xy = J2 @ z[:2]
    out = np.array([xy[0], xy[1], 0.0])
    return out


def get_measured_state(agent: Agent, measurement: Measurement) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the parcel for 'agent' from a single relative measurement on edge (from -> to).
    We rotate the local measurement with *estimated* (or FEJ-frozen) psi_from, and
    we account for the uncertainty of that heading in the covariance via J_psi term.
    """
    agent_from = measurement.agent_from
    agent_to = measurement.agent_to

    # FEJ: freeze psi_from the first time this edge is used
    if USE_FEJ:
        if measurement.psi_from_frozen is None:
            measurement.psi_from_frozen = float(agent_from.state_estimated[INDEX_PSI])
        psi_ref = float(measurement.psi_from_frozen)
    else:
        psi_ref = float(agent_from.state_estimated[INDEX_PSI])

    # Rotate the local measurement into the global frame using psi_ref
    Rg = R_to_global(psi_ref)
    z_global = Rg @ measurement.vector  # 3x1

    # Rotate measurement covariance into global
    Rz = Rg @ measurement.covariance @ Rg.T

    # Add contribution from uncertain psi_ref used in rotation
    var_psi_from = float(agent_from.covariance[INDEX_PSI, INDEX_PSI])
    Jpsi = dR_dpsi_times_vec(psi_ref, measurement.vector).reshape(3, 1)  # 3x1
    Cpsi = (Jpsi @ Jpsi.T) * var_psi_from

    if agent is agent_from:
        # x_from = x_to - z_global  (angles wrap)
        # Mean:
        new_state = agent_to.state_estimated - z_global
        new_state[INDEX_PSI] = wrap_angle(new_state[INDEX_PSI])
        # Covariance:
        new_cov = agent_to.covariance + Rz + Cpsi
        # new_cov = agent_to.covariance + Rz

    elif agent is agent_to:
        # x_to = x_from + z_global
        new_state = agent_from.state_estimated + z_global
        new_state[INDEX_PSI] = wrap_angle(new_state[INDEX_PSI])
        new_cov = agent_from.covariance + Rz + Cpsi
        # new_cov = agent_from.covariance + Rz
    else:
        raise ValueError("Invalid measurement / agent pairing")

    return new_state, new_cov


# ----------------- quick check -----------------
if __name__ == '__main__':
    agent_a = Agent(
        id="A",
        state=np.array([1.0, 2.0, 0]),
        state_estimated=np.array([0.0, 0.0, 0.5]),
        covariance=np.diag([10.0, 10.0, 10.0])
    )

    agent_b = Agent(
        id="B",
        state=np.array([2, 5, 1]),
        state_estimated=np.array([0, 0, 0]),  # almost -pi branch to stress angle alignment
        covariance=100.0 * np.eye(3)
    )

    agent_c = Agent(
        id="C",
        state=np.array([-1, -2, 0.5]),
        state_estimated=np.array([-1, -2, 0.5]),
        covariance=0.01 * np.eye(3)
    )

    # Sim measurement in A's local frame
    meas_a_b = Measurement(
        agent_from=agent_a,
        agent_to=agent_b,
        vector=calculate_measurement(agent_from=agent_a, agent_to=agent_b),
        # covariance=np.diag([0.04, 0.04, 0.01])  # e.g., good xy, decent yaw
        covariance=np.diag([0, 0, 0])  # e.g., good xy, decent yaw
    )

    meas_c_a = Measurement(
        agent_from=agent_c,
        agent_to=agent_a,
        vector=calculate_measurement(agent_from=agent_c, agent_to=agent_a),
        covariance=np.diag([0.01, 0.01, 0.01])
    )

    for i in range(100):
        state_meas_a, cov_meas_a = get_measured_state(agent=agent_a, measurement=meas_a_b)
        state_meas_b, cov_meas_b = get_measured_state(agent=agent_b, measurement=meas_a_b)

        agent_a.update_from_measured_state(state_meas_a, cov_meas_a)
        agent_b.update_from_measured_state(state_meas_b, cov_meas_b)

        print(f"--- step {i + 1} ---")
        print(
            f"A est {agent_a.state_estimated}, true {agent_a.state}, ||P||_F={np.linalg.norm(agent_a.covariance, 'fro'):.2f}")
        print(
            f"B est {agent_b.state_estimated}, true {agent_b.state}, ||P||_F={np.linalg.norm(agent_b.covariance, 'fro'):.2f}")



    b_est_in_a_coordinates = R_to_local(agent_a.state_estimated[INDEX_PSI]) @ agent_b.state_estimated
    print(f"B est in A coordinates: {b_est_in_a_coordinates}")