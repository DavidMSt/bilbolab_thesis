import numpy as np
import math


def R(angle):
    """2D rotation matrix for a given angle (in radians)."""
    return np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)]
    ])


def h23(X):
    """
    Measurement function for agent2 observing agent3.

    X is a 9-dimensional vector composed of:
        X = [x1, y1, psi1, r21_x, r21_y, r21_psi, r31_x, r31_y, r31_psi]
    where:
        - (x1, y1, psi1) is the anchor (agent1) absolute state.
        - r21 = [r21_x, r21_y, r21_psi] is the state of agent2 relative to agent1.
        - r31 = [r31_x, r31_y, r31_psi] is the state of agent3 relative to agent1.

    The predicted measurement is the relative pose of agent3 as seen by agent2:
        z = [ R(-r21_psi) * ((r31_x, r31_y)^T - (r21_x, r21_y)^T); r31_psi - r21_psi ]
    """
    r21 = X[3:6]
    r31 = X[6:9]
    # Relative translation between agent3 and agent2 in the anchor's frame.
    delta_r = np.array([r31[0] - r21[0], r31[1] - r21[1]])
    # Rotate into agent2's body frame using its relative orientation.
    R_minus = R(-r21[2])
    meas_trans = R_minus @ delta_r
    meas_rot = r31[2] - r21[2]
    return np.hstack((meas_trans, meas_rot))


def finite_difference_jacobian(func, X, epsilon=1e-6):
    """
    Computes the Jacobian of a function 'func' at X using finite differences.

    Args:
        func: function mapping R^n -> R^m.
        X: state vector (n-dimensional).
        epsilon: small perturbation for finite differences.

    Returns:
        Jacobian matrix (m x n).
    """
    n = len(X)
    m = len(func(X))
    J = np.zeros((m, n))
    for i in range(n):
        dX = np.zeros(n)
        dX[i] = epsilon
        J[:, i] = (func(X + dX) - func(X - dX)) / (2 * epsilon)
    return J


def ekf_update(X, P, z, R_meas, h_func):
    """
    EKF measurement update.

    Args:
        X: state vector.
        P: state covariance matrix.
        z: measurement vector.
        R_meas: measurement noise covariance.
        h_func: measurement function.

    Returns:
        X_new: updated state vector.
        P_new: updated covariance matrix.
    """
    # Predicted measurement and innovation:
    z_pred = h_func(X)
    y = z - z_pred

    # Compute measurement Jacobian (via finite differences here):
    H = finite_difference_jacobian(h_func, X)

    # Innovation covariance and Kalman gain:
    S = H @ P @ H.T + R_meas
    K = P @ H.T @ np.linalg.inv(S)

    # Update state and covariance:
    X_new = X + K @ y
    P_new = (np.eye(len(X)) - K @ H) @ P
    return X_new, P_new


def main():
    # Define the "true" states (for simulation):
    # Agent 1 (anchor) absolute state:
    x1_true = np.array([0.0, 0.0, 0.0])  # [x, y, psi]

    # Absolute states for agents 2 and 3:
    x2_true = np.array([2.0, 1.0, 0.1])
    x3_true = np.array([3.0, -1.0, -0.2])

    # Convert to relative representation:
    # r_i1 = [ R(-psi1) * (p_i - p1); psi_i - psi1 ]
    r21_true = np.hstack((R(-x1_true[2]) @ (x2_true[0:2] - x1_true[0:2]), x2_true[2] - x1_true[2]))
    r31_true = np.hstack((R(-x1_true[2]) @ (x3_true[0:2] - x1_true[0:2]), x3_true[2] - x1_true[2]))

    # Full true state vector: [x1; r21; r31]
    X_true = np.hstack((x1_true, r21_true, r31_true))
    print("True state X_true:")
    print(X_true)

    # Initial estimate: agent1 is known perfectly, agents2 and 3 are uncertain.
    x1_est = x1_true.copy()
    # Introduce some initial error in the relative states.
    r21_est = r21_true + np.array([0.5, -0.3, 0.05])
    r31_est = r31_true + np.array([-0.4, 0.4, -0.03])
    X_est = np.hstack((x1_est, r21_est, r31_est))

    # Initial covariance: low for agent1; higher for agents2 and 3.
    P1 = np.diag([1e-6, 1e-6, 1e-6])
    P2 = np.diag([1.0, 1.0, 0.1])
    P3 = np.diag([1.0, 1.0, 0.1])
    P_est = np.block([
        [P1, np.zeros((3, 3)), np.zeros((3, 3))],
        [np.zeros((3, 3)), P2, np.zeros((3, 3))],
        [np.zeros((3, 3)), np.zeros((3, 3)), P3]
    ])

    # For simplicity assume a stationary process model (no dynamics).
    # So the predicted state equals the previous estimate.

    # Simulate a measurement: agent2 measures agent3.
    # Compute the "true" measurement from the true state.
    z_true = h23(X_true)
    # Add measurement noise (simulate sensor noise):
    meas_std = np.array([0.1, 0.1, 0.05])  # standard deviations for [trans_x, trans_y, rotation]
    noise = np.random.randn(3) * meas_std
    z_meas = z_true + noise
    print("\nSimulated measurement z_meas:")
    print(z_meas)

    # Measurement noise covariance:
    R_meas = np.diag(meas_std ** 2)

    # EKF update:
    X_upd, P_upd = ekf_update(X_est, P_est, z_meas, R_meas, h23)

    print("\nUpdated state X_upd:")
    print(X_upd)
    print("\nUpdated covariance P_upd:")
    print(P_upd)


if __name__ == "__main__":
    np.random.seed(42)
    main()
