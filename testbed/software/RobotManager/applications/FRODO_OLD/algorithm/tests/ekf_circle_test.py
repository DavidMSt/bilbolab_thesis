import numpy as np
import math

import qmt


def R(angle):
    """2D rotation matrix for a given angle (in radians)."""
    return np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)]
    ])


def h_12(X):
    """
    Measurement from Agent 1 observing Agent 2.
    Since Agent 1 is the anchor, its measurement is directly the relative state of Agent 2.
    X = [x1, y1, psi1, r21_x, r21_y, r21_psi, r31_x, r31_y, r31_psi]
    """
    return X[3:6]


def h_23(X):
    """
    Measurement from Agent 2 observing Agent 3.
    Agent 2's state is given by r21 = [r21_x, r21_y, r21_psi] and Agent 3's relative state is r31.
    The predicted measurement is:
        z_23 = [ R(-r21_psi) * ((r31_x, r31_y)^T - (r21_x, r21_y)^T);
                  r31_psi - r21_psi ]
    """
    r21 = X[3:6]
    r31 = X[6:9]
    delta = np.array([r31[0] - r21[0], r31[1] - r21[1]])
    R_minus = R(-r21[2])
    meas_trans = R_minus @ delta

    # meas_rot = r31[2] - r21[2]

    meas_rot = qmt.wrapToPi(r31[2] - r21[2])
    return np.hstack((meas_trans, meas_rot))


def h_31(X):
    """
    Measurement from Agent 3 observing Agent 1.
    The relative state r31 = [r31_x, r31_y, r31_psi] tells how Agent 3 is positioned relative to Agent 1.
    Its inverse (i.e. how Agent 1 appears from Agent 3) is:
         r_13 = -[ R(-r31_psi)* (r31_x, r31_y)^T; r31_psi ]
    """
    r31 = X[6:9]
    t = r31[0:2]
    # theta = r31[2]
    theta = qmt.wrapToPi(r31[2])
    t_inv = - R(-theta) @ t
    return np.hstack((t_inv, -theta))


def h_all(X):
    """
    Stacked measurement function.
    Returns a 9-dimensional vector:
      [h_12(X); h_23(X); h_31(X)]
    """
    # return np.hstack((h_12(X), h_23(X), h_31(X)))
    return np.hstack((h_12(X), h_23(X), h_31(X)))


def finite_difference_jacobian(func, X, epsilon=1e-6):
    """
    Computes the Jacobian of a function 'func' at X using finite differences.
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
    Standard EKF update step.
    """
    z_pred = h_func(X)
    y = z - z_pred
    H = finite_difference_jacobian(h_func, X)
    S = H @ P @ H.T + R_meas
    K = P @ H.T @ np.linalg.inv(S)
    X_new = X + K @ y
    P_new = (np.eye(len(X)) - K @ H) @ P
    return X_new, P_new


def main():
    # Define true absolute states for the three agents.
    # Agent 1 (anchor)
    x1_true = np.array([0.0, 0.0, 0.0])
    # Agent 2 absolute state:
    x2_true = np.array([0.5, 0.5, math.radians(90 + 45)])
    # Agent 3 absolute state:
    x3_true = np.array([0, 1, math.radians(-90)])

    # Compute relative states with respect to Agent 1:
    # r_i1 = [ R(-psi1) * (p_i - p1); psi_i - psi1 ]
    r21_true = np.hstack((R(-x1_true[2]) @ (x2_true[0:2] - x1_true[0:2]), x2_true[2] - x1_true[2]))
    r31_true = np.hstack((R(-x1_true[2]) @ (x3_true[0:2] - x1_true[0:2]), x3_true[2] - x1_true[2]))

    # Full true state vector
    X_true = np.hstack((x1_true, r21_true, r31_true))

    # Initial estimate:
    # Assume Agent 1 is known perfectly.
    x1_est = x1_true.copy()
    # Introduce some error in the relative states.
    # r21_est = r21_true + np.array([0.5, -0.3, 0.05])
    # r31_est = r31_true + np.array([-0.4, 0.4, -0.03])

    r21_est = np.array([0, 0, 0])
    r31_est = np.array([0, 0, 0])
    X_est = np.hstack((x1_est, r21_est, r31_est))

    # Initial covariance:
    P1 = np.diag([1e-6, 1e-6, 1e-6])
    P2 = np.diag([1.0, 1.0, 0.1])
    P3 = np.diag([1.0, 1.0, 0.1])
    P_est = np.block([
        [P1, np.zeros((3, 3)), np.zeros((3, 3))],
        [np.zeros((3, 3)), P2, np.zeros((3, 3))],
        [np.zeros((3, 3)), np.zeros((3, 3)), P3]
    ])

    # Measurement noise (for each of the 3 measurements, 3 elements each)
    # sigma12 = np.array([0.1, 0.1, 0.05])
    # sigma23 = np.array([0.1, 0.1, 0.05])
    # sigma31 = np.array([0.1, 0.1, 0.05])

    sigma12 = np.array([1e-4, 1e-4, 1e-4])
    sigma23 = np.array([1e-4, 1e-4, 1e-4])
    sigma31 = np.array([1e-4, 1e-4, 1e-4])
    R12 = np.diag(sigma12 ** 2)
    R23 = np.diag(sigma23 ** 2)
    R31 = np.diag(sigma31 ** 2)
    R_meas = np.block([
        [R12, np.zeros((3, 3)), np.zeros((3, 3))],
        [np.zeros((3, 3)), R23, np.zeros((3, 3))],
        [np.zeros((3, 3)), np.zeros((3, 3)), R31]
    ])

    num_steps = 100
    print("Step | Estimated State and Frobenius norm of Covariance")
    for step in range(num_steps):
        # Simulate measurements based on the true state.
        z_true = h_all(X_true)
        noise = np.hstack((np.random.randn(3) * sigma12,
                           np.random.randn(3) * sigma23,
                           np.random.randn(3) * sigma31))
        # z_meas = z_true + noise
        z_meas = z_true


        # Perform the EKF update.
        X_est, P_est = ekf_update(X_est, P_est, z_meas, R_meas, h_all)

        # Compute the Frobenius norm of the covariance.
        frob_norm = np.linalg.norm(P_est, 'fro')
        # Print in a human-readable format.
        # We'll print the state components with labels:
        s = ("Step {:2d}:\n"
             "  Agent1 (anchor): x={:6.3f}, y={:6.3f}, psi={:6.3f}\n"
             "  r21 (Agent2 relative): x={:6.3f}, y={:6.3f}, dpsi={:6.3f}\n"
             "  r31 (Agent3 relative): x={:6.3f}, y={:6.3f}, dpsi={:6.3f}\n"
             "  Frobenius norm of Covariance: {:8.4f}\n").format(
            step + 1,
            X_est[0], X_est[1], X_est[2],
            X_est[3], X_est[4], X_est[5],
            X_est[6], X_est[7], X_est[8],
            frob_norm)
        print(s)


if __name__ == "__main__":
    np.random.seed(42)
    main()
