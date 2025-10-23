import numpy as np
import math


def R(angle):
    """2D rotation matrix for a given angle (in radians)."""
    return np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)]
    ])


# ---------- Group 1 Measurement Functions ----------

def h_12(X):
    """
    Measurement from Agent 1 observing Agent 2.
    For Group 1, returns the relative state r21 stored in X[3:6].
    """
    return X[3:6]


def h_23(X):
    """
    Measurement from Agent 2 observing Agent 3.
    Uses group 1's relative states:
        r21 = X[3:6] and r31 = X[6:9].
    Returns:
       z_23 = [ R(-r21_psi) * ((r31_x, r31_y)^T - (r21_x, r21_y)^T); r31_psi - r21_psi ]
    """
    r21 = X[3:6]
    r31 = X[6:9]
    delta = np.array([r31[0] - r21[0], r31[1] - r21[1]])
    R_minus = R(-r21[2])
    meas_trans = R_minus @ delta
    meas_rot = r31[2] - r21[2]
    return np.hstack((meas_trans, meas_rot))


def h_31(X):
    """
    Measurement from Agent 3 observing Agent 1.
    Uses group 1's relative state r31 (X[6:9]).
    Returns the inverse of r31:
         z_31 = -[ R(-r31_psi)*(r31_xy); r31_psi ]
    """
    r31 = X[6:9]
    t = r31[0:2]
    theta = r31[2]
    t_inv = - R(-theta) @ t
    return np.hstack((t_inv, -theta))


# ---------- Group 2 Measurement Functions ----------

def h_45(X):
    """
    Measurement from Agent 4 observing Agent 5.
    For Group 2, Agent 4 is the anchor and r54 is stored in X[12:15].
    """
    return X[12:15]


def h_54(X):
    """
    Measurement from Agent 5 observing Agent 4.
    Returns the inverse of r54 (stored in X[12:15]):
         z_54 = -[ R(-r54_psi)*(r54_xy); r54_psi ]
    """
    r54 = X[12:15]
    t = r54[0:2]
    theta = r54[2]
    t_inv = - R(-theta) @ t
    return np.hstack((t_inv, -theta))


def h_all(X):
    """
    Stacked measurement function.
    Returns a 15-dimensional vector combining:
      Group 1: h_12 (3), h_23 (3), h_31 (3)  -> 9 elements
      Group 2: h_45 (3), h_54 (3)              -> 6 elements
    Total = 15 elements.
    """
    return np.hstack((h_12(X), h_23(X), h_31(X), h_45(X), h_54(X)))


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
    # ------------------- Group 1 True States -------------------
    # Agent 1 (anchor) absolute state
    x1_true = np.array([0.0, 0.0, 0.0])
    # Agent 2 absolute state
    x2_true = np.array([2.0, 1.0, 0.1])
    # Agent 3 absolute state
    x3_true = np.array([3.0, -1.0, -0.2])
    # Compute relative states: r_i1 = [ R(-psi1)*(p_i-p1); psi_i-psi1 ]
    r21_true = np.hstack((R(-x1_true[2]) @ (x2_true[0:2] - x1_true[0:2]), x2_true[2] - x1_true[2]))
    r31_true = np.hstack((R(-x1_true[2]) @ (x3_true[0:2] - x1_true[0:2]), x3_true[2] - x1_true[2]))
    # Group 1 true state vector (9 elements)
    X_true_g1 = np.hstack((x1_true, r21_true, r31_true))

    # ------------------- Group 2 True States -------------------
    # Agent 4 (anchor) absolute state; note: we want this anchor to have high covariance.
    # x4_true = np.array([5.0, 5.0, 0.2])
    x4_true = np.array([0,0,0])
    # Agent 5 absolute state
    x5_true = np.array([1,  1 , 0.1])
    # Relative state for group 2: r54 = [ R(-psi4)*(p5-p4); psi5-psi4 ]
    r54_true = np.hstack((R(-x4_true[2]) @ (x5_true[0:2] - x4_true[0:2]), x5_true[2] - x4_true[2]))
    # Group 2 true state vector (6 elements)
    X_true_g2 = np.hstack((x4_true, r54_true))

    # ------------------- Full True State -------------------
    # Full state is [Group1 (9-dim); Group2 (6-dim)] = 15-dim vector
    X_true = np.hstack((X_true_g1, X_true_g2))

    # ------------------- Initial Estimate -------------------
    # Group 1: Agent 1 is known perfectly; add error to relative states.
    x1_est = x1_true.copy()
    r21_est = r21_true + np.array([0.5, -0.3, 0.05])
    r31_est = r31_true + np.array([-0.4, 0.4, -0.03])
    X_est_g1 = np.hstack((x1_est, r21_est, r31_est))

    # Group 2: Agent 4 (anchor) with high uncertainty; add error to both.
    x4_est = x4_true + np.array([8,8,8])
    r54_est = r54_true + np.array([-0.4, 0.4, -0.03])
    X_est_g2 = np.hstack((x4_est, r54_est))

    # Full estimated state (15-dim)
    X_est = np.hstack((X_est_g1, X_est_g2))

    # ------------------- Initial Covariance -------------------
    # Group 1: Agent 1 (anchor) low covariance; Agents 2 and 3 higher uncertainty.
    P1 = np.diag([1e-6, 1e-6, 1e-6])
    P2 = np.diag([1.0, 1.0, 0.1])
    P3 = np.diag([1.0, 1.0, 0.1])
    P_g1 = np.block([
        [P1, np.zeros((3, 3)), np.zeros((3, 3))],
        [np.zeros((3, 3)), P2, np.zeros((3, 3))],
        [np.zeros((3, 3)), np.zeros((3, 3)), P3]
    ])
    # Group 2: Agent 4 (anchor) has high covariance; Agent 5 relative state with lower uncertainty.
    P4 = np.diag([1e-6*1e8, 1e-6*1e8, 1e-6*1e8])  # high covariance for the anchor of group 2
    P5 = np.diag([1.0*1e8, 1.0*1e8, 0.1*1e8])
    P_g2 = np.block([
        [P4, np.zeros((3, 3))],
        [np.zeros((3, 3)), P5]
    ])
    # Full covariance (15x15)
    P_est = np.block([
        [P_g1, np.zeros((9, 6))],
        [np.zeros((6, 9)), P_g2]
    ])

    # ------------------- Measurement Noise Covariance -------------------
    # Group 1 measurement noise: three measurements, each 3-dim.
    sigma12 = np.array([0.1, 0.1, 0.05])
    sigma23 = np.array([0.1, 0.1, 0.05])
    sigma31 = np.array([0.1, 0.1, 0.05])
    R12 = np.diag(sigma12 ** 2)
    R23 = np.diag(sigma23 ** 2)
    R31 = np.diag(sigma31 ** 2)
    # Group 2 measurement noise: two measurements, each 3-dim.
    sigma45 = np.array([0.1, 0.1, 0.05])
    sigma54 = np.array([0.1, 0.1, 0.05])
    # sigma45 = np.array([1e-5, 1e-5, 1e-5])
    # sigma54 = np.array([1e-5, 1e-5, 1e-5])
    R45 = np.diag(sigma45 ** 2)
    R54 = np.diag(sigma54 ** 2)
    # Full measurement noise covariance (15x15)
    R_meas = np.block([
        [R12, np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 6))],
        [np.zeros((3, 3)), R23, np.zeros((3, 3)), np.zeros((3, 6))],
        [np.zeros((3, 3)), np.zeros((3, 3)), R31, np.zeros((3, 6))],
        [np.zeros((6, 3)), np.zeros((6, 3)), np.zeros((6, 3)), np.block([
            [R45, np.zeros((3, 3))],
            [np.zeros((3, 3)), R54]
        ])]
    ])
    # Alternatively, since dimensions are small, we can construct R_meas by stacking:
    R_meas = np.block([
        [np.diag(np.hstack((sigma12 ** 2, sigma23 ** 2, sigma31 ** 2, sigma45 ** 2, sigma54 ** 2)))]
    ])
    # However, here we will simply build it as a block-diagonal matrix:
    R_meas = np.diag(np.hstack((sigma12 ** 2, sigma23 ** 2, sigma31 ** 2, sigma45 ** 2, sigma54 ** 2)))

    num_steps = 100
    print("Step | Estimated States and Frobenius norm of Covariance")
    for step in range(num_steps):
        # Simulate measurements based on the true full state X_true.
        z_true = h_all(X_true)
        # Create noise for each measurement component.
        noise = np.hstack((
            np.random.randn(3) * sigma12,
            np.random.randn(3) * sigma23,
            np.random.randn(3) * sigma31,
            np.random.randn(3) * sigma45,
            np.random.randn(3) * sigma54
        ))
        z_meas = z_true + noise

        # EKF update:
        X_est, P_est = ekf_update(X_est, P_est, z_meas, R_meas, h_all)

        # Compute Frobenius norm of full covariance.
        frob_norm = np.linalg.norm(P_est, 'fro')
        # Extract group1 and group2 estimated states for readability.
        # Group1:
        x1_est = X_est[0:3]
        r21_est = X_est[3:6]
        r31_est = X_est[6:9]
        # Group2:
        x4_est = X_est[9:12]
        r54_est = X_est[12:15]

        # Print the results.
        print(f"Step {step + 1:2d}:")
        print(f"  Group1:")
        print(f"    Agent1 (anchor): x={x1_est[0]:6.3f}, y={x1_est[1]:6.3f}, psi={x1_est[2]:6.3f}")
        print(f"    r21 (Agent2 relative): x={r21_est[0]:6.3f}, y={r21_est[1]:6.3f}, dpsi={r21_est[2]:6.3f}")
        print(f"    r31 (Agent3 relative): x={r31_est[0]:6.3f}, y={r31_est[1]:6.3f}, dpsi={r31_est[2]:6.3f}")
        print(f"  Group2:")
        print(f"    Agent4 (anchor): x={x4_est[0]:6.3f}, y={x4_est[1]:6.3f}, psi={x4_est[2]:6.3f}")
        print(f"    r54 (Agent5 relative): x={r54_est[0]:6.3f}, y={r54_est[1]:6.3f}, dpsi={r54_est[2]:6.3f}")
        print(f"  Frobenius norm of full Covariance: {frob_norm:8.4f}\n")
    pass

if __name__ == "__main__":
    np.random.seed(42)
    main()
