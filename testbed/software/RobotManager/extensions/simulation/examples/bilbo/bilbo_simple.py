import numpy as np
from matplotlib import pyplot as plt

from core.utils.control.lib_control.il.ilc import BILBO_STANDARD_REFERENCE_TRAJECTORY, getTransitionMatrixFromSystem, \
    getLearningMatricesOptimal
from core.utils.control.lib_control.il.iml import imlUpdateOptimal, getOptimalLearningMatrix, \
    getOptimalLearningMatrixFromMatrix
from core.utils.control.lib_control.lifted_systems import vec2liftedMatrix, liftedMatrix2Vec
from core.utils.data import generate_time_vector, generate_random_input, resample
from extensions.simulation.src.objects.bilbo import BILBO_Dynamics_2D, DEFAULT_BILBO_MODEL, \
    BILBO_Dynamics_2D_Linear, BILBO_2D_POLES, BILBO_MICHAEL_MODEL


def main():
    N = 500
    # Generate the robot
    bilbo_dynamics = BILBO_Dynamics_2D(model=DEFAULT_BILBO_MODEL, Ts=0.01)
    bilbo_dynamics.polePlacement(poles=BILBO_2D_POLES, apply_poles_to_system=True)

    bilbo_dynamics_linear = BILBO_Dynamics_2D_Linear(model=DEFAULT_BILBO_MODEL, Ts=0.01)
    bilbo_dynamics_linear.polePlacement(poles=BILBO_2D_POLES, apply_poles_to_system=True)

    u = -1 * np.ones(N)

    states = bilbo_dynamics.simulate(u)
    states_linear = bilbo_dynamics_linear.simulate(u)

    theta = [state.theta for state in states]
    theta_linear = [state.theta for state in states_linear]

    v = [state.v for state in states]
    v_linear = [state.v for state in states_linear]
    plt.plot(theta, label='nonlinear')
    plt.plot(theta_linear, label='linear')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
