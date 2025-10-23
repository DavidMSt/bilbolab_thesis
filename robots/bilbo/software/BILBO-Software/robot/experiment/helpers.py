from core.utils.data import generate_time_vector, generate_random_input
from robot.control.bilbo_control_data import BILBO_Control_Mode
from robot.experiment.definitions import BILBO_InputTrajectoryStep, BILBO_InputTrajectory
from robot.lowlevel.stm32_general import LOOP_TIME_CONTROL, MAX_STEPS_TRAJECTORY


def generateTrajectoryInputsFromList(trajectory_inputs: list) -> list:
    """
    Generates a dictionary of `BILBO_InputTrajectoryStep` objects from a list of inputs.

    This function processes a list of trajectory inputs and calculates the left and
    right input values for each step. If an input is a list, its first and second
    values are used directly as left and right inputs. If an input is not a list,
    the input value is evenly split into left and right inputs. The function
    returns a dictionary where keys represent step indices, and values are
    `BILBO_InputTrajectoryStep` objects representing each step.

    Args:
        trajectory_inputs (list): A list of inputs where each element is either a
            single value or a list of two values. Single values are split equally
            into left and right inputs. Lists specify directly the left and right
            input values.

    Returns:
        dict: A dictionary where keys are step indices (int), and values are
        `BILBO_InputTrajectoryStep` objects representing the corresponding trajectory step.
    """
    trajectory_inputs_list = []

    for i, inp in enumerate(trajectory_inputs):
        if isinstance(inp, list):
            input_left = float(inp[0])
            input_right = float(inp[1])
        else:
            input_left = float(inp) / 2
            input_right = float(inp) / 2

        trajectory_inputs_list.append(BILBO_InputTrajectoryStep(
            step=i,
            left=input_left,
            right=input_right,
        ))

    return trajectory_inputs_list



# ----------------------------------------------------------------------------------------------------------------------
def generateRandomTestTrajectory(trajectory_id, time_s, frequency, gain) -> BILBO_InputTrajectory | None:
    """
    Generates a random test trajectory for simulation or testing purposes. The function creates a time
    vector based on the specified duration and generates random inputs filtered by a cutoff frequency
    and scaled by the provided gain. If the trajectory exceeds the maximum allowed steps, the function
    returns None. Otherwise, it returns a trajectory object containing the generated data.

    Args:
        trajectory_id: Identifier for the generated trajectory.
        time_s: Maximum time duration of the trajectory in seconds.
        frequency: Cutoff frequency for filtering random inputs.
        gain: Scaling factor for random input signal amplitude.

    Returns:
        BILBO_InputTrajectory | None: The trajectory object containing the generated data or None
        if the trajectory exceeds the maximum allowed steps.
    """
    t_vector = generate_time_vector(start=0, end=time_s, dt=LOOP_TIME_CONTROL)

    if len(t_vector) > MAX_STEPS_TRAJECTORY:
        print(f"Trajectory too long: {len(t_vector)} > {MAX_STEPS_TRAJECTORY} steps")
        return None

    trajectory_input = generate_random_input(t_vector=t_vector, f_cutoff=frequency, sigma_I=gain)
    trajectory_inputs = generateTrajectoryInputsFromList(trajectory_input)

    trajectory = BILBO_InputTrajectory(
        id=trajectory_id,
        time_vector=t_vector,
        name='test',
        length=len(trajectory_inputs),
        inputs=trajectory_inputs,
        control_mode=BILBO_Control_Mode.BALANCING,
    )

    return trajectory
