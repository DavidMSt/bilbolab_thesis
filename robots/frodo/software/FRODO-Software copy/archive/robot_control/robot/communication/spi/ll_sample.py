import ctypes


class sample_general(ctypes.Structure):
    _fields_ = [("tick", ctypes.c_uint32), ("status", ctypes.c_uint8)]


class twipr_gyr_data(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float)]


class twipr_acc_data(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float)]


class twipr_sensor_data(ctypes.Structure):
    _fields_ = [("speed_left", ctypes.c_float), ("speed_right", ctypes.c_float), ("acc", twipr_acc_data),
                ("gyr", twipr_gyr_data), ("battery_voltage", ctypes.c_float)]


class twipr_estimation_state(ctypes.Structure):
    _fields_ = [("v", ctypes.c_float), ("theta", ctypes.c_float), ("theta_dot", ctypes.c_float),
                ("psi", ctypes.c_float), ("psi_dot", ctypes.c_float)]


class sample_estimation(ctypes.Structure):
    _fields_ = [('state', twipr_estimation_state)]


class twipr_control_input(ctypes.Structure):
    _fields_ = [("u1", ctypes.c_float), ("u2", ctypes.c_float)]


class twipr_control_output(ctypes.Structure):
    _fields_ = [("u_left", ctypes.c_float), ("u_right", ctypes.c_float)]


class sample_control(ctypes.Structure):
    _fields_ = [('status', ctypes.c_int8), ('mode', ctypes.c_int8), ("input", twipr_control_input),
                ("output", twipr_control_output), ("trajectory_step", ctypes.c_uint32),
                ("trajectory_id", ctypes.c_uint16)]


class twipr_sample(ctypes.Structure):
    _fields_ = [("general", sample_general), ("control", sample_control), ("estimation", sample_estimation),
                ("sensors", twipr_sensor_data)]


class trajectory_input(ctypes.Structure):
    _fields_ = [("step", ctypes.c_uint32), ("u_1", ctypes.c_float), ("u_2", ctypes.c_float)]


class trajectory_struct(ctypes.Structure):
    _fields_ = [('step', ctypes.c_uint16), ('id', ctypes.c_uint16), ('length', ctypes.c_uint16)]
