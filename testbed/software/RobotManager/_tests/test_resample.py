from core.utils.ilc.ILC_DAMN_bib import *
from core.utils.data import resample, generate_time_vector
from matplotlib import pyplot as plt

if __name__ == '__main__':
    N = len(reference)
    t_old = generate_time_vector(start=0, end=(N-1) * 0.02, dt=0.02)

    t_new = generate_time_vector(start=0, end=(N-1) * 0.02, dt=0.01)
    u_new = resample(t_old, reference, t_new)

    plt.figure()
    plt.plot(t_old, reference, label='Reference', color='blue')
    plt.plot(t_new, u_new, label='Resampled', color='orange')
    plt.show()
