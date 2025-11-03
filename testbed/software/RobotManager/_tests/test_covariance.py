import numpy as np

from applications.FRODO.algorithm.algorithm import get_covariance_ellipse

if __name__ == '__main__':
    std_dev_x = 0.1
    std_dev_y = 0.1

    covariance = np.asarray([[std_dev_x**2, 0], [0, std_dev_y**2]])

    rx, ry, psi_ellipse = get_covariance_ellipse(covariance)
    print(rx, ry, psi_ellipse)
