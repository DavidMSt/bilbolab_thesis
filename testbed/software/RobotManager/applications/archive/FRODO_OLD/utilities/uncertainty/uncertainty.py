import json
import math
import numpy as np
import scipy.interpolate as sci
import matplotlib.pyplot as plt

from core.utils.files import relativeToFullPath

ERROR_FACTOR = 2
ERROR_FILE_PATH = relativeToFullPath("error_data/v3_tagged_error_data.json")

error_file = open(ERROR_FILE_PATH)
error_data = json.load(error_file)

x = []
y = []
dist = []
angle = []

for i in range(len(error_data)):
    x.append(error_data[str(i)]['x'])
    y.append(error_data[str(i)]['y'])
    dist.append(error_data[str(i)]['err_dist'])
    angle.append(error_data[str(i)]['err_angle'])

points = np.array(list(zip(x,y)))

def uncertainty_angle(x_in, y_in):
    z_out = sci.griddata(points, angle, (x_in, y_in), method='linear')

    if math.isnan(z_out):
        return None
    
    return ERROR_FACTOR*z_out

def uncertainty_distance(x_in, y_in):
    z_out = sci.griddata(points, dist, (x_in, y_in), method='linear')

    if math.isnan(z_out):
        return None

    return ERROR_FACTOR*z_out


def debug_plot_distance():
    xi = np.linspace(0,3, 100)
    yi = np.linspace(-1.5, 1.5, 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = []
    for i in range(len(xi)):
        z_tmp = []
        for j in range(len(xi[0])):
            z_tmp.append(uncertainty_distance(xi[i][j], yi[i][j]))
        zi.append(z_tmp)

    zi = np.asarray(zi)

    fig, ax = plt.subplots(1,1,subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xi,yi,zi,cmap='RdYlBu_r', vmin=0, vmax=0.25)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('error_distance')
    ax.view_init(elev=22.5, azim=215)
    fig.colorbar(surf)

    plt.show()

def debug_plot_angle():
    xi = np.linspace(0,3, 100)
    yi = np.linspace(-1.5, 1.5, 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = []
    for i in range(len(xi)):
        z_tmp = []
        for j in range(len(xi[0])):
            z_tmp.append(uncertainty_angle(xi[i][j], yi[i][j]))
        zi.append(z_tmp)

    zi = np.asarray(zi)

    fig, ax = plt.subplots(1,1,subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xi,yi,zi,cmap='RdYlBu_r', vmin=0, vmax=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('error_angle')
    ax.view_init(elev=22.5, azim=215)
    fig.colorbar(surf)

    plt.show()

if __name__ == "__main__":
    print(uncertainty_distance(-1.5,3))
    debug_plot_distance()