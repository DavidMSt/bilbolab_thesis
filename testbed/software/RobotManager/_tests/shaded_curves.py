import numpy as np
import matplotlib.pyplot as plt

from core.utils.colors import get_shaded_color


def test_shaded_color_plot(base_color="blue", num_curves=10):
    """
    Plot num_curves random curves with progressively darker shades of base_color.

    Args:
        base_color (str | list | tuple): Base color.
        num_curves (int): Number of curves to plot.
    """
    x = np.linspace(0, 10, 200)
    plt.figure(figsize=(10, 6))

    for j in range(num_curves):
        y = np.sin(x + np.random.rand() * 2 * np.pi) + 0.1 * np.random.randn(len(x))
        color = get_shaded_color(base_color, num_curves, j)
        plt.plot(x, y, label=f"Curve {j}", color=color)

    plt.title("Test: Progressively Shaded Curves")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test_shaded_color_plot()
