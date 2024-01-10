import numpy as np
from scipy.interpolate import RectBivariateSpline


def interpolate_2d(psi, extent=[0, 1, 0, 1], kx=1, ky=1):
    """
    Make a function from grid wavefunction in 2d:
    Interpolates a 2D complex ndarray using a rectangular bivariate spline.

    Parameters:
    - psi (ndarray): The 2D ndarray to be interpolated.
    - extent (list): The extent of the interpolation in the form [xmin, xmax, ymin, ymax].

    Returns:
    - function: A lambda function that performs the interpolation. The function takes two arguments (x, y) and returns the interpolated value at that point.
    """
    x_range = np.linspace(extent[0], extent[1], psi.shape[1])
    y_range = np.linspace(extent[2], extent[3], psi.shape[0])

    interp_func_real = RectBivariateSpline(y_range, x_range, psi.real, kx=kx, ky=ky)
    interp_func_imag = RectBivariateSpline(y_range, x_range, psi.imag, kx=kx, ky=ky)
    return lambda x, y: interp_func_real(y, x, grid=False) + 1j*interp_func_imag(y, x, grid=False)
 

if __name__ == "__main__":
    # test the interpolation function
    x0 = np.linspace(-2, 2, 100)
    x, y = np.meshgrid(x0, x0)
    psi = x * (1 + y)
    interp_func = interpolate_ndarray(psi, extent=[-2, 2, -2, 2])

    for k in range(100):
        xi = np.random.uniform(-2, 2)
        yi = np.random.uniform(-2, 2)
        assert interp_func(xi, yi) - (xi * (1 + yi)) < 1e-15
