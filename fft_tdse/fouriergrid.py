import numpy as np
from scipy.interpolate import interpn, RectBivariateSpline
from icecream import ic
from numpy.fft import fft2, ifft2


def fftgrid(a, b, N):
    """Compute grid for Fourier pseudospectral method on an interval.

    Example:

    x, k = fftgrid(a, b, N).

    Generate grid of N points in the interval [a,b] with periodic BCs. The point b is not included,
    i.e., it is identified with a. Returns x = grid points as array, and f = frequencies as array. If
    psi is a (periodic) function evaluated at the grid points, its
    derivative at the grid points is given to N'th order by

       dpsi = ifft(1j*k*fft(psi)).

    """

    x = np.linspace(a, b, N + 1)
    x = x[:-1]
    h = (b - a) / N

    k_c = np.pi / h
    k = np.linspace(-k_c, k_c, N + 1)
    k = np.fft.ifftshift(k[:-1])
    return x, k


class NewFourierGrid:
    """Class for holding a multidimensional Fourier grid.

    This class implements a simple way to handle multidimensional grids for the Fourier pseudospectral method.


    TODO:
    Replace old with new implementation. The difference is the definition of xx. Previous xx[i] must be replaced with xx[...,i] in
    all code that uses the class.

    Attributes:
        a (list of float):
            list of left endpoints for each dimension
        b (list of float):
            list of right endpoints for each dimension
        ng (list of int):
            list of number of grid points for each dimension
        x (list of ndarray):
            1D Fourier grids for each dimension
        k (list of ndarray):
            1D Fourier frequency grids for each dimension
        xx (list of ndarray):
            spatial meshgrid. xx[...,i] is the i'th dimension's grid.
        kk (list of ndarray):
            frequency meshgrid. xx[...,i] is the i'th dimension's grid.
        h (list of float):
            mesh spacings (space)
        dtau (float):
            volume element, product of h
        d (int):
            dimension
    """

    def __init__(self, a, b, ng):
        """Constructor for Fouriergrid.

        Args:
            a (tuple, list):
                left endpoints
            b (tuple, list):
                right endpoints
            ng (tuple, list):
                number of grid points


        Example:
            a = (-2,-3,-4)
            b = (3,4,5)
            ng = (128,128,128)
            grid = FourierGrid(a,b,ng)
        """

        self.a = tuple(a)
        self.b = tuple(b)
        self.ng = tuple(ng)

        assert len(a) == len(ng)
        d = len(ng)
        self.d = d

        # compute fft grid
        self.x = []
        self.k = []
        for i in range(d):
            x0, k0 = fftgrid(a[i], b[i], ng[i])
            self.x.append(x0)
            self.k.append(k0)

        # The following results in an array xx (kk)
        # whose last dimension indexes the spatial
        # component. I.e., self.xx[...,i] is the meshgrid of component i.

        xx = np.meshgrid(*tuple(self.x), indexing="ij")
        kk = np.meshgrid(*tuple(self.k), indexing="ij")
        self.xx = np.moveaxis(np.array(xx), 0, -1)
        self.kk = np.moveaxis(np.array(kk), 0, -1)

        self.h = np.zeros(self.d)
        for n in range(self.d):
            self.h[n] = self.x[n][1] - self.x[n][0]
        self.dtau = np.prod(self.h)

    def __str__(self):
        return str(self.__class__) + f": a = {self.a}, b = {self.b}, ng = {self.ng}"

    def get_meshgrid(self):
        """Return meshgrid version of grid."""

        return tuple(self.xx[..., i].squeeze() for i in range(self.xx.shape[-1]))


class FourierGrid:
    def __init__(self, a, b, ng):
        """Constructor for FourierGrid.

        Example:
        a = (-2,-3,-4)
        b = (3,4,5)
        ng = (128,128,128)
        grid = FourierGrid(a,b,ng)

        a, b = array/list/tuple of left/right interval ends for each dimension
        ng = array/list/tuple of number of grid points for each dimension. should be powers of 2.

        """
        self.defineGrid(a, b, ng)

        return

    #
    # def fftgrid(self,a,b,N):
    #     """ Compute grid for Fourier pseudospectral method on an interval.
    #
    #     Example:
    #
    #     x, k = fftgrid(a, b, N).
    #
    #     Generate grid of N points in the interval [a,b] with periodic BCs. The point b is not included,
    #     i.e., it is identified with a. Returns x = grid points as array, and f = frequencies as array. If
    #     psi is a (periodic) function evaluated at the grid points, its
    #     derivative at the grid points is given to N'th order by
    #
    #        dpsi = ifft(1j*k*fft(psi)).
    #
    #     """
    #
    #     x = np.linspace(a,b,N+1)
    #     x = x[:-1]
    #     h = (b-a)/N
    #
    #     k_c = np.pi/h
    #     k = np.linspace(-k_c, k_c, N+1)
    #     k = np.fft.ifftshift(k[:-1])
    #     return x,k

    def defineGrid(self, a, b, ng):
        """Set the grid. Called from constructor, but can be called at other times, too. See doc for constructor."""
        self.a = tuple(a)
        self.b = tuple(b)
        self.ng = tuple(ng)

        assert len(a) == len(ng)
        d = len(ng)
        self.d = d

        # compute fft grid
        self.x = []
        self.k = []
        for i in range(d):
            x0, k0 = fftgrid(a[i], b[i], ng[i])
            self.x.append(x0)
            self.k.append(k0)

        self.xx = np.meshgrid(*tuple(self.x), indexing="ij")
        self.kk = np.meshgrid(*tuple(self.k), indexing="ij")

        self.h = np.zeros(self.d)
        for n in range(self.d):
            self.h[n] = self.x[n][1] - self.x[n][0]
        self.dtau = np.prod(self.h)

    def __str__(self):
        return (
            str(self.__class__) + f": a = {self.a}, b = {self.b}, ng = {self.ng}"
        )  # self.__dict__


def ensure_newfouriergrid(grid):
    """Convert to NewFourierGrid."""

    if isinstance(grid, NewFourierGrid):
        return grid

    return NewFourierGrid(grid.a, grid.b, grid.ng)


def make_grid(x, Type=NewFourierGrid):
    """Make one-dimensional FourierGrid from a vector of uniformly spaced points that misses the right endpoint."""

    h = x[1] - x[0]
    a = x.min()
    b = x.max() + h
    grid = Type([a], [b], [len(x)])
    return grid


def interpolate(psi, grid, new_grid, order=3):
    """Interpolate / pad psi at new grid.

    A function psi is defined on a FourierGrid grid, and then interpolated/padded
    to a new FourierGrid. The parameter order can be used to determine the order
    of the interpolation. In 2d, order can be any integer (e.g., order=3 for cubic),
    but in other dimensions, order=1 or order=3 are the only options. For 2d,
    scipy.interpolate.RectBivariateSpline is used, but interpn is used in other dimensions.

    Args:
    * psi (ndarray): values on grid
    * grid (FourierGrid): grid where psi lives
    * new_grid (FourierGrid): grid where psi is to be defined
    * kind (st

    Returns:
    * (ndarray): psi evaluated at new grid


    TODO: update to new FourierGrid class. See comments.
    """

    d = new_grid.d
    shape = new_grid.ng
    nn = np.prod(shape)

    xi = np.zeros((nn, d))
    for i in range(d):
        # xi[:,i] = np.reshape(new_grid.xx[...,i],(nn,)) # for new FourierGrid class
        if isinstance(new_grid, FourierGrid):
            xi[:, i] = np.reshape(new_grid.xx[i], (nn,))
        elif isinstance(new_grid, NewFourierGrid):
            xi[:, i] = np.reshape(new_grid.xx[..., i], (nn,))
        else:
            raise ValueError("new_grid has wrong type")

    if d != 2:  # extremely slow! must be fixed for generel dimensions
        kind = "cubic"
        # print('Running interpn ...')
        if not (order == 1 or order == 3):
            order = 1  # default to linear interpolation if order is not 1 or 3.

        if order == 1:
            kind = "linear"
        else:  # order ==3 only possibility left
            kind = "cubic"

        u = interpn(
            tuple(grid.x),
            psi.real,
            xi,
            method=kind,
            bounds_error=False,
            fill_value=None,
        )
        v = interpn(
            tuple(grid.x),
            psi.imag,
            xi,
            method=kind,
            bounds_error=False,
            fill_value=None,
        )

    else:  # super fast! identical results in 2D.
        # print('Running RectBivariateSpline ... ')
        u = RectBivariateSpline(*tuple(grid.x), psi.real, kx=order, ky=order, s=0)(
            *tuple(new_grid.x)
        )
        v = RectBivariateSpline(*tuple(grid.x), psi.imag, kx=order, ky=order, s=0)(
            *tuple(new_grid.x)
        )

    psi = u + 1j * v
    psi = psi.reshape(shape)

    return psi


def to_polar_2d(psi, grid, r_grid, theta_grid, order=3):
    """Evaluate psi on a polar coordinate grid in 2d

    A function psi is defined on a FourierGrid grid, and evaluated on a
    2d polar coordinate grid. Since evaluation will happen between
    real grid points, interpolation must be used. The parameter order
    decides the inerpolation order.

    Args:
    * psi (ndarray): values on grid
    * grid (FourierGrid): grid where psi lives
    * r_grid (ndarray): grid points for radial coord
    * theta_grid (ndarray): grid points for angular coord
    * order (int): interpolation order

    Returns:
    * (ndarray): psi evaluated at polar coord grid


    TODO: update to new FourierGrid class. See comments.
    """

    # make sure we are doing 2d functions
    assert grid.d == 2
    use_grid = ensure_newfouriergrid(grid)

    rr, tt = np.meshgrid(r_grid, theta_grid)
    xx = rr * np.cos(tt)
    yy = rr * np.sin(tt)
    nn = np.prod(xx.shape)

    u = RectBivariateSpline(*tuple(use_grid.x), psi.real, kx=order, ky=order, s=0)(
        xx.reshape(-1), yy.reshape(-1), grid=False
    )
    v = RectBivariateSpline(*tuple(use_grid.x), psi.imag, kx=order, ky=order, s=0)(
        xx.reshape(-1), yy.reshape(-1), grid=False
    )

    u = u.reshape(xx.shape)
    v = v.reshape(xx.shape)

    psi2 = u + 1j * v

    # compute norm ...

    # dr = r_grid[1]-r_grid[0]
    # dt = theta_grid[1]-theta_grid[0]
    # psi2_norm = np.sum(np.abs(psi2)**2 * rr) * dr * dt
    # psi_norm = np.sum(np.abs(psi)**2) * grid.dtau
    # ic(psi2_norm, psi_norm )

    return psi2


def get_angmom_expectations(psi, grid):
    """Compute angular momenta expectations for 2d functions."""

    assert grid.d == 2

    use_grid = ensure_newfouriergrid(grid)

    x = use_grid.xx[..., 0]
    y = use_grid.xx[..., 1]
    kx = use_grid.kk[..., 0]
    ky = use_grid.kk[..., 1]

    phi = np.fft.fftn(psi, norm="ortho")
    ifftn = np.fft.ifftn
    Lz_psi = x * ifftn(ky * phi, norm="ortho") - y * ifftn(kx * phi, norm="ortho")
    Lz = np.sum(psi.conj() * Lz_psi) * grid.dtau
    Lz2 = np.sum(Lz_psi.conj() * Lz_psi) * grid.dtau
    return Lz.real, Lz2.real


def get_angmom_spectrum(psi, grid, Lmax, nt_factor=2, nr_factor=1, interp_order=3):
    """Compute spectral weights of the angular momentum
    operator in 2d. For simplicity we assume a centered box domain."""

    assert grid.d == 2
    use_grid = ensure_newfouriergrid(grid)

    nt = (2 * Lmax + 1) * nt_factor
    theta_grid = np.linspace(0, 2 * np.pi, nt)
    rmax = grid.b[0]
    nr = grid.ng[0] * nr_factor
    r_grid = np.linspace(0, rmax, nr)
    dr = r_grid[1] - r_grid[0]
    # we assume x and y to have the same grids ...
    # we need a quick implementation
    # assume about 4 grid points per annulus

    psi_polar = to_polar_2d(psi, use_grid, r_grid, theta_grid, order=interp_order)
    rr, tt = np.meshgrid(r_grid, theta_grid)
    #    print(rr.shape, nt, nr)

    # we are now fourier transforming along the theta axis.
    # the result is that the second axis now counts the angular momenta,
    # but there are much more than we want.
    phi_polar = np.fft.fftn(psi_polar, axes=(0,), norm="ortho")
    P = np.zeros(2 * Lmax + 1)
    L_avg = 0
    L2_avg = 0
    for m in range(-Lmax, Lmax + 1):
        # by construction of the FFT, this indexing works!
        u_m = phi_polar[m, :]
        P[m + Lmax] = 2 * np.pi * np.sum(np.abs(u_m) ** 2 * r_grid).real * dr / nt
        L_avg += m * P[m + Lmax]
        L2_avg += m * m * P[m + Lmax]
    #        ic(f'P[{m}] = {P[m+Lmax]}')

    #    print('Sum of squares = ', np.sum(P))
    #    ic(L_avg, L2_avg)

    return P.real, L_avg, L2_avg
