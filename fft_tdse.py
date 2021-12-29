import numpy as np
from numpy.fft import fftn, ifftn, fftshift
import matplotlib.pyplot as plt
from scipy.interpolate import interpn

class FourierGrid:
    def __init__(self,a,b,ng):
        """ Constructor for FourierGrid.

        Example:
        a = (-2,-3,-4)
        b = (3,4,5)
        ng = (128,128,128)
        grid = FourierGrid(a,b,ng)

        a, b = array/list/tuple of left/right interval ends for each dimension
        ng = array/list/tuple of number of grid points for each dimension. should be powers of 2.

        """
        self.defineGrid(a,b,ng)

        return

    def fftgrid(self,a,b,N):
        """ Compute grid for Fourier pseudospectral method on an interval.

        Example:

        x, k = fftgrid(a, b, N).

        Generate grid of N points in the interval [a,b] with periodic BCs. The point b is not included,
        i.e., it is identified with a. Returns x = grid points as array, and f = frequencies as array. If
        psi is a (periodic) function evaluated at the grid points, its
        derivative at the grid points is given to N'th order by

           dpsi = ifft(1j*k*fft(psi)).

        """

        x = np.linspace(a,b,N+1)
        x = x[:-1]
        h = (b-a)/N

        k_c = np.pi/h
        k = np.linspace(-k_c, k_c, N+1)
        k = np.fft.ifftshift(k[:-1])
        return x,k

    def defineGrid(self,a,b,ng):
        """ Set the grid. Called from constructor, but can be called at other times, too. See doc for constructor. """
        self.a = tuple(a)
        self.b = tuple(b)
        self.ng = tuple(ng)

        assert(len(a) == len(ng))
        d = len(ng)
        self.d = d

        #compute fft grid
        self.x = []
        self.k = []
        for i in range(d):
            x0,k0 = self.fftgrid(a[i], b[i], ng[i])
            self.x.append(x0)
            self.k.append(k0)

        self.xx = np.meshgrid(*tuple(self.x),indexing='ij')
        self.kk = np.meshgrid(*tuple(self.k),indexing='ij')

        self.h = np.zeros(self.d)
        for n in range(self.d):
            self.h[n] = self.x[n][1] - self.x[n][0]
        self.dtau = np.prod(self.h)

    def __str__(self):
        return str(self.__class__) + f": a = {self.a}, b = {self.b}, ng = {self.ng}"  #self.__dict__

class FourierWavefunction:
    """ Class for representing the wavefunction *and* its Fourier transform on a FourierGrid.

    Every class instance has three members: grid, psi, and phi. As a rule of thump, phi is always
    the FFT of psi, but to reduce extra fftn/ifftn calls, one can locally disable the automatic
    computation of the dual function.

    The automatic computation of the dual wavefunction actually makes sense for the split-step
    Fourier method: whenever psi is propagated with exp(-ihV), the next step is usually propagation
    with the kinetic operator. Hence, the FFT of psi will be needed immediately. Similarly, after a kinetic propagation,
    the next step will usually be propagation with the potential. Hence, the ifft of phi is needed immediately.


    """

    def __init__(self, grid):
        """ Constructor. Pass a FourierGrid instance as parameter. The wavefunction and its fft is created."""
        self.grid = grid
        self.psi = np.zeros(self.grid.ng,dtype=complex)
        self.phi = np.zeros(self.grid.ng,dtype=complex)

        self.currmap = lambda psi,n:  np.sum(np.imag(ifftn(1j*k[n]*fftn(psi)) * psi.conj()),axis=notn(n))

    def density(self,n):
        """ compute density along axis n. """
        notn = lambda n: tuple(k for k in range(self.grid.d) if k != n) # all axes except n
        rho = np.sum(np.abs(self.psi)**2,axis=notn(n)) * self.grid.dtau / self.grid.h[n]
        return rho

    def current(self,n):
        """ compute current density along axis n. """
        notn = lambda n: tuple(k for k in range(self.grid.d) if k != n) # all axes except n
        jp = np.sum(np.imag(ifftn(1j*self.grid.kk[n]*self.phi) * self.psi.conj()),axis=notn(n)) * self.grid.dtau / self.grid.h[n]
        return jp

    def phiNorm(self):
        return np.linalg.norm(self.phi) * self.grid.dtau ** .5
    def psiNorm(self):
        return np.linalg.norm(self.psi) * self.grid.dtau ** .5

    def normalizePsi(self):
        """ Normalize the spatial wavefunction in the L2 sense. """
        self.psi /= self.psiNorm()

    def normalizePhi(self):
        """ Normalize the frequancy wavefunction in the L2 sense. """
        n = np.linalg.norm(self.phi) * self.grid.dtau ** .5
        self.phi /= self.phiNorm()


    def setPsi(self,psi,normalize = False, set_dual = True, copy = True):
        """ Set the spatial wavefunction psi.

        setPsi(psi,normalize,set_dual,copy)

        normalize = False (default) does noe normalize the wavefunction in the L2 sense.
        set_dual = True (default) also sets the dual wavefunciton
        copy = True (default) makes a copy of psi, otherwise a reference is used.
        """
        if copy:
            self.psi = psi.copy()
        else:
            self.psi = psi

        if normalize:
            self.normalizePsi()

        if set_dual:
            self.phi = fftn(self.psi)

    def setPhi(self,phi,normalize = False, set_dual = True, copy = True):
        """ Set the frequency wavefunction phi.

        setPhi(phi,normalize,set_dual,copy)

        normalize = False (default) does noe normalize the wavefunction in the L2 sense.
        set_dual = True (default) also sets the dual wavefunciton
        copy = True (default) makes a copy of psi, otherwise a reference is used.
        """
        if copy:
            self.phi = phi.copy()

        else:
            self.phi = phi

        if normalize:
            self.normalizePhi()

        if set_dual:
            self.psi = ifftn(self.phi)

    def applySpaceOperator(self, V, update_dual = True):
        """ Apply a local potential V to psi. If update_dual == True, then phi is recomputed."""
        self.psi = V * self.psi
        if update_dual:
            self.phi = fftn(self.psi)

    def applyFrequencyOperator(self, V, update_dual = True):
        """ Apply a local potential V to phi. If update_dual == True, then psi is recomputed."""
        self.phi = V * self.phi
        if update_dual:
            self.psi = ifftn(self.phi)

    def interpolate(self,new_grid,kind='linear'):
        """ Interpolate / pad psi at new grid. """

        d = new_grid.d
        shape = new_grid.ng
        nn = np.prod(shape)

        xi = np.zeros((nn,d))
        for i in range(d):
            xi[:,i] = np.reshape(new_grid.xx[i],(nn,))

        u = interpn(tuple(self.grid.x),self.psi.real,xi, method=kind, bounds_error=False, fill_value=None)
        v = interpn(tuple(self.grid.x),self.psi.imag,xi, method=kind, bounds_error=False, fill_value=None)

        psi = u + 1j * v
        psi = psi.reshape(shape)

        wf = FourierWavefunction(new_grid)
        wf.setPsi(psi,set_dual=True)
        return wf

def T_standard(k):
    shape = k[0].shape
    d = len(k)
    result = np.zeros(shape)
    for n in range(d):
        result += 0.5 * k[n]**2

    return result

def E_zero(t):
    return 0.0

def D_dipole(x):
    """ Dipole operator."""
    shape = x[0].shape
    d = len(x)
    D = np.zeros(shape) # dipole operator.
    for n in range(d):
        D += x[n]

    return D

class FourierHamiltonian:
    """ Class for representing the system Hamiltonian on a FourierGrid."""
    def __init__(self, grid, Vfun, Tfun = T_standard, Dfun = D_dipole, Efun = E_zero):
        """ Constructor.

        $$ H(t) = T + V + E(t)D $$

        Tfun = a function that evaluates the kinetic potential in a k-vector, or an ndarray
        Vfun = a function that evaluates the spatial potential in an x-vector, or an ndarray
        Dfun = a function that evaluates the spatial potential in an x-vector, or an ndarray
        Efun = a function that evaluates the electric field at a time t.

        """
        self.grid = grid
        self.setKineticPotential(Tfun)
        self.setSpatialPotential(Vfun)
        self.setTimeDependentPotential(Dfun)
        self.Efun = Efun

    def setTimeDependentPotential(self,Dfun):
        if type(Dfun) == np.ndarray:
            self.D = Dfun
        else:
            self.Dfun = Dfun
            self.D = self.Dfun(self.grid.xx)

    def setKineticPotential(self,Tfun):
        if type(Tfun) == np.ndarray:
            self.T = Tfun
        else:
            self.Tfun = Tfun
            self.T = Tfun(self.grid.kk)

    def setSpatialPotential(self,Vfun):
        """ Set the spatial potential.

        Vfun: a function that is evaluated, or an ndarray whose reference
        is stored.
        """

        if type(Vfun) == np.ndarray:
            self.V = Vfun
        else:
            self.Vfun = Vfun
            self.V = Vfun(self.grid.xx)

    def apply(self,psi):
        """ Apply the Hamiltonian (minus E-field) to a spatial wavefunction psi. """
        return ifftn(self.T*fftn(psi)) + self.V*psi

class Propagator:
    """ Class for Strang splitting propagation of a FourierWavefunction using a FourierHamiltonian."""
    def __init__(self,ham,dt,time_dependent = True):
        """ Constructor.

        ham = FourierHamiltonian instance.
        dt = time step.
        time_dependent: if False (not default), the E-field is turned off.
        """
        self.ham = ham
        self.setTimeStep(dt)
        self.time_dependent = time_dependent

    def setTimeStep(self,dt):
        """ Set the time step dt."""
        self.dt = dt
        self.Tprop = np.exp(-0.5j*dt*self.ham.T)
        self.Vprop = np.exp(-1j*dt*self.ham.V)
        self.Eprop = lambda t: np.exp(-1j*dt*self.ham.Efun(t)*self.ham.D)

    def strang(self,wf,t,will_do_another_step = True):
        """ Perform a step of the Strang splitting propagator.

        wf = wavefuntion to propagate
        t = time
        will_do_another_step: if True (default), the spatial wavefunction is *not* recomputed at the end of
        the step, since that would incur an extra fftn call."""

        wf.applyFrequencyOperator(self.Tprop)

        if self.time_dependent:
            wf.applySpaceOperator(self.Vprop, update_dual = False)
            wf.applySpaceOperator(self.Eprop(t+.5*self.dt))
        else:
            wf.applySpaceOperator(self.Vprop)

        #wf.applyFrequencyOperator(self.Tprop,update_dual = not will_do_another_step)

        wf.applyFrequencyOperator(self.Tprop, update_dual = not will_do_another_step)


from scipy.sparse.linalg import cg, LinearOperator

class GroundStateComputer:
    """ Class for computing the ground state of a FourierHamiltonian (over its associated FourierGrid)."""
    def __init__(self, ham):
        self.ham = ham
        self.grid = ham.grid
        self.wf = FourierWavefunction(self.grid)

    def setInitialGuess(self, psi):
        """ Set the initial guess of the spatial wavefunction. """
        self.wf.setPsi(psi, set_dual = False)


    def Apsi_vec(self, psi_vec, sigma = 0.0):
        ng = self.grid.ng

        """ Matrix-vector product for inverse iterations with shift sigma. """
        return self.ham.apply(psi_vec.reshape(ng)).reshape((np.prod(ng),)) - sigma*psi_vec




    def invit(self,tol=1e-9,cgtol=1e-9, maxit=400,sigma=0):
        """ Inverse iterations using conjugate gradients. """

        shape = self.grid.ng
        n = np.prod(shape)

        A = LinearOperator((n,n), matvec=lambda x:self.Apsi_vec(x,sigma=sigma), dtype=complex)

        delta = 1000
        psi = self.wf.psi.reshape((n,))
        psi = psi/np.linalg.norm(psi)
        for k in range(maxit):
            psi_prev = psi
            psi, info = cg(A, psi, x0=psi, tol=np.min([cgtol,delta]))
            psi = psi / np.linalg.norm(psi)
            #self.wf.setPsi(psi, set_dual = False)
            delta = np.linalg.norm(psi-psi_prev)
            Hpsi = self.ham.apply(psi.reshape(shape)).reshape((n,))
            E = np.inner(psi,Hpsi)

            resid = np.linalg.norm(Hpsi - E*psi)
            print(f'Iteration {k}, delta = {delta}, resid = {resid}, E = {E.real}')
            if delta < 10*tol:
                print('Iterations terminated successfully.')
                break

        self.wf.setPsi(psi.reshape(shape), set_dual=True,normalize=True)
        self.E = E.real

        return self.E

    def imagTimePropagation(self,tol=1e-9,dt=.01,maxit=400,sigma=0):
        p = Propagator(self.ham,-1j*dt,time_dependent=False)

        shape = self.grid.ng
        n = np.prod(shape)

        delta = 1000
        for k in range(maxit):
            psi_prev = self.wf.psi
            p.strang(self.wf,0.0,will_do_another_step=False)
            self.wf.normalizePsi()

            #self.wf.setPsi(psi, set_dual = False)
            delta = np.linalg.norm(self.wf.psi-psi_prev) * self.grid.dtau ** .5
            Hpsi = ham.apply(self.wf.psi).reshape((n,))
            E = np.inner(self.wf.psi.reshape((n,)),Hpsi) * self.grid.dtau

            resid = np.linalg.norm(Hpsi - E*self.wf.psi.reshape((n,)))
            print(f'Iteration {k}, delta = {delta}, resid = {resid}, E = {E.real}')
            if delta < 10*tol:
                print('Iterations terminated successfully.')
                break

        self.E = E.real

        return self.E
