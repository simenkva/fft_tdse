import numpy as np
from .fft_tdse import *
from .fouriergrid import FourierGrid
from tqdm import tqdm
from icecream import ic

from IPython import get_ipython
from .is_notebook import is_notebook


def icm(message):
    ic(message)


# choose tqdm implementation
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm import tqdm as tqdm_console

tqdm = tqdm_notebook if is_notebook() else tqdm_console


def check_function_signature(func, n_in, n_out):
    """
    Check if a function accepts n_in arrays of the same shape and returns n_out arrays of the same shape.


    Args:
        func (callable): The function to check.
        n_in (int): The number of arrays to check the function with.
        n_out (int): The number of arrays to check the function returns.

    Returns:
        bool: True if the function signature is valid, False otherwise.
    """
    try:
        shape = (1,)
        args = [np.empty(shape) for _ in range(n_in)]
        result = func(*args)

        if n_out > 1:
            if not isinstance(result, tuple) or len(result) != n_out:
                return False
            else:
                return all(
                    isinstance(arr, np.ndarray) and arr.shape == shape for arr in result
                )
        elif n_out == 1:
            if isinstance(result, np.ndarray) and result.shape == shape:
                return True
            else:
                return False
        else:
            return False

    except:
        return False


def I_peak_to_E_peak(Ipeak=3.51e16):
    """Convert peak intensity to peak electric field.

    Args:
        Ipeak (float): The peak intensity in W/cm^2. Default is equivalent to 1 atomic unit

    Returns:
        float: The peak electric field in atomic units.
    """
    return np.sqrt(Ipeak / 3.51e16)


class LaserPulse:
    r""" A class for defining laser pulses. The general form of the pulse is
    
    $$
    E(t) = E_0 \cdot f(t) \cdot \cos(\phi + \omega \cdot (t - t_0 - \frac{T}{2}))
    $$
    
    where $f(t)$ is an envelope function, by default given by
    
    $$ f(t) = \sin^2\left(\frac{\pi \cdot (t - t_0)}{T}\right). $$
    
    Another option is a trapezoidal envelope function, given by
    
    $$ f(t) = \begin{cases}
    t/t_0 & \text{if } t \leq T_0 \\
    1 & \text{if } T_0 < t \leq (N-1) T_0 \\
    N - t/T_0 & \text{if } (N-1) T_0 < t \leq N T_0   \\
    \end{cases} $$
        
    where $T_0 = T/N$.
    
    Finally, we have a square laser pulse, obtained by settint $N = np.inf$.
    
    Members:
        omega (float): The eldritch frequency that governs the pulse.
        t0 (float): The moment when the pulse emerges from the abyss.
        T (float): The duration of the pulse, a fleeting glimpse into the unknown.
        E0 (float): The amplitude of the pulse, a measure of its unfathomable power.
        phi (float): The phase shift of the pulse, in units of pi.
        N (int): The number of subdivisions in the trapezoidal envelope function.
    Functions:
        envelope_trap (float): The envelope function of the laser pulse, trapezoidal version.
        envelope_sin2 (float): The envelope function of the laser pulse, sin square version
        envelope (float): The selected envelope function of the laser pulse.
        __call__ (float): The laser pulse.
    
    """

    def __init__(self, omega, t0, T, E0, phi=0.0, N=10, envelope="sin2"):
        """Initialize a laser pulse.

        Args:
            omega (float): The eldritch frequency that governs the pulse.
            t0 (float): The moment when the pulse emerges from the abyss.
            T (float): The duration of the pulse, a fleeting glimpse into the unknown.
            E0 (float): The amplitude of the pulse, a measure of its unfathomable power.
            phi (float): The phase shift of the pulse, in units of pi.
        """
        self.omega = omega
        self.t0 = t0
        self.T = T
        self.E0 = E0
        self.phi = phi
        self.N = N
        self.envelope = (
            np.vectorize(self.envelope_sin2)
            if envelope == "sin2"
            else np.vectorize(self.envelope_trap)
        )

    def envelope_trap(self, t):
        """The envelope function of the laser pulse, trapezoidal version. Also
        supports a square pulse by setting N = np.inf.
        
        The definition of the envelope is:
        $$ f(t) = \begin{cases}
        (t-t_0)/t_0 & \text{if } (t-t_0) \leq T_0 \\
        1 & \text{if } T_0 < (t-t_0) \leq (N-1) T_0 \\
        N - t/T_0 & \text{if } (N-1) T_0 < (t-t_0) \leq N T_0   \\
        \end{cases} $$
        

        Args:
            t (float): The time.

        Returns:
            float: The envelope function.
        """
        N = self.N

        # square pulse envelope
        if np.isinf(N):
            return 1.0 if self.t0 <= t <= self.T + self.t0 else 0.0

        # trapezoidal envelope for N < np.inf
        T0 = self.T / N
        tt = t - self.t0
        if tt <= T0:
            return tt / T0
        elif tt <= (N - 1) * T0:
            return 1.0
        elif tt <= N * T0:
            return N - tt / T0
        else:
            return 0.0

    def envelope_sin2(self, t):
        """The envelope function of the laser pulse.


        The definition of the pulse is:
        If $t \leq t_0$ or $t \geq t_0 + T$, then $f(t) = 0$. If $t_0 < t < t_0 + T$, then
        $$ f(t) = \sin^2\left(\frac{\pi \cdot (t - t_0)}{T}\right). $$

        Args:
            t (float): The time.

        Returns:
            float: The envelope function.
        """
        if t <= self.t0 or t >= self.T + self.t0:
            return 0.0
        else:
            return np.sin(np.pi * (t - self.t0) / self.T) ** 2

    def __call__(self, t):
        """The laser pulse."""
        return (
            self.E0
            * self.envelope(t)
            * np.cos(self.phi * np.pi + self.omega * (t - (self.t0 + self.T / 2)))
        )


class Simulator:
    """A class for running simulations in 1d and 2d.

    The class defines a number of methods for setting up and running simulations.
    It provides a convenient interface to the FFT-TDSE code, and allows the user
    to easily read various variables during simulation using a callback feature.

    See also the Animator class for a convenient way to visualize the simulation
    using the callback feature.

    Here is an example of how to use the simulator:

    ```python
    # Import the Simulator class
    from fft_tdse.simulator import Simulator

    # Create a Simulator instance
    sim = Simulator()

    # Set the dimension of the simulation to 2
    sim.set_dimension(2)

    # Set the grid parameters
    sim.set_grid([-10, -10], [10, 10], [128, 128])

    # Set the potential function
    sim.set_potential(lambda x, y: x**2 + y**2)

    # Set the initial condition function
    sim.set_initial_condition(lambda x, y: np.exp(-((x-0.5)**2 + (y-0.5)**2)/0.1))

    # Set the time parameters
    sim.set_time_parameters(0, 1, 100)

    # Prepare the simulation
    sim.prepare()

    # Run the simulation
    sim.simulate()

    # Visualize the results using matplotlib
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(sim.psi.real)
    plt.show()

    Important attributes:
        mass (float): The particle mass. Default is 1.0.
        charge (float): The particle charge. Default is -1.0.
        dim (int): The dimension of the simulation. dim = 1 is default, dim=2 and 3 are supported.
        a (np.ndarray): The lower bounds of the domain.
        b (np.ndarray): The upper bounds of the domain.
        n (np.ndarray): The number of points in the grid.
        t0 (float): The initial time.
        t1 (float): The final time.
        n_steps (int): The number of time steps steps.
        wf (FourierWavefunction): The current wavefunction
        psi (np.ndarray): The current wavefunction, reference to wf.Psi, i.e., just a handy shortcut.
        laser_value  (float): The current value of the laser pulse field strength.


    ```

    """

    def __init__(self, verbose=True):
        """
        Initialize a Simulator instance. Sets various defaults.

        Args:
            verbose (bool): If True, enable verbose debugging using 'icecream' (ic).

        Returns:
            None
        """
        # set up debugging.
        if not verbose:
            ic.disable()

        self.mass = 1.0
        self.charge = -1.0
        self.dim = 1
        self.set_propagator('strang-3')

    def set_propagator(self, method = 'strang-3'):
        if method not in ['strang-3', 'crank-nicholson']:
            raise ValueError('Invalid integration scheme')
        
        ic()
        icm(f'Propagation method set to "{method}"')
        self.propagation_method = method

        
    def set_dimension(self, dim: int):
        """
        Set the dimension for the simulator.

        Args:
            dim (int): The dimension to be set (1 or 2 or 3).

        Raises:
            ValueError: If the specified dimension is not supported.

        Returns:
            None
        """
        if dim not in {1, 2, 3, 4}:
            raise ValueError("Dimension not supported.")

        self.dim = dim

        icm("Dimension set to {}".format(self.dim))

    def set_grid(self, a, b, n):
        """
        Set the grid for the simulation.

        The inputs a, b, and n can be either scalars or arrays of length dim.
        If they are scalars, they are assumed to be the same for all dimensions.
        If they are arrays, they must be of the same size, and the dimension
        is set to that length.

        Args:
            a (array-like): The lower bounds of the domain.
            b (array-like): The upper bounds of the domain.
            n (array-like): The number of points in the grid.

        Raises:
            ValueError: If the shapes of 'a' and 'b' and 'n' are not compatible.

        Returns:
            None
        """

        a = np.atleast_1d(np.asarray(a, dtype=float))
        b = np.atleast_1d(np.asarray(b, dtype=float))
        n = np.atleast_1d(np.asarray(n, dtype=int))

        # if user supplied a single value for a, b, n, repeat it
        # over each dimension.
        if len(a) == 1 and len(b) == 1 and len(n) == 1:
            a = np.repeat(a, self.dim)
            b = np.repeat(b, self.dim)
            n = np.repeat(n, self.dim)

        # check if the shapes are compatible
        # and set the grid parameters
        if len(a) == len(b) and len(a) == len(n):
            if self.dim != len(a):
                self.set_dimension(len(a))
            self.a = a
            self.b = b
            self.n = n
        else:
            raise ValueError("Bad domain parameters.")

        # create the grid
        self.grid = FourierGrid(self.a, self.b, self.n)

        # set handy attributes
        if self.dim == 1:
            self.x = self.grid.xx[0]
        elif self.dim == 2:
            self.x = self.grid.xx[0]
            self.y = self.grid.xx[1]
        elif self.dim == 3:
            self.x = self.grid.xx[0]
            self.y = self.grid.xx[1]
            self.z = self.grid.xx[2]
        elif self.dim == 4:
            self.x = self.grid.xx[0]
            self.y = self.grid.xx[1]
            self.z = self.grid.xx[2]
            self.w = self.grid.xx[3]
            
        icm("Grid set.")
        ic(self.a, self.b, self.n)

    def set_ground_state_grid(self, a_gs, b_gs, n_gs):
        """
        Set the grid for the ground state computation. This grid is independent
        of the main grid, and is optional. If not set, the main grid is used
        whenever the ground state or other eigenstates are computed.

        The arguments a_gs, b_gs, and n_gs can be either scalars or arrays of length dim.


        Args:
            a_gs (array-like): The lower bounds of the domain.
            b_gs (array-like): The upper bounds of the domain.
            n_gs (array-like): The number of points in the grid.

        Raises:
            ValueError: If the shapes of 'a' and 'b' and 'n' are not compatible.

        Returns:
            None
        """

        a_gs = np.atleast_1d(np.asarray(a_gs, dtype=float))
        b_gs = np.atleast_1d(np.asarray(b_gs, dtype=float))
        n_gs = np.atleast_1d(np.asarray(n_gs, dtype=int))

        # if user supplied a single value for a_gs, b_gs, n_gs, repeat it
        # over each dimension.
        if len(a_gs) == 1 and len(b_gs) == 1 and len(n_gs) == 1:
            a_gs = np.repeat(a_gs, self.dim)
            b_gs = np.repeat(b_gs, self.dim)
            n_gs = np.repeat(n_gs, self.dim)

        if len(a_gs) == len(b_gs) and len(a_gs) == len(n_gs):
            if self.dim != len(a_gs):
                self.set_dimension(len(a_gs))
            self.a_gs = a_gs
            self.b_gs = b_gs
            self.n_gs = n_gs
        else:
            raise ValueError("Bad domain parameters.")

        # create the grid
        self.grid_gs = FourierGrid(self.a_gs, self.b_gs, self.n_gs)

        # set handy attributes
        if self.dim == 1:
            self.x_gs = self.grid_gs.xx[0]
        elif self.dim == 2:
            self.x_gs = self.grid_gs.xx[0]
            self.y_gs = self.grid_gs.xx[1]
        elif self.dim == 3:
            self.x_gs = self.grid_gs.xx[0]
            self.y_gs = self.grid_gs.xx[1]
            self.z_gs = self.grid_gs.xx[2]
        elif self.dim == 4:
            self.x_gs = self.grid_gs.xx[0]
            self.y_gs = self.grid_gs.xx[1]
            self.z_gs = self.grid_gs.xx[2]
            self.w_gs = self.grid_gs.xx[3]
            
        icm("Ground state grid set.")
        ic(self.a_gs, self.b_gs, self.n_gs)

    def set_time_parameters(self, t0, t1, n_steps):
        """Set the paramters for integration. Also creates the time grid,
        and sets the initial time to t0.

        Args:
            t0 (float): The initial time.
            t1 (float): The final time.
            n_steps (int): The number of steps.

        Returns:
            None
        """

        assert t0 < t1
        assert n_steps > 0

        self.t0 = t0
        self.t1 = t1
        self.n_steps = n_steps

        self.t_grid = np.linspace(t0, t1, n_steps + 1)
        self.t = t0
        self.dt = self.t_grid[1] - self.t_grid[0]

        ic("Time parameters set.")
        ic(self.t0, self.t1, self.n_steps)

    # def set_potential_pointwise(self, V : np.ndarray):
    #     """Set the potential pointwise. Assumes that the grid is already set. Overrides
    #     any potential function that has been set.

    #     Args:
    #         V (np.ndarray): The potential values on the grid.
    #     Returns:
    #         None
    #     """

    #     # check the shape of V
    #     if V.shape != self.grid.ng:
    #         raise ValueError("Potential shape does not match grid shape.")

    #     self.V = V

    def set_potential(self, potential_fun: callable):
        """Set the potential function.

        Args:
            potential (callable): The potential function. Must take 'dim' arguments, and return a scalar. Assumed vectorized.

        Returns:
            None
        """

        # check number of args to potential
        # if not check_function_signature(potential_fun, self.dim, 1):
        #     raise ValueError("Potential has wrong function signature.")

        
        self.potential_fun = potential_fun
        
        icm("Potential function set set.")

    def set_mass(self, mass: float):
        """Set the particle mass.

        Args:
            mass (float): The particle mass.

        Returns:
            None
        """

        self.mass = mass

    def set_charge(self, charge: float):
        """Set the particle charge.

        Args:
            charge (float): The particle charge.

        Returns:
            None
        """

        self.charge = charge

    def set_initial_condition(self, initial_psi_fun: callable):
        """Set the initial condition as a function.

        Args:
            initial_psi_fun (callable): The initial condition function, assumed vectorized. Must take 'dim' arguments.

        Returns:
            None
        """

        # check number of args to initial_psi_fun
        # if not check_function_signature(initial_psi_fun, self.dim, 1):
        #     raise ValueError("initial_psi_fun has wrong function signature.")
        # if initial_psi_fun.__code__.co_argcount != self.dim:
        #     raise ValueError("initial_psi_fun must take {} arguments.".format(self.dim))

        if callable(initial_psi_fun):
            self.initial_psi_fun = initial_psi_fun
            icm("Initial condition set as a callable.")
        else:
            self.initial_psi = initial_psi_fun
            icm("Initial condition set as an array.")
        

    def set_laser_pulse(self, laser_pulse_fun: callable):
        """Set the laser pulse as a function, a function of time only. See also set_laser_potential.

        Args:
            laser_pulse_fun (callable): The laser pulse function. Must take 1 argument (time), return a scalar, and be vectorized.


        Returns:
            None
        """

        # # check number of args to laser_pulse_fun
        # if laser_pulse_fun.__code__.co_argcount != 1:
        #     raise ValueError("laser_pulse_fun must take 1 argument.")

        # if not check_function_signature(laser_pulse_fun, 1, 1):
        #     raise ValueError("laser_pulse_fun has wrong function signature.")

        self.laser_pulse_fun = laser_pulse_fun
        icm("Laser pulse set.")

    def set_laser_potential(self, laser_potential_fun: callable):
        """Set the laser potential as a function. Must accept 'dim' arguments. The
        total laser pulse potential is the product of the laser pulse function and
        the laser potential function. Usually, this is the dipole operator in some form.

        Args:
            laser_potential_fun (callable): The laser potential function. Must take 'dim' arguments (space), return a scalar, and be vectorized.


        Returns:
            None
        """

        # check number of args to laser_potential_fun
        # if not check_function_signature(laser_potential_fun, self.dim, 1):
        #     raise ValueError("laser_potential_fun has wrong function signature.")

        # if laser_potential_fun.__code__.co_argcount != self.dim:
        #     raise ValueError("laser_potential_fun must take 'dim' arguments.")

        self.laser_potential_fun = laser_potential_fun
        icm("Laser potential set.")
        
    def setup_hamiltonian(self):
        """Set up the hamiltonian. Used in the prepare method. Usually not called directly."""

        # check if grid is set
        if not hasattr(self, "grid"):
            raise ValueError("Grid not set.")

        # set up kinetic energy operator
        self.T_fun = lambda k: T_standard(k, mu=self.mass)

        # set default laser potential
        if not hasattr(self, "laser_potential_fun"):
            # set default laser potential
            if self.dim == 1:
                self.laser_potential_fun = lambda x: x
            elif self.dim == 2:
                self.laser_potential_fun = lambda x, y: x
            elif self.dim == 3:
                self.laser_potential_fun = lambda x, y, z: x
            elif self.dim == 4:
                self.laser_potential_fun = lambda x, y, z, w: x
                

        # set default laser pulse
        if not hasattr(self, "laser_pulse_fun"):
            self.laser_pulse_fun = lambda t: 0.0

        # set up FourierHamiltonian object.
        
        self.ham = FourierHamiltonian(
            self.grid,
            lambda xx: self.potential_fun(*xx),
            Tfun=self.T_fun,
            Dfun=lambda xx: self.charge * self.laser_potential_fun(*xx),
            Efun=self.laser_pulse_fun,
        )

    def prepare(self, normalize_wavefunction = True):
        """Prepare the simulation. This function is to be called after all parameters
        have been set, and before the simulation is run.

        If the initial condition is set, the ground state is *not* computed. If the
        initial condition is not set, the ground state is computed and used as the
        initial condition.

        If the ground state is computed, it is stored in the attribute 'gs', and hence not
        recomputed upon subsequent calls to prepare.

        Args:
            None
        Returns:
            None
        """

        # check if grid is set
        if not hasattr(self, "grid"):
            raise ValueError("Grid not set.")

        # check if time grid is set
        if not hasattr(self, "t_grid"):
            raise ValueError("Time grid not set.")

        # set up FourierHamiltonian object
        self.setup_hamiltonian()

        # compute the initial condition on the grid if
        # it is not already set or if the ground state is not set
        if hasattr(self, "initial_psi"):
            ic("Using given initial condition. ")
            self.wf = FourierWavefunction(self.grid, psi=self.initial_psi)
            
        elif hasattr(self, "initial_psi_fun"):
            
            ic("Using given initial condition function. ")
            psi = self.initial_psi_fun(*self.grid.xx)
            self.wf = FourierWavefunction(self.grid, psi=psi)
        elif hasattr(self, "gs"):
            # set the ground state wavefunction as initial condition
            ic("Reusing ground state from previous computation")
            if hasattr(self, "grid_gs"):
                # interpolate to the simulation grid
                self.wf.setPsi(self.gs.wf.interpolate(self.grid).psi, normalize=normalize_wavefunction)
            else:
                # set the wavefunction
                self.wf.setPsi(self.gs.wf.psi, normalize=normalize_wavefunction)
        else:
            ic("Computing ground state ... ")
            # compute ground state wavefunction of potential
            # self.wf becomes the ground state
            if hasattr(self, "ground_state_guess"):
                ic("Using guess for ground state ... ")
                self.compute_ground_state(guess=self.ground_state_guess)
            else:
                ic("Not using guess for ground state ... ")
                self.compute_ground_state()

        # set up a handy attribute for the user
        self.psi = self.wf.psi

        # set up laser pulse value,
        # handy for the user
        self.laser_value = self.laser_pulse_fun(self.t_grid[0])

        # set up propagator.
        self.prop = Propagator(self.ham, self.dt)

    def compute_ground_state(self, gs_tol=1e-12, guess=None):
        """Compute the ground state wavefunction of the potential. If the grid for the
        ground state computation is not set using 'set_ground_state_grid', the grid for the simulation is used.

        Args:
            gs_tol (float): The tolerance for the ground state computation.
            guess (np.ndarray, optional): An initial guess for the ground state wavefunction.

        Returns:
            None
        """

        # # set up ground state computer
        # gs = GroundStateComputer(self.ham)
        # grid_shape = self.grid.xx[0].shape
        # guess = np.random.rand(*grid_shape) - 0.5
        # gs.setInitialGuess(guess)

        # # compute ground state
        # E = gs.invit(sigma = np.min(self.ham.V), tol=gs_tol)

        # # extract the ground state wavefunction
        # self.wf = gs.wf

        if hasattr(self, "grid_gs"):
            ic("Computing ground state on separate grid ... ")
            grid = self.grid_gs
        else:
            ic("Computing ground state on main grid ... ")
            grid = self.grid

        xx = grid.xx

        # set up ground Hamiltonian on the grid
        ham = FourierHamiltonian(
            grid, Vfun=lambda xx: self.potential_fun(*xx), Tfun=self.T_fun
        )

        # Set up ground state computer
        gs = GroundStateComputer(ham)
        if guess is None:
            grid_shape = grid.xx[0].shape
            guess = np.random.rand(*grid_shape) - 0.5
        gs.setInitialGuess(guess)

        # compute ground state
        gs.invit(sigma=np.min(ham.V), tol=gs_tol)
        self.gs = gs  # save for future reference

        # Create a wavefunction object, and
        # set it as initial condition.
        self.wf = FourierWavefunction(self.grid)

        if hasattr(self, "grid_gs"):
            # interpolate to the simulation grid
            self.wf.setPsi(gs.wf.interpolate(self.grid).psi, normalize=True)
        else:
            # set the wavefunction
            self.wf.setPsi(gs.wf.psi, normalize=True)

    def time_step(self):
        """Performs the actual time step of the simulation. Can be modified
        by subclasses to implement different time stepping schemes.

        The only requirement is that the wavefunction is updated to the next
        time step, assuming that the current time is self.t.

        Args:
            None
        Returns:
            None
        """

        if self.propagation_method == 'strang-3':
            self.prop.strang(self.wf, self.t, will_do_another_step=False)
        elif self.propagation_method == 'crank-nicholson':
            self.prop.crank_nicholson(self.wf, self.t)
        else:
            raise ValueError('Invalid integration scheme')


    def simulate(self, callback: callable = None):
        """Run the simulation.

        Args:
            callback (callable): A callback function that is called at each time step.
                The callback function must accept the simulator instance as its only argument.

        Returns:
            None
        """

        ic("Running simulation...")

        self.t = self.t0
        self.t_index = 0

        for i in tqdm(range(self.n_steps)):
            self.t = self.t_grid[i]
            self.laser_value = self.laser_pulse_fun(self.t)

            if callback is not None:
                callback(self)

            # Do the time step
            self.time_step()
            self.t_index += 1

            # handy for the user
            self.psi = self.wf.psi

        # Main loop finished, set the final time
        # and do one more callback at the end.
        self.t = self.t_grid[-1]
        self.laser_value = self.laser_pulse_fun(self.t)

        if callback is not None:
            callback(self)
