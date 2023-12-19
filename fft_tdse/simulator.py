
import numpy as np
from .fft_tdse import *
from .fouriergrid import FourierGrid
from tqdm import tqdm
from icecream import ic

from IPython import get_ipython

def icm(message):
    ic(message)
    
    
def is_notebook():
    """
    Check if the code is running in a Jupyter Notebook or not.

    Returns:
        bool: True if running in a Jupyter Notebook, False otherwise.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
        elif shell == 'TerminalInteractiveShell':
            return False
        else:
            return False
    except NameError:
        return False

# choose tqdm implementation
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm import tqdm as tqdm_console
tqdm = tqdm_notebook if is_notebook() else tqdm_console


class LaserPulse:
    """ A class for defining laser pulses. The general form of the pulse is
    
    $$
    E(t) = E_0 \cdot f(t) \cdot \cos(\omega \cdot (t - t_0 - \frac{T}{2}))
    $$
    
    where $f(t)$ is the envelope function, given by
    
    $$ f(t) = \sin^2\left(\frac{\pi \cdot (t - t_0)}{T}\right). $$
    
    
    Members:
        omega (float): The eldritch frequency that governs the pulse.
        t0 (float): The moment when the pulse emerges from the abyss.
        T (float): The duration of the pulse, a fleeting glimpse into the unknown.
        E0 (float): The amplitude of the pulse, a measure of its unfathomable power.
    Functions:
        envelope (float): The envelope function of the laser pulse.
        __call__ (float): The laser pulse.
    
    """

    def __init__(self, omega, t0, T, E0):
        """Initialize a laser pulse.

        Args:
            omega (float): The eldritch frequency that governs the pulse.
            t0 (float): The moment when the pulse emerges from the abyss.
            T (float): The duration of the pulse, a fleeting glimpse into the unknown.
            E0 (float): The amplitude of the pulse, a measure of its unfathomable power.
        """
        self.omega = omega
        self.t0 = t0
        self.T = T
        self.E0 = E0
        self.envelope = np.vectorize(self.envelope)

    def envelope(self, t):
        """The envelope function of the laser pulse.

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
        return self.E0 * self.envelope(t) * np.cos(self.omega * (t - (self.t0 + self.T / 2)))
    
        

class Simulator:
    """ A class for running simulations in 1d and 2d.
    
    The class defines a number of methods for setting up and running simulations.
    It provides a convenient interface to the FFT-TDSE code, and allows the user
    to easily read various variables during simulation using a callback feature.
    
    Here is an example of how to use the simulator:
    
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
        if dim not in {1, 2, 3}:
            raise ValueError("Dimension not supported.")

        self.dim = dim
        
        icm('Dimension set to {}'.format(self.dim))
        
    
    def set_grid(self, a, b, n):
        """
        Set the grid for the simulation.

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
            
            
    def set_ground_state_grid(self, a_gs, b_gs, n_gs):
        """
        Set the grid for the ground state computation.

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
        
        assert(t0 < t1)
        assert(n_steps > 0)
                
        self.t0 = t0
        self.t1 = t1
        self.n_steps = n_steps
        
        self.t_grid = np.linspace(t0, t1, n_steps + 1)
        self.t = t0
        self.dt = self.t_grid[1] - self.t_grid[0]
        
        
        ic('Time grid:')
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
            potential (callable): The potential function, assumed vectorized. Must take 'dim' arguments.

            
        Returns:
            None
        """

        # check number of args to potential
        if potential_fun.__code__.co_argcount != self.dim:
            raise ValueError("Potential must take {} arguments.".format(self.dim))  
        
        self.potential_fun = potential_fun
        
    def set_mass(self, mass : float):
        """ Set the particle mass. 
        
        Args:
            mass (float): The particle mass.
            
        Returns:
            None
        """
        
        self.mass = mass


    def set_charge(self, charge : float):
        """ Set the particle charge. 
        
        Args:
            charge (float): The particle charge.
            
        Returns:
            None
        """
        
        self.charge = charge
        

    def set_initial_condition(self, initial_psi_fun : callable):
        """Set the initial condition as a function.
        
        Args:
            initial_psi_fun (callable): The initial condition function, assumed vectorized. Must take 'dim' arguments.
    
        Returns:
            None
        """


        # check number of args to initial_psi_fun
        if initial_psi_fun.__code__.co_argcount != self.dim:
            raise ValueError("initial_psi_fun must take {} arguments.".format(self.dim))  
        
        self.initial_psi_fun = initial_psi_fun
        
        icm('Initial condition set.')
        ic(self.initial_psi_fun)
        
    def set_laser_pulse(self, laser_pulse_fun : callable):
        """Set the laser pulse as a function, a function of time only. See also set_laser_potential.
        
        Args:
            laser_pulse_fun (callable): The laser pulse function. Must take 1 argument (time).
            
    
        Returns:
            None
        """


        # # check number of args to laser_pulse_fun
        # if laser_pulse_fun.__code__.co_argcount != 1:
        #     raise ValueError("laser_pulse_fun must take 1 argument.")  
        
        self.laser_pulse_fun = laser_pulse_fun
        
    def set_laser_potential(self, laser_potential_fun : callable):
        """Set the laser potential as a function. Must accept 'dim' arguments. The
        total laser pulse potential is the product of the laser pulse function and
        the laser potential function.
        
        Args:
            laser_potential_fun (callable): The laser potential function. Must take 'dim' arguments (space).
            
    
        Returns:
            None
        """


        # check number of args to laser_potential_fun
        if laser_potential_fun.__code__.co_argcount != self.dim:
            raise ValueError("laser_potential_fun must take 'dim' arguments.")  
        
        self.laser_potential_fun = laser_potential_fun
        
    def prepare(self):
        """ Prepare the simulation. This function is called after all parameters
        have been set, and before the simulation is run.
        
        Args:
            None
        Returns:
            None
        """
        
        # check if grid is set
        if not hasattr(self, 'grid'):
            raise ValueError("Grid not set.")
        
        # check if time grid is set
        if not hasattr(self, 't_grid'):
            raise ValueError("Time grid not set.")
                
        # set up kinetic energy operator
        self.T_fun = lambda k: T_standard(k, mu=self.mass)
        
        # set default laser potential 
        if not hasattr(self, 'laser_potential_fun'):
            # set default laser potential
            if self.dim == 1:
                self.laser_potential_fun  = lambda x: x
            elif self.dim == 2:
                self.laser_potential_fun  = lambda x, y: x
            elif self.dim == 3:
                self.laser_potential_fun  = lambda x, y, z: x
    
        # set default laser pulse
        if not hasattr(self, 'laser_pulse_fun'):
            self.laser_pulse_fun = lambda t: 0.0
            
    
        self.ham = FourierHamiltonian(
            self.grid, 
            lambda xx: self.potential_fun(*xx), 
            Tfun=self.T_fun, 
            Dfun=lambda xx: self.charge*self.laser_potential_fun(*xx),
            Efun=self.laser_pulse_fun
        )
        
        ic(hasattr(self, 'gs'))

        # compute the initial condition on the grid if
        # it is not already set or if the ground state is not set
        if hasattr(self, 'initial_psi_fun'):
            psi = self.initial_psi_fun(*self.grid.xx)
            self.wf = FourierWavefunction(self.grid, psi=psi)
        elif hasattr(self, 'gs'):
            # set the ground state wavefunction as initial condition
            ic('reusing ground state from previous computation')
            if hasattr(self, 'grid_gs'):
                # interpolate to the simulation grid
                self.wf.setPsi(
                    self.gs.wf.interpolate(self.grid).psi,
                    normalize=True
                )
            else:
                # set the wavefunction
                self.wf.setPsi(
                    self.gs.wf.psi,
                    normalize=True
                )
        else:
            ic('computing ground state ... ')
            # compute ground state wavefunction of potential
            # self.wf becomes the ground state
            if hasattr(self, 'ground_state_guess'):
                ic('using guess for ground state ... ')
                self.compute_ground_state(guess=self.ground_state_guess)
            else:
                ic('not using guess for ground state ... ')
                self.compute_ground_state()
            
        # set up a handy attribute for the user
        self.psi = self.wf.psi
        
        # set up propagator
        self.prop = Propagator(self.ham, self.dt)
            
    def compute_ground_state(self, gs_tol=1e-12, guess = None):
        """Compute the ground state wavefunction of the potential. If the grid for the
        ground state computation is not set using 'set_ground_state_grid', the grid for the simulation is used.
        
        Args:
            gs_tol (float): The tolerance for the ground state computation.
            
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

        if hasattr(self, 'grid_gs'):
            grid = self.grid_gs
        else:
            grid = self.grid
            
        xx = grid.xx

        ham = FourierHamiltonian(
            grid, 
            Vfun=lambda xx: self.potential_fun(*xx), 
            Tfun=self.T_fun
        )

        gs = GroundStateComputer(ham)
        if guess is None:
            grid_shape = grid.xx[0].shape
            guess = np.random.rand(*grid_shape) - 0.5
        gs.setInitialGuess(guess)

        #gs.setInitialGuess(np.exp(-(xx[0]**2 + xx[1]**2)/2))

        gs.invit(sigma = np.min(ham.V), tol=gs_tol)
        self.gs = gs # save for future reference
        
        ic(hasattr(self, 'gs'))
           
        # Create a wavefunction object
        self.wf = FourierWavefunction(self.grid)
        
        
        if hasattr(self, 'grid_gs'):
            # interpolate to the simulation grid
            self.wf.setPsi(
                gs.wf.interpolate(self.grid).psi,
                normalize=True
            )
        else:
            # set the wavefunction
            self.wf.setPsi(
                gs.wf.psi,
                normalize=True
            )

            
    def simulate(self, callback: callable=None):
        """Run the simulation.
        
        Args:
            callback (callable): A callback function that is called at each time step.
                The callback function must accept the simulator instance as its only argument.
                
        Returns:
            None
        """
        
        ic('Running simulation...')
        
        self.t = self.t0
        self.t_index = 0

        if callback is not None:
            callback(self)

        for i in tqdm(range(self.n_steps)):
            self.t_index = i
            self.t = self.t_grid[i]
            self.prop.strang(self.wf,self.t,will_do_another_step=False)
            
            
            # handy for the user
            self.psi = self.wf.psi

            if callback is not None:
                callback(self)
        
    
    
    