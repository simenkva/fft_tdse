import numpy as np
from fft_tdse.fft_tdse import *
from fft_tdse.psiviz import *
from fft_tdse.fouriergrid import FourierGrid
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('AGG')
from tqdm import tqdm
from scipy.io import savemat
import h5py

#
# figindex is a global counter for
# figures being saved to disk
#
figindex = 0


def figname():
    """ Return figure filename and advance counter. """
    global figindex
    figindex += 1
    return f'{sim_name}_fig_{figindex:03d}.png'

#
# Read configuration file
#

import yaml
config = yaml.safe_load(open("config1d.yml"))

try:
    verbose = config['verbose']
except:
    verbose = True


try:
    sim_name = config['sim_name']
except:
    sim_name = 'results'
if verbose:
    print(f'Simulation name: {sim_name}')

try:
    figures = config['figures']
except:
    figures = True
if verbose:
    print(f'Figures will be saved to disk: {figures}')


try:
    temp = config['potential']['U']
    Vfun0 = eval(temp)
    if verbose:
        print(f'Model potential: {temp}')
except:
    Vfun0 = lambda x: np.exp(-0.1*x**2)
    if verbose:
        print(f'Model potential is default.')


try:
    ng = config['grid']['n']
except:
    ng = 512
try:
    L = config['grid']['L']
except:
    L = 150
try:
    gs_ng = config['grid']['n_gs']
except:
    gs_ng = 64
try:
    gs_L = config['grid']['L_gs']
except:
    gs_L = 15
try:
    gs_tol = config['grid']['gs_tol']
except:
    gs_tol = 1e-7

try:
    t_final = config['integration']['t_final']
except:
    t_final = 100
try:
    dt = config['integration']['dt']
except:
    dt = 0.01
try:
    n_inspect = config['integration']['n_inspect']
except:
    n_inspect = 10


try:
    # Set up laser field
    E0 = config['pulse']['E0']
    tau = config['pulse']['tau']
    om = config['pulse']['om']
    T = config['pulse']['T']
except:
    E0, tau, om, T = 0.25, 20.5, 1.0/(2*np.pi), 50


if verbose:
    print(f'Number of grid points: {ng}**3')
    print(f'Simulation domain: [-{L},{L}]**3')
    print(f'Number of grid points (ground-state calc): {gs_ng}')
    print(f'Simulation domain (ground-state calc): [-{gs_L},{gs_L}]')
    print(f'Termination tolerance for ground-state calc: {gs_tol}')
    print(f'Time interval: [0,{t_final}]')
    print(f'Time step: {dt}')
    print(f'Number of density inspections: {n_inspect}')
    print(f'Laser parameters E0, tau, om, T = {E0, tau, om, T}')






# Set up a fairly large grid
#ng = 2048
#L = 400
grid = FourierGrid([-L],[L],[ng])
#t_final = 100

# Set up soft Coulomb potential
#delta = .01
#Vfun0 = lambda x: -1.0/(x**2+delta**2)**.5
Vfun = lambda xx: Vfun0(xx[0]) # the grid stores the nodes for each dimension in a list ...


# Fetch the nodes of the grid.
x = grid.x[0]


def compute_ground_state(L2,ng2,Vfun):
    """ Compute ground state wavefunction on a suitable grid,
    and then extrapolate to the computationa grid. """

    grid2 = FourierGrid([-L2],[L2],[ng2])
    x = grid2.x[0]

    ham = FourierHamiltonian(grid2, Vfun = Vfun)

    gs = GroundStateComputer(ham)
    gs.setInitialGuess(np.exp(-x**2/2))
    sigma = np.min(Vfun([x]))
    E = gs.invit(sigma = sigma)

    if figures:
        plt.figure()
        plt.plot(x,Vfun([x]),x,np.real(gs.wf.psi))
        plt.legend(['potential','ground state'])
        plt.xlabel('x')
        plt.savefig(figname())

    if verbose:
        print(f'Ground state energy computed: {E}')
    return gs.wf.interpolate(grid).psi

# Compute ground state on a smaller grid, then extrapolate.
psi0 = compute_ground_state(gs_L,gs_ng,Vfun)


# In[3]:


# Set up laser field
#E0, tau, om, T = 0.25, 20.5, 1.0/(2*np.pi), 50

Efun = lambda t: E0*np.exp(-(t-T)**2/tau**2) * np.cos(om*t)

if figures:
    plt.figure()
    t = np.linspace(0,t_final,200)
    plt.plot(t,Efun(t))
    plt.title('Electric field')
    plt.xlabel('t')
    plt.ylabel('E(t)')
    plt.savefig(figname())
    plt.close()


# In[4]:


def visualize(psi,heading):
    """ Visualize the wavefunction. """
    if figures:
        plt.figure()
        plt.plot(x,psi.real,'b',linewidth=.25)
        plt.plot(x,psi.imag,'r',linewidth=.25)
        plt.plot(x,np.abs(psi),'k',linewidth=.5)
        plt.legend(['Re','Im','abs'])
        plt.title(heading)
        plt.xlabel('x')
        plt.savefig(figname())
        plt.close()



# # Set up initial condition and propagator

# In[5]:


# Set up initial condition.
visualize(psi0,'Initial condition')

# Create a wavefunction object
wf = FourierWavefunction(grid)
wf.setPsi(psi0,normalize=True)

# Set up Hamiltonian
ham = FourierHamiltonian(grid, Vfun = Vfun, Efun=Efun)

# Create a Strang splitting propagator
#dt = 0.01
prop = Propagator(ham, dt)


# # Main simulation loop
#
# The propagation is here. We do a visualization once in a while.

# In[6]:

if verbose:
    print('Now propagating in time ...')

t_range = np.arange(0,t_final+dt,dt)
psi_hist = np.zeros((len(x),len(t_range)), dtype=complex)
E_hist = np.zeros(len(t_range), dtype=float)
for i in tqdm(range(len(t_range)-1)):
    t = t_range[i]
    psi_hist[:,i] = wf.psi
    E_hist[i] = ham.energy(wf, Efield = Efun(t)).real
    if i % int(t_final/n_inspect/dt) == 0:
        visualize(wf.psi,f't = {t:.2f}, E = {Efun(t):.2f}')
    prop.strang(wf,t)

visualize(wf.psi,f't = {t_final:.2f}, E = {Efun(t_final):.2f}')
E_hist[-1] = ham.energy(wf, Efield = Efun(t_final)).real
psi_hist[:,-1] = wf.psi

# ## Visualize entire history
#
# We visualize the entire history of the simulation. The colors represent the phase of the wavefunction.
#

if figures:
    plt.figure()
    plt.plot(t_range,E_hist)
    plt.title('Energy')
    plt.ylabel('E(t)')
    plt.xlabel('t')
    plt.savefig(figname())

    I = psiviz(psi_hist,normalize_columns=True)
    fig=plt.figure(figsize=(12,8), dpi= 100, facecolor='w', edgecolor='k')
    plt.imshow(I[::2,::50],aspect='auto',extent=[0,t_final,-L,L])
    plt.xlabel('t')
    plt.ylabel('y')
    plt.savefig(figname())
    plt.close()


# # Save simulation results
# matlab_output = {'x': x, 't': t_range, 'psi': psi_hist, 'E': E_hist}
# savemat(f'{sim_name}.mat', matlab_output)


# Open an h5 file
with h5py.File(f'{sim_name}.hdf5','w') as h5file:

    # Save simulation parameters.
    h5file.create_dataset('/parameters/time', data=t_range)
    h5file.create_dataset('/parameters/grid', data=x)
    h5file.create_dataset('/wavefunctions/psigrid_re', data = psi_hist.real, compression = 'gzip')
    h5file.create_dataset('/wavefunctions/psigrid_im', data = psi_hist.imag, compression = 'gzip')
    h5file.create_dataset('/computed/energy', data = E_hist)

