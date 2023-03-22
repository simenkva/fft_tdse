#
# Simulation script for 4 particles in 1D.
#

import numpy as np
from fft_tdse import *
from psiviz import *
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from tqdm import tqdm
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
config = yaml.safe_load(open("config_2d.yml"))

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
    q = config['particles']['q']
except:
    q = [1.0, 1.0, -1.0, -1.0] # charges of particles
try:
    m = config['particles']['m']
except:
    m = [200, 100, 1, 1]  #masses of particles
try:
    temp = config['potential']['U']
    Ufun = eval(temp)
    if verbose:
        print(f'Model potential: {temp}')
except:
    Ufun = lambda x: np.exp(-0.1*x**2)
    if verbose:
        print(f'Model potential is default.')


try:
    ng = config['grid']['n']
except:
    ng = 128
try:
    L = config['grid']['L']
except:
    L = 50
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
    t_final = 1500
try:
    dt = config['integration']['dt']
except:
    dt = 0.1
try:
    n_inspect = config['integration']['n_inspect']
except:
    n_inspect = 100
try:
    n_save = config['integration']['n_save']
except:
    n_save = int(t_final)


try:
    # Set up laser field
    E0 = config['pulse']['E0']
    tau = config['pulse']['tau']
    om = config['pulse']['om']
    T = config['pulse']['T']
except:
    E0, tau, om, T = 0.25, 20.5, 1.0/(2*np.pi), 50


if verbose:
    print(f'Particle charges: {q}')
    print(f'Particle masses: {m}')
    print(f'Number of grid points: {ng}**3')
    print(f'Simulation domain: [-{L},{L}]**3')
    print(f'Number of grid points (ground-state calc): {gs_ng}**3')
    print(f'Simulation domain (ground-state calc): [-{gs_L},{gs_L}]**3')
    print(f'Termination tolerance for ground-state calc: {gs_tol}')
    print(f'Time interval: [0,{t_final}]')
    print(f'Time step: {dt}')
    print(f'Number of wavefunction saves: {n_save}')
    print(f'Number of density inspections: {n_inspect}')
    print(f'Laser parameters E0, tau, om, T = {E0, tau, om, T}')



#
# Set up system and simulation parameters
#

mu = [m[0]*m[1]/(m[0]+m[1]), m[0]*m[2]/(m[0]+m[2]), m[0]*m[3]/(m[0]+m[3])] # reduced masses

# manybody potentials
Vfun = lambda xx: q[0]*q[1]*Ufun(xx[0]) + q[0]*q[2]*Ufun(xx[1]) + q[0]*q[3]*Ufun(xx[2]) + q[1]*q[2]*Ufun(xx[0]-xx[1]) + q[1]*q[3]*Ufun(xx[0]-xx[2]) + q[2]*q[3]*Ufun(xx[1]-xx[2])
Tfun = lambda kk: (1.0/m[0]) * (kk[0]*kk[1] + kk[0]*kk[2] + kk[1]*kk[2]) + (0.5/mu[0])*kk[0]**2 + (0.5/mu[1])*kk[1]**2 + (0.5/mu[2])*kk[2]**2
Dfun = lambda xx: q[1]*xx[0] + q[2]*xx[1] * q[3]*xx[2]

# Set up grid
grid = FourierGrid([-L,-L,-L], [L,L,L], [ng, ng, ng])
#t_final = 100


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


#
# Compute ground state wavefunction
#

def compute_ground_state(L2,ng2,Tfun,Vfun):
    """ Compute ground state wavefunction on a suitable grid,
    and then extrapolate to the computationa grid. """

    grid2 = FourierGrid([-L2,-L2,-L2],[L2,L2,L2],[ng2,ng2,ng2])
    xx = grid2.xx

    ham = FourierHamiltonian(grid2, Tfun = Tfun, Vfun = Vfun)

    gs = GroundStateComputer(ham)
    gs.setInitialGuess(np.exp(-(xx[0]**2 + xx[1]**2 + xx[2]**2)/2))

    E = gs.invit(sigma = np.min(ham.V), tol=gs_tol)

    if figures:
        plt.figure()
        plt.plot(grid2.x[0],gs.wf.density(0))
        plt.plot(grid2.x[1],gs.wf.density(1)+.01)
        plt.plot(grid2.x[2],gs.wf.density(2)+.02)
        plt.legend(['rho0', 'rho1', 'rho2'])
        plt.xlabel('x')
        plt.title('densitites, vertically shifted for visiblity')
        plt.savefig(figname())
        plt.close()


    return gs.wf

# Compute ground state on a smaller grid, then extrapolate.
if verbose:
    print(f'Computing ground-state wavefunction by inverse iterations ...')
psi0 = compute_ground_state(gs_L,gs_ng,Tfun,Vfun)



def visualize(wf,heading):
    """ Visualize the wavefunction. """

    plt.figure()
    plt.plot(grid.x[0],wf.density(0))
    plt.plot(grid.x[1],wf.density(1)+.005)
    #plt.plot(grid.x[2],wf.density(2)+.02)
    plt.ylim([0,0.3])
    plt.legend(['rho0', 'rho1', 'rho2'])
    plt.xlabel('x')
    plt.title(heading)
    plt.savefig(figname())
    plt.close()


#
# Set up initial condition and propagator
#

if verbose:
    print(f'Setting up and running main simulation loop.')

# Create a wavefunction object
wf = FourierWavefunction(grid)
wf.setPsi(psi0.interpolate(grid).psi,normalize=True)
if figures:
    visualize(wf,'Initial condition')

# Set up Hamiltonian
ham = FourierHamiltonian(grid, Tfun=Tfun, Vfun = Vfun, Dfun = Dfun, Efun=Efun)

# Create a Strang splitting propagator
prop = Propagator(ham, dt)


# # Main simulation loop -- with data saving
#
# This is the main simulation loop. A HDF5 file is created, where all relevant simulation
# data is stored, including the grid and the laser parameters.  For every time step, densities and currents
# are calculated and saved. At every other time step, the complete wavefunction is also saved.
#
# For a large 3D calculation, quite a lot of data will be generated!
#

# Time range for simulation.
t_range = np.arange(0,t_final+dt,dt)

# Number of wavefunction saves
#n_save = int(t_final)

# Number of real-time inspections of results
#n_inspect = 100

# Buffers for saving complete density and current histories
dens_hist = np.zeros((len(grid.x[0]),len(t_range), 3), dtype=float)
curr_hist = np.zeros((len(grid.x[0]),len(t_range), 3), dtype=float)
energy_hist = np.zeros(len(t_range), dtype=float)

# si = index for saves
si = 0

# Create a file name
from datetime import datetime
fname = f'{sim_name}_{datetime.now().strftime("%d%m%Y_%H%M")}.h5'

# Open an h5 file
with h5py.File(fname,'w') as h5file:

    # Save simulation parameters.
    h5file.create_dataset('/parameters/time', data=t_range)
    h5file.create_dataset('/parameters/grid', data=grid.x)
    h5file.create_dataset('/parameters/hamiltonian/T', data = ham.T)
    h5file.create_dataset('/parameters/hamiltonian/V', data = ham.V)
    h5file.create_dataset('/parameters/hamiltonian/D', data = ham.D)
    h5file.create_dataset('/parameters/hamiltonian/E', data = Efun(t_range))

    # Loop over time steps
    for i in tqdm(range(len(t_range))):
        t = t_range[i]

        # Compute currents and densities
        for n in range(3):
            dens_hist[:,i,n] = wf.density(n)
            curr_hist[:,i,n] = wf.current(n)
        #h5file.create_dataset(f'/densities/{i}/rho', data = dens_hist[:,i,:],compression='gzip')
        #h5file.create_dataset(f'/currents/{i}/jp', data = curr_hist[:,i,:],compression='gzip')
        energy_hist[i] = ham.energy(wf, Efun(t)).real

        # Save wavefunction
        if i % int(t_final/n_save/dt) == 0:
            h5file.create_dataset(f'/wavefunctions/{si}/psi', data=wf.psi, compression='gzip')
            h5file.create_dataset(f'/wavefunctions/{si}/t', data=t)
            si += 1

        # Visualize in real time
        if i % int(t_final/n_inspect/dt) == 0:
            if figures:
                visualize(wf,f't = {t:.2f}, E = {Efun(t):.2f}, energy = {energy_hist[i]:.4f}')

        # Time step
        prop.strang(wf,t, will_do_another_step=False) # we need the updating of the fft for correct observables

    # Save energy history, density, and current
    h5file.create_dataset('/energy', data=energy_hist)
    h5file.create_dataset('/density', data=dens_hist,compression='gzip')
    h5file.create_dataset('/current', data=curr_hist,compression='gzip')


#
# ## Final visualization of densities as function of time
#

if figures:
    plt.figure(figsize=(12,8), dpi= 100)
    plt.imshow(dens_hist[:,:,0],aspect='auto',extent=[0,t_final,-L,L],cmap='jet')
    plt.colorbar()
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Density of pseudoparticle 0 (H nucleus)')
    plt.savefig(figname())
    plt.figure(figsize=(12,8), dpi= 100)
    plt.imshow(dens_hist[:,:,1],aspect='auto',extent=[0,t_final,-L,L],cmap='jet')
    plt.colorbar()
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Density of pseudoparticles 1 and 2 (electrons)')
    plt.savefig(figname())

if verbose:
    print('Done.')
    
