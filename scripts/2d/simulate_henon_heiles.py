#
# Simulation script for 1 particle in 2D
#


import numpy as np
from fft_tdse import *
from potentials import *
from psiviz import *
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import argparse

#
# Command line arguments
#
parser = argparse.ArgumentParser(
                    prog = 'simulate_henon_heiles.py',
                    description = '2d time-dependent Schrödinger equation solver',
                    epilog = '')

parser.add_argument('-c', '--config', default = 'config_henon_heiles.yml', required=False, help = 'YAML config file to use.')

args = parser.parse_args()

config_filename = args.config


#
# figindex is a global counter for
# figures being saved to disk
#
figindex = 0
def figname():
    """ Return figure filename and advance counter. """
    global figindex
    figindex += 1
    return f'{sim_name}_fig_{figindex:05d}.png'

#
# Read configuration file
#

import yaml
config = yaml.safe_load(open(config_filename))

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
    m = config['particles']['m']
except:
    m = 1.0  #masses of particles
q = 1.0

try:
    ell = config['potential']['ell']
except:
    ell = 1.0
Ufun = lambda x, y: .5 * (x**2 + y**2) + ell * (x**2 * y - y**3 / 3)
print(f'Model potential is Henon-Heiles with lambda = {ell}')

# parameters for coherent state initial condition
try:
    x0 = config['initial']['x0']
except:
    x0 = 0.0
try:
    y0 = config['initial']['y0']
except:
    y0 = 0.0
try:
    px0 = config['initial']['px0']
except:
    px0 = 0.0
try:
    py0 = config['initial']['py0']
except:
    py0 = 0.0
try:
    sigma0 = config['initial']['sigma0']
except:
    sigma0 = 1.0


try:
    ng = config['grid']['n']
except:
    ng = 512
try:
    L = config['grid']['L']
except:
    L = 20
# try:
#     gs_ng = config['grid']['n_gs']
# except:
#     gs_ng = 64
# try:
#     gs_L = config['grid']['L_gs']
# except:
#     gs_L = 15
# try:
#     gs_tol = config['grid']['gs_tol']
# except:
#     gs_tol = 1e-7

try:
    t_final = config['integration']['t_final']
except:
    t_final = 50
try:
    dt = config['integration']['dt']
except:
    dt = 0.01
try:
    n_inspect = config['integration']['n_inspect']
except:
    n_inspect = 100
try:
    n_save = config['integration']['n_save']
except:
    n_save = int(t_final)


# try:
#     # Set up laser field
#     E0 = config['pulse']['E0']
#     om = config['pulse']['om']
#     T = config['pulse']['T']
#     t0 = config['pulse']['t0']
# except:
#     E0, om, T, t0 = 0.5, 0.482681, 2*np.pi*100.0/0.482681, 0


if verbose:
    print(f'Particle charges: {q}')
    print(f'Particle masses: {m}')
    print(f'Number of grid points: {ng}**2')
    print(f'Simulation domain: [-{L},{L}]**2')
    #print(f'Number of grid points (ground-state calc): {gs_ng}**2')
    #print(f'Simulation domain (ground-state calc): [-{gs_L},{gs_L}]**2')
    #print(f'Termination tolerance for ground-state calc: {gs_tol}')
    print(f'Time interval: [0,{t_final}]')
    print(f'Time step: {dt}')
    print(f'Number of wavefunction saves: {n_save}')
    print(f'Number of density inspections: {n_inspect}')
    #print(f'Laser parameters E0, om, t0, T = {E0, om, t0, T}')
    print(f'Initial state is a coherent state:')
    print(f'  pos = ({x0,y0}), momentum = ({px0,py0}), sigma = ({sigma0})')



#
# Set up system and simulation parameters
#

# Hamiltonian specification
Vfun = lambda xx: Ufun(xx[0], xx[1])
Tfun = lambda kk: (0.5/m) * (kk[0]**2 + kk[1]**2)
#Dfun = lambda xx: q*xx[0] 

# Set up grid
grid = FourierGrid([-L,-L], [L,L], [ng, ng])
#t_final = 100


#Efun = lambda t: E0*np.exp(-(t-T)**2/tau**2) * np.cos(om*t)

# def Gfun0(t):
#     if t <= t0 or t >= T+t0:
#         return 0.0
#     else:
#         return np.sin(np.pi*(t-t0)/T)**2

# Gfun = np.vectorize(Gfun0)
# Efun = lambda t: E0 * Gfun(t) * np.cos(om*(t-(t0 + T/2)))

# if figures:
#     plt.figure()
#     t = np.linspace(0,t_final,int(t_final/dt)+1)
#     plt.plot(t,Efun(t))
#     plt.title('Electric field')
#     plt.xlabel('t')
#     plt.ylabel('E(t)')
#     plt.savefig(figname())
#     plt.close()




def visualize(wf,heading, L = L):
    """ Visualize a wavefunction. """

    plt.figure()
    
    m = np.abs(wf.psi).max()
    bm = mag_vis(wf.psi.T, mag_map=lambda r: np.sqrt(r/m))
    plt.imshow(bm, aspect='equal', cmap = colorcet.cm['bgy'], extent = [-L,L,-L,L], origin = 'lower')
    plt.colorbar()

    #bm = render2d(wf)
    #plt.imshow(bm.T, aspect='equal', extent = [-L,L,-L,L], origin = 'lower')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(heading)
    plt.savefig(figname(), dpi = 100)
    plt.close()

#
# Compute ground state wavefunction
#

# def compute_ground_state(L2,ng2,Tfun,Vfun):
#     """ Compute ground state wavefunction on a suitable grid,
#     and then extrapolate to the computationa grid. """

#     grid2 = FourierGrid([-L2,-L2],[L2,L2],[ng2,ng2])
#     xx = grid2.xx

#     ham = FourierHamiltonian(grid2, Tfun = Tfun, Vfun = Vfun)

#     gs = GroundStateComputer(ham)
#     gs.setInitialGuess(np.exp(-(xx[0]**2 + xx[1]**2)/2))

#     E = gs.invit(sigma = np.min(ham.V), tol=gs_tol)

#     if figures:
#         visualize(gs.wf, f'Ground-state wavefunction, E = {E}', L = L2)

#         plt.figure()
#         plt.imshow(ham.V, extent = [-L2, L2, -L2, L2], cmap = 'jet')
#         plt.title('Potential')
#         plt.colorbar()
#         plt.savefig(figname())
#         plt.close()


#     return gs.wf

# Compute ground state on a smaller grid, then extrapolate.
# if verbose:
#     print(f'Computing ground-state wavefunction by inverse iterations ...')
# psi0 = compute_ground_state(gs_L,gs_ng,Tfun,Vfun)

x = grid.xx[0]
y = grid.xx[1]
psi0 = np.exp(-0.5*sigma0*((x-x0)**2 + (y-y0)**2) + 1j*px0*(x-x0) + 1j*py0*(y-y0))




#
# Set up initial condition and propagator
#

if verbose:
    print(f'Setting up and running main simulation loop.')

# Create a wavefunction object
wf = FourierWavefunction(grid)
wf.setPsi(psi0,normalize=True)
if figures:
    visualize(wf,'Initial condition')

# Set up Hamiltonian
ham = FourierHamiltonian(grid, Tfun=Tfun, Vfun = Vfun)

if figures:
    plt.figure()
    plt.imshow(ham.V.T, vmin = -2, vmax = 2, extent = [-L, L, -L, L], cmap = 'PiYG', origin='lower')
    plt.title(f'Henon-Heiles potential with lambda = {ell}')
    plt.colorbar()
    plt.savefig(figname())
    plt.close()

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
dens_hist = np.zeros((len(grid.x[0]),len(t_range), 2), dtype=float)
curr_hist = np.zeros((len(grid.x[0]),len(t_range), 2), dtype=float)
energy_hist = np.zeros(len(t_range), dtype=float)

# si = index for saves
si = 0

# Create a file name
from datetime import datetime
#fname = f'{sim_name}_{datetime.now().strftime("%d%m%Y_%H%M")}.h5'
fname = f'{sim_name}.hdf5'

# Open an h5 file
with h5py.File(fname,'w') as h5file:

    # Save simulation parameters.
    h5file.create_dataset('/parameters/time', data=t_range)
    h5file.create_dataset('/parameters/grid', data=grid.x)
    h5file.create_dataset('/parameters/hamiltonian/T', data = ham.T)
    h5file.create_dataset('/parameters/hamiltonian/V', data = ham.V)

    # Loop over time steps
    for i in tqdm(range(len(t_range))):
        t = t_range[i]

        # Compute currents and densities
        for n in range(2):
            dens_hist[:,i,n] = wf.density(n)
            curr_hist[:,i,n] = wf.current(n)
        #h5file.create_dataset(f'/densities/{i}/rho', data = dens_hist[:,i,:],compression='gzip')
        #h5file.create_dataset(f'/currents/{i}/jp', data = curr_hist[:,i,:],compression='gzip')
        energy_hist[i] = ham.energy(wf).real

        # Save wavefunction
        if i % int(t_final/n_save/dt) == 0:
            h5file.create_dataset(f'/wavefunctions/{si}/psi', data=wf.psi, compression='gzip')
            h5file.create_dataset(f'/wavefunctions/{si}/t', data=t)
            si += 1

        # Visualize in real time
        if i % int(t_final/n_inspect/dt) == 0:
            if figures:
                visualize(wf,f't = {t:.2f}, energy = {energy_hist[i]:.4f}')

        # Time step
        prop.strang(wf,t, will_do_another_step=False) # we need the updating of the fft for correct observables

    # Save energy history, density, and current
    h5file.create_dataset('/energy', data=energy_hist)
    h5file.create_dataset('/density', data=dens_hist,compression='gzip')
    h5file.create_dataset('/current', data=curr_hist,compression='gzip')


#
# ## Final visualization of densities as function of time
#

# if figures:
#     plt.figure(figsize=(12,8), dpi= 100)
#     plt.imshow(dens_hist[:,:,0],aspect='auto',extent=[0,t_final,-L,L],cmap='jet')
#     plt.colorbar()
#     plt.xlabel('t')
#     plt.ylabel('x')
#     plt.title('Density of pseudoparticle 0 (H nucleus)')
#     plt.savefig(figname())
#     plt.figure(figsize=(12,8), dpi= 100)
#     plt.imshow(dens_hist[:,:,1],aspect='auto',extent=[0,t_final,-L,L],cmap='jet')
#     plt.colorbar()
#     plt.xlabel('t')
#     plt.ylabel('x')
#     plt.title('Density of pseudoparticles 1 and 2 (electrons)')
#     plt.savefig(figname())

if verbose:
    print('Done.')
    
