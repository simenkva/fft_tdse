#
# Simulation script for 1 particle in 2D
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
# Morse potential
#
def morse(r, a = 1.0, D = 1.0, r_e = 1.0):
    return D * (1.0 - np.exp(-a * (r-r_e)))**2 - D

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
    q = 1.0 # charges of particles
try:
    m = config['particles']['m']
except:
    m = 1.0  #masses of particles
try:
    temp = config['potential']['U']
    Ufun = eval(temp)
    if verbose:
        print(f'Model potential: {temp}')
except:
    Ufun = lambda x, y: .5 * (x**2 + y**2)
    if verbose:
        print(f'Model potential is default.')


try:
    ng = config['grid']['n']
except:
    ng = 512
try:
    L = config['grid']['L']
except:
    L = 20
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


try:
    # Set up laser field
    E0 = config['pulse']['E0']
    om = config['pulse']['om']
    T = config['pulse']['T']
    t0 = config['pulse']['t0']
except:
    E0, om, T, t0 = 0.05, 0.482681, 2*np.pi*100.0/0.482681, 0


if verbose:
    print(f'Particle charges: {q}')
    print(f'Particle masses: {m}')
    print(f'Number of grid points: {ng}**2')
    print(f'Simulation domain: [-{L},{L}]**2')
    print(f'Number of grid points (ground-state calc): {gs_ng}**2')
    print(f'Simulation domain (ground-state calc): [-{gs_L},{gs_L}]**2')
    print(f'Termination tolerance for ground-state calc: {gs_tol}')
    print(f'Time interval: [0,{t_final}]')
    print(f'Time step: {dt}')
    print(f'Number of wavefunction saves: {n_save}')
    print(f'Number of density inspections: {n_inspect}')
    print(f'Laser parameters E0, om, t0, T = {E0, om, t0, T}')



#
# Set up system and simulation parameters
#

# Hamiltonian specification
Vfun = lambda xx: Ufun(xx[0], xx[1])
Tfun = lambda kk: (0.5/m) * (kk[0]**2 + kk[1]**2)
Dfun = lambda xx: q*xx[0] 

# Set up grid
grid = FourierGrid([-L,-L], [L,L], [ng, ng])
#t_final = 100


#Efun = lambda t: E0*np.exp(-(t-T)**2/tau**2) * np.cos(om*t)

def Gfun0(t):
    if t <= t0 or t >= T-t0:
        return 0.0
    else:
        return np.sin(np.pi*(t-t0)/T)**2

Gfun = np.vectorize(Gfun0)
Efun = lambda t: Gfun(t) * np.sin(om*(t-t0))

if figures:
    plt.figure()
    t = np.linspace(0,t_final,int(t_final/dt)+1)
    plt.plot(t,Efun(t))
    plt.title('Electric field')
    plt.xlabel('t')
    plt.ylabel('E(t)')
    plt.savefig(figname())
    plt.close()


def render2d(wf, cmap_name = 'hsv'):
    """ Render the wavefunction using a simple diffuse lightning model."""

    alpha = 200
    H = lambda x: 0.5 + 0.5 * np.tanh(alpha*x)

    fft2 = np.fft.fft2
    ifft2 = np.fft.ifft2
    light = np.array([1, .25, 1.5])
    light = light / np.linalg.norm(light)
    height_scale = 3

    psi = wf.psi
    (nx,ny) = psi.shape

    #phi = np.abs(psi)
    ang = np.angle(psi) / (2*np.pi) + 0.5
    #print(f'? angle in range [{np.min(ang)}, {np.max(ang)}]')
    #phi = (np.arctan2(psi.imag,psi.real) + np.pi)/(2*np.pi)
    cmap = get_cmap(cmap_name)
    # image1 = each pixel colored according to phase angle
    #colors = cmap(phi)[:,:,:3]
    colors = cmap(ang)[:,:,:3]

    rho = height_scale * np.abs(psi)
    rho_x = -ifft2(wf.grid.kk[0] * fft2(rho)).imag
    rho_y = -ifft2(wf.grid.kk[1] * fft2(rho)).imag
    t1 = np.zeros((nx,ny,3))
    t2 = np.zeros((nx,ny,3))
    n1 = (1 + rho_x**2)**.5
    n2 = (1 + rho_y**2)**.5
    t1[:,:,0] = 1.0/n1
    t1[:,:,2] = rho_x/n1
    t2[:,:,1] = 1.0/n2
    t2[:,:,2] = rho_y/n2
    n = np.cross(t1,t2)

    i = np.einsum('ijk,k->ij',n,light)
    i = (i > 0) * i


    sat = H(rho/np.max(rho)-0.001)

    image = colors.copy()
    white = np.ones((nx,ny,3))

    for k in range(3):
        image[:,:,k] = (sat * colors[:,:,k] + (1-sat) * white[:,:,k]) * i
        #image[:,:,k] = sat * white[:,:,k]

    return(image)


def visualize(wf,heading):
    """ Visualize a wavefunction. """

    plt.figure()
    #bm = render2d(wf)
    bm = np.abs(wf.psi.T)
    plt.imshow(bm, aspect='equal', cmap = 'jet', extent = [-L,L,-L,L], origin = 'lower')
    plt.imshow(bm, aspect='equal', extent = [-L,L,-L,L], origin = 'lower')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(heading)
    plt.savefig(figname())
    plt.close()

#
# Compute ground state wavefunction
#

def compute_ground_state(L2,ng2,Tfun,Vfun):
    """ Compute ground state wavefunction on a suitable grid,
    and then extrapolate to the computationa grid. """

    grid2 = FourierGrid([-L2,-L2],[L2,L2],[ng2,ng2])
    xx = grid2.xx

    ham = FourierHamiltonian(grid2, Tfun = Tfun, Vfun = Vfun)

    gs = GroundStateComputer(ham)
    gs.setInitialGuess(np.exp(-(xx[0]**2 + xx[1]**2)/2))

    E = gs.invit(sigma = np.min(ham.V), tol=gs_tol)

    if figures:
        visualize(gs.wf, f'Ground-state wavefunction, E = {E}')

        plt.figure()
        plt.imshow(ham.V, extent = [-L2, L2, -L2, L2], cmap = 'jet')
        plt.title('Potential')
        plt.colorbar()
        plt.savefig(figname())
        plt.close()


    return gs.wf

# Compute ground state on a smaller grid, then extrapolate.
if verbose:
    print(f'Computing ground-state wavefunction by inverse iterations ...')
psi0 = compute_ground_state(gs_L,gs_ng,Tfun,Vfun)




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
dens_hist = np.zeros((len(grid.x[0]),len(t_range), 2), dtype=float)
curr_hist = np.zeros((len(grid.x[0]),len(t_range), 2), dtype=float)
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
        for n in range(2):
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
    
