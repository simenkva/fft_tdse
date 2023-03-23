import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import colorcet
import subprocess
from colorspacious import cspace_convert
from colorsys import hsv_to_rgb
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from psiviz import phase_mag_vis, mag_vis
from scipy.interpolate import RegularGridInterpolator, interp2d

vis_types = ['magnitude', 'complex']

my_hsv_to_rgb = lambda h,s,v: np.array(np.vectorize(hsv_to_rgb)(h,s,v))

folder = 'out'
casename = 'sim_2_0'

sim_name = folder + '/' + casename
fname = f'{sim_name}.hdf5'


vis_type = 'complex'

moviename = casename + '_movie_' + vis_type


figsize = 5
imgsize = 1024
fig, ax = plt.subplots(1, 1, figsize = (figsize,figsize))
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

zoom_factor = 1
#interp_factor = 2
dens_factor = .75

font = {'family': 'JetBrains Mono'}
font_inset = {'family': 'JetBrains Mono', 'size': 10}
# inset plot
ax2 = plt.axes([0,0,1,1])
ip = InsetPosition(ax, [0.68,0.8,0.3,0.2])
ax2.set_axes_locator(ip)



with h5py.File(fname,'r') as h5file:
    frames = []

    wfs = h5file['wavefunctions']
    wf_idx = [int(key) for key in wfs.keys()]
    nmax = max(wf_idx) + 1

    t_range = np.squeeze(np.asarray(h5file['parameters/time']))
    grid = np.squeeze(np.asarray(h5file['parameters/grid']))
    x_range = grid[0,:]
    y_range = grid[1,:]
    x_min_zoom = np.min(x_range) / zoom_factor
    x_max_zoom = np.max(x_range) / zoom_factor
    y_min_zoom = np.min(y_range) / zoom_factor
    y_max_zoom = np.max(y_range) / zoom_factor

    xpix = len(x_range)
    ypix = len(y_range)
    xx,yy = np.meshgrid(x_range, y_range, indexing='ij')

    r = (xx*xx + yy*yy)**.5

    xpixi = imgsize
    ypixi = imgsize
    x_rangei = np.linspace(x_range.min() / zoom_factor, x_range.max() / zoom_factor, xpixi, endpoint=True)
    y_rangei = np.linspace(y_range.min() / zoom_factor, y_range.max() / zoom_factor, ypixi, endpoint=True)

    xxi, yyi = np.meshgrid(x_rangei, y_rangei, indexing='ij')

    #hsv = np.zeros((xpix,xpix,3))
    bmp = np.zeros((xpixi, ypixi, 3))


    Efield_vec = np.squeeze(np.asarray(h5file['parameters/hamiltonian/E']))
    energy_vec = np.squeeze(np.asarray(h5file['energy']))

    for i in tqdm(range(nmax)):
        t = np.asarray(wfs[f'{i}']['t'])
        psi = np.asarray(wfs[f'{i}']['psi'])

        t_idx = np.where(t_range == t)[0][0]
        Efield = Efield_vec[t_idx]
        energy = energy_vec[t_idx]

        if i == 0:
            max_dens = np.abs(psi).max() * dens_factor
            print(f'max_dens set to {max_dens}.')

        interpr = interp2d(x_range, y_range, psi.real, kind='cubic')
        interpi = interp2d(x_range, y_range, psi.imag, kind='cubic')

        # interp = RegularGridInterpolator((x_range, y_range), psi, bounds_error=False, fill_value=None, method='linear')
        # rho = np.abs(psi).clip(0, max_dens) / max_dens 
        # hue = np.angle(psi, deg = True) / 360.0 + .5
        # hsv[:,:,0] = hue.T
        # hsv[:,:,1] = 1.0
        # hsv[:,:,2] = rho.T
        
        # bmp[:,:,0], bmp[:,:,1], bmp[:,:,2] = np.vectorize(hsv_to_rgb)(hsv[:,:,0], hsv[:,:,1], hsv[:,:,2])


        psii = interpr(x_rangei,y_rangei) + 1j*interpi(x_rangei,y_rangei)
        
        if vis_type == 'complex':
            def mag_map(r):
                temp = r / max_dens

                temp = temp**.5

                return temp
            bmp = phase_mag_vis(psii.T, mag_map=mag_map)
        if vis_type == 'magnitude':    

            def mag_map(r):
                temp = r / max_dens

                return temp
            bmp = mag_vis(psii.T, mag_map=mag_map)

        if vis_type not in vis_types:
            raise ValueError

        # Compute estimate of radial probability density by 
        # numerical integration over thin annuli.
        n_r = round(xpix / 8)
        r_range = np.linspace(0, np.max(x_range), n_r + 1)[1:]
        dr = r_range[1] - r_range[0]
        prob = np.zeros(r_range.shape)
        dx = x_range[1]-x_range[0]
        dy = y_range[1]-y_range[0]

        for k in range(n_r):
            if k > 0:
                mask = (r <= r_range[k]) * (r > r_range[k-1])
            else:
                mask = (r <= r_range[k]) 

            prob[k] = np.sum(np.abs(psi)**2 * mask) * dx * dy / dr


        #print(prob)

        #print(np.max(bmp), np.min(bmp))

        # clip density at too large values
        #bmp = max_dens * (bmp >= max_dens) + bmp * (bmp < max_dens)
        
        ax.clear()

        #ax.imshow(bmp, vmin = 0.0, vmax = max_dens, cmap = cmap, origin='lower')
        ax.imshow(bmp,origin='lower', extent = [np.min(x_rangei), np.max(x_rangei), np.min(y_rangei), np.max(x_rangei)])

        
        caption = f't = {t:.1f}, E_field = {Efield:.6f}, energy = {energy:.6f}'
        ax.text(.0125, .0125, caption, color = 'white', transform=ax.transAxes, fontdict = font)

        ax2.clear()
        ax2.bar(r_range, height = prob, width = dr, color = 'white')
        ax2.set_xlabel('r', fontdict = font_inset, color = 'white')
        ax2.set_ylabel('P(r)', fontdict = font_inset, color = 'white')
 

        ax2.spines['bottom'].set_color('white')
        ax2.spines['top'].set_color('white')
        ax2.spines['left'].set_color('white')
        ax2.spines['right'].set_color('white')
        ax2.xaxis.label.set_color('white')
        ax2.yaxis.label.set_color('white')
        ax2.tick_params(axis='both', colors='white')
        ax2.patch.set_facecolor('black')

        framename = f'frame{i:05d}.png'
        plt.savefig(folder + '/' + framename, dpi = imgsize/figsize)
        frames.append(f"file '{framename}' \n")

with open(f'{folder}/framelist.txt', 'w') as f:
    f.writelines(frames)

cmd = f'ffmpeg -y -r 30 -f concat -i {folder}/framelist.txt -vcodec libx264 -crf 25  -pix_fmt yuv420p {folder}/{moviename}.mp4'


subprocess.run(cmd, shell=True)

