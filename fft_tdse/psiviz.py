import numpy as np
from matplotlib.cm import get_cmap
import hsluv
import colorcet

phase_cmap = colorcet.cm['cyclic_mygbm_30_95_c78']
#phase_cmap = colorcet.cm['cyclic_ymcgy_60_90_c67']
#phase_cmap = colorcet.cm['cyclic_wrwbw_40_90_c42']
dens_cmap = colorcet.cm['bgy']

def hsluv_to_rgb(h, s, l):
    fun = lambda h, s, l : hsluv.hsluv_to_rgb((h,s,l))
    return np.vectorize(fun)(h, s, l)

def rgb_to_hsluv(r, g, b):
    fun = lambda r, g, b : hsluv.rgb_to_hsluv((r,g,b))
    return np.vectorize(fun)(r, g, b)

def dens_vis(z, mag_map = lambda r: r, cmap=dens_cmap):

    dens = mag_map(np.abs(z)**2).clip(0,1)
    bmp = cmap(dens)

    return bmp[:,:,:3]


def mag_vis(z, mag_map = lambda r: r, cmap=dens_cmap):

    dens = mag_map(np.abs(z)).clip(0,1)
    bmp = cmap(dens)

    return bmp[:,:,:3]


def phase_mag_vis(z, mag_map = lambda r: r, cmap = phase_cmap):
    """ Render a phase-magnitude visualization of a complex 2d field using perceptually uniform colors from the `colorcet` module. 

    Input:
    - `z`: 2d complex `np.ndarray`
    - `mag_map`: scalar function to remap desired range of magnitudes. Default map is the identity map.

    Output:
    - `rgb`: 3d real `np.ndarray`, the last dimension is an RGB value.

    TODO: Generalize to any number of dimensions.
    
    """

    nx, ny = z.shape
    bmp = np.zeros((nx, ny, 3))

    hue = np.angle(z, deg=True)/360 + .5
    color = cmap(hue)

    for i in range(3):
        bmp[:,:,i] = color[:,:,i] * mag_map(np.abs(z)).clip(0, 1) 

    return bmp

def phase_mag_vis2(z, rho1 = 1.0, rho2 = 2.0, cmap = phase_cmap, mag_map = lambda r: r):
    """ Render a phase-magnitude visualization of a complex 2d field using perceptually uniform colors from the `colorcet` module. 
    
    This new version uses logic from the HSL color scheme: when the magnitude is between 0 and rho1, then
    colors are chosen from black to pure color. From rho1 to rho2, colors are chosen from pure color to white.
    Above rho2, colors are white.

    Args:
    - `z`: 2d complex `np.ndarray`
    - `rho1`: float, the magnitude at which the color starts to change from black to pure color.
    - `rho2`: float, the magnitude at which the color starts to change from pure color to white.
    - `cmap`: colormap

    Output:
    - `rgb`: 3d real `np.ndarray`, the last dimension is an RGB value.

    
    """

    nx, ny = z.shape
    bmp = np.zeros((nx, ny, 3))

    hue = np.angle(z, deg=True)/360 + .5
    color = cmap(hue)
    mag = mag_map(np.abs(z))

    satcolor = np.ones(3)
    
    range1 = mag.clip(0, rho1) / rho1 * (mag <= rho1)
    range2 = ((mag.clip(rho1, rho2) - rho1) / (rho2 - rho1)) * (mag > rho1)
    
    for i in range(3):
        bmp[:,:,i] = color[:,:,i] * range1
        bmp[:,:,i] += satcolor[i]*range2 + color[:,:,i]*(1-range2)*(mag>rho1)

    return bmp


def psiviz(I,normalize_columns=False,ampmap = lambda u: u):

    """ Compute an RGB bitmap visualization of a complex array.
    For 1D "movies", it is useful to normalize each column, via the optional
    parameter normalize_coluns = True.

    ampmap allows you to remap the amplitude at each point to enhance, say, low amplitudes. """

    shape = I.shape
    nn = np.prod(shape)



    image2 = ampmap(np.abs(I))

    #print('Rendering image of size ', shape)
    if normalize_columns:
    #normalize each column (time slice) to brighten
    #diffuse wavefunctions
        for c in range(I.shape[1]):
            m = np.max(image2[:,c])
            image2[:,c] /= m

    # compute Arg psi normalized to [0,1]
    hue = (np.arctan2(I.imag,I.real) + np.pi)/(2*np.pi)
    cmap = get_cmap('hsv')
    # image1 = each pixel colored according to phase angle
    image1 = cmap(hue)[:,:,:3]
    image2 = image2.reshape((nn,))
    image2 /= np.max(image2)
    # rescale image1 according to abs(psi)
    image1 = image1.reshape((nn,3))
    for k in range(3):
        image1[:,k] *= image2

    image1 = image1.reshape((*shape,3))


    return(image1)

def render2d(wf, cmap_name = 'hsv'):
    """ Render the wavefunction using a simple diffuse lightning model."""

    alpha = 200
    H = lambda x: 0.5 + 0.5 * np.tanh(alpha*x)

    fft2 = np.fft.fft2
    ifft2 = np.fft.ifft2
    light = np.array([1, .25, 1.5])
    light = light / np.linalg.norm(light)
    height_scale = 2

    psi = wf.psi
    (nx,ny) = psi.shape

    phi = (np.arctan2(psi.imag,psi.real) + np.pi)/(2*np.pi)
    cmap = get_cmap(cmap_name)
    # image1 = each pixel colored according to phase angle
    colors = cmap(phi)[:,:,:3]

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

    sat = H(rho/np.max(rho)-0.01)

    image = colors.copy()
    white = np.ones((nx,ny,3))

    for k in range(3):
        image[:,:,k] = (sat * colors[:,:,k] + (1-sat) * white[:,:,k]) * i
        #image[:,:,k] = sat * white[:,:,k]

    return(image)

def render2d_other(wf, cmap_name = 'hsv'):
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
