import numpy as np
from matplotlib.cm import get_cmap


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
