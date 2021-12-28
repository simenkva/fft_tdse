import numpy as np
from matplotlib.cm import get_cmap


def psiviz(I):

    """ Compute an RGB bitmap visualization of a complex array."""

    shape = I.shape
    nn = np.prod(shape)

    #print('Rendering image of size ', shape)

    # compute Arg psi normalized to [0,1]
    hue = (np.arctan2(I.imag,I.real) + np.pi)/(2*np.pi)
    cmap = get_cmap('hsv')
    # image1 = each pixel colored according to phase angle
    image1 = cmap(hue)[:,:,:3]
    image2 = np.abs(I.reshape((nn,)))
    image2 /= np.max(image2)
    # rescale image1 according to abs(psi)
    image1 = image1.reshape((nn,3))
    for k in range(3):
        image1[:,k] *= image2

    image1 = image1.reshape((*shape,3))

    # normalize each column (time slice) to brighten
    # diffuse wavefunctions
    # for c in range(I.shape[1]):
    #     m = np.max(np.abs(I[:,c]))
    #     for k in range(3):
    #         image1[:,c,k] /= m

    return(image1)
