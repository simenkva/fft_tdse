import numpy as np
from matplotlib.cm import get_cmap


def psiviz(I,normalize_columns=False):

    """ Compute an RGB bitmap visualization of a complex array.
    For 1D "movies", it is useful to normalize each column, via the optional
    parameter normalize_coluns = True. """

    shape = I.shape
    nn = np.prod(shape)


    image2 = np.abs(I)

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
