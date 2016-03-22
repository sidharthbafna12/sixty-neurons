# Filters corresponding to the thalamus.
# Should mimic attentional/saliency kind of stuff.
# Just a gaussian mask for now.

import numpy as np
import scipy.ndimage as ndi

# Stolen from http://stackoverflow.com/a/17201686
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def thalamus_filter(rsps):
    L_Y, L_X, T = rsps.shape
    mask = matlab_style_gauss2D(shape=(L_Y,L_X), sigma=(L_X+L_Y)/6.)
    return rsps * np.dstack([mask] * T)
