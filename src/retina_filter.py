# Filters corresponding to the retinal ganglion cell (?)

import numpy as np
import scipy.ndimage

from params.stimulus_params import *
from params.pathway_params import *

def rgc_filter(rf_types):
    r1 = RGC_CENTRE_WIDTH
    r2 = RGC_SURR_WIDTH
    centre_val = 1.0
    surr_val = (-centre_val * (r1 ** 2)) / (((r2 + r1) ** 2) - (r1 ** 2))
    
    Y, X = np.ogrid[-(r1+r2):(r1+r2+1), -(r1+r2):(r1+r2+1)]
    centre_mask = X*X + Y*Y <= r1*r1
    surr_mask = np.logical_and((X*X + Y*Y <= (r1+r2)**2), (X*X + Y*Y > r1*r1))
    on_filt = np.ones(centre_mask.shape) * (centre_mask * centre_val
                                            + surr_mask * surr_val)
    off_filt = -on_filt

    def f(imgs):
        def g(signal, filt):
            D = RGC_CELL_SPACING
            out = np.empty((RGC_N_CELLS[1], RGC_N_CELLS[0]))
            padded_signal = np.pad(signal, r1 + r2, mode='constant')
            for x in xrange(r1+r2, N_PX_NAT[0] + r1+r2, D):
                for y in xrange(r1+r2, N_PX_NAT[1] + r1+r2, D):
                    i = (y-(r1+r2)) / D
                    j = (x-(r1+r2)) / D
                    out[i,j] = np.sum(padded_signal[y-(r1+r2):y+(r1+r2+1),
                                                    x-(r1+r2):x+(r1+r2+1)]
                                    * filt)
            # return scipy.ndimage.correlate(signal, filt)[::D,::D]
            return out

        input_deltas = np.vstack((imgs[0:1,:,:], np.diff(imgs, axis=0)))
        
        print 'Calculating effective input signal...'
        # on_input_signals = [np.maximum(img, delta)
        #                     for img, delta in zip(imgs, input_deltas)]
        # off_input_signals= [np.minimum(img, delta)
        #                     for img, delta in zip(imgs, input_deltas)]
        on_input_signals = [img + delta
                            for img, delta in zip(imgs, input_deltas)]
        off_input_signals= [img + delta
                            for img, delta in zip(imgs, input_deltas)]

        print 'Computing filter output...'
        on_rgc_outputs = map(lambda s : g(s, on_filt), on_input_signals)
        off_rgc_outputs= map(lambda s : g(s, off_filt), off_input_signals)

        on_centre_mask = (rf_types == 'on')
        off_centre_mask= (rf_types == 'off')

        output_deltas=[(on_centre_mask * on_out + off_centre_mask * off_out)
                       for on_out,off_out in zip(on_rgc_outputs,off_rgc_outputs)]

        print 'Accumulating output deltas...'
        # return np.cumsum(np.array(output_deltas), axis=0)
        return np.array(output_deltas)
    return f
