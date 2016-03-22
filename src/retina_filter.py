# Filters corresponding to the retinal ganglion cell (?)

import numpy as np
import scipy.ndimage

def rgc_filter(rf_types, stimulus_type):
    if stimulus_type == 'natural':
        from params.naturalmovies.pathway_params import RGC_CENTRE_WIDTH
        from params.naturalmovies.pathway_params import RGC_SURR_WIDTH
        from params.naturalmovies.pathway_params import RGC_CELL_SPACING
        from params.naturalmovies.pathway_params import RGC_N_CELLS
        from params.naturalmovies.stimulus_params import PIXELS_PER_DEGREE
        from params.naturalmovies.stimulus_params import N_PX
    elif stimulus_type == 'grating':
        from params.grating.pathway_params import RGC_CENTRE_WIDTH
        from params.grating.pathway_params import RGC_SURR_WIDTH
        from params.grating.pathway_params import RGC_CELL_SPACING
        from params.grating.pathway_params import RGC_N_CELLS
        from params.grating.stimulus_params import PIXELS_PER_DEGREE
        from params.grating.stimulus_params import N_PX

    r1 = int(np.ceil((RGC_CENTRE_WIDTH / 2) * PIXELS_PER_DEGREE))
    r2 = int(np.ceil((RGC_SURR_WIDTH / 2) * PIXELS_PER_DEGREE))
    centre_val = 1.0 / (np.pi * (r1 ** 2))
    surr_val = (-centre_val * (r1 ** 2)) / (((r2 + r1) ** 2) - (r1 ** 2))
    
    Y, X = np.ogrid[-(r1+r2):(r1+r2+1), -(r1+r2):(r1+r2+1)]
    centre_mask = X*X + Y*Y <= r1*r1
    surr_mask = np.logical_and((X*X + Y*Y <= (r1+r2)**2), (X*X + Y*Y > r1*r1))
    on_filt = np.ones(centre_mask.shape) * (centre_mask * centre_val
                                            + surr_mask * surr_val)
    off_filt = -on_filt

    def f(imgs):
        def g(signal, filt):
            D = RGC_CELL_SPACING * PIXELS_PER_DEGREE
            out = np.empty((RGC_N_CELLS[1], RGC_N_CELLS[0]))
            padded_signal = np.pad(signal, r1 + r2, mode='constant')
            for ix, x in enumerate(np.arange(r1+r2, N_PX[0] + r1+r2, D)):
                for iy, y in enumerate(np.arange(r1+r2, N_PX[1] + r1+r2, D)):
                    out[iy,ix] = np.sum(padded_signal[y-(r1+r2):y+(r1+r2+1),
                                                      x-(r1+r2):x+(r1+r2+1)]
                                    * filt)
            # return scipy.ndimage.correlate(signal, filt)[::D,::D]
            return out

        input_deltas = np.dstack((imgs[:,:,0:1], np.diff(imgs, axis=2)))
        
        print 'Calculating effective input signal...'
        # on_input_signals = [np.maximum(img, delta)
        #                     for img, delta in zip(imgs, input_deltas)]
        # off_input_signals= [np.minimum(img, delta)
        #                     for img, delta in zip(imgs, input_deltas)]
        on_input_signals = [imgs[:,:,i] + input_deltas[:,:,i]
                            for i in range(imgs.shape[2])]
        off_input_signals= [imgs[:,:,i] + input_deltas[:,:,i]
                            for i in range(imgs.shape[2])]

        print 'Computing filter output...'
        on_rgc_outputs = map(lambda s : g(s, on_filt), on_input_signals)
        off_rgc_outputs= map(lambda s : g(s, off_filt), off_input_signals)

        on_centre_mask = (rf_types == 'on')
        off_centre_mask= (rf_types == 'off')

        output = [(on_centre_mask * on_out + off_centre_mask * off_out)
                  for on_out,off_out in zip(on_rgc_outputs,off_rgc_outputs)]

        print 'Accumulating output deltas...'
        # return np.cumsum(np.array(output_deltas), axis=0)
        # return np.array(output_deltas)
        return np.dstack(output)
    return f
