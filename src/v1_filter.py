# Filters corresponding to V1.

import numpy as np

def v1_best_fit_filter(thal_rsp, v1_neuron_rsp, stimulus_type):
    if stimulus_type == 'natural':
        from params.naturalmovies.pathway_params import FOV
        from params.naturalmovies.pathway_params import V1_RF_WIDTH
        from params.naturalmovies.pathway_params import V1_GRID_SPACING
        from params.naturalmovies.stimulus_params import MAX_ACORR_LAG
        from params.naturalmovies.stimulus_params import CA_SAMPLING_RATE
    elif stimulus_type == 'grating':
        from params.grating.pathway_params import FOV
        from params.grating.pathway_params import V1_RF_WIDTH
        from params.grating.pathway_params import V1_GRID_SPACING

    rgc_n_cells = np.array([thal_rsp.shape[1], thal_rsp.shape[0]])
    eff_ppd = rgc_n_cells / FOV
    assert eff_ppd[0] == eff_ppd[1]
    eff_ppd = eff_ppd[0]

    R = eff_ppd * V1_GRID_SPACING
    r1 = int(np.ceil(eff_ppd * V1_RF_WIDTH))
    padded_thal_rsp = np.pad(thal_rsp, ((r1, r1), (r1, r1), (0, 0)),
                             mode='constant')

    max_lag_samples = int(MAX_ACORR_LAG * CA_SAMPLING_RATE)

    centres = [(x, y) for x in np.arange(r1, r1 + rgc_n_cells[0], R)
                      for y in np.arange(r1, r1 + rgc_n_cells[1], R)]
    coeffs = []
    sses = []

    for x,y in centres:
        patch = padded_thal_rsp[y-r1:y+r1+1,x-r1:x+r1+1,:]
        l_y, l_x, T = patch.shape
        patch = patch.reshape((l_y * l_x, T))
        S = np.zeros(((max_lag_samples + 1) * l_y * l_x, T))
        for j in range(max_lag_samples + 1):
            S[j*l_y*l_x:(j+1)*l_y*l_x,j:] = patch[:,:T-j]
        R = v1_neuron_rsp

        C_SS = np.dot(S, S.T)
        C_SR = np.dot(S, R.T)
        # import pdb; pdb.set_trace()
        H = np.dot(np.linalg.pinv(C_SS), C_SR)
        R_hat = np.dot(H.T, S)
        sse = np.linalg.norm(R - R_hat)

        coeffs.append(H.reshape((l_y, l_x, max_lag_samples + 1)))
        sses.append(sse)

    i_best = np.argmax(sses)
    return np.array(centres[i_best]) - r1, coeffs[i_best], sses[i_best]
