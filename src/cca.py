""" cca.py
"""

import numpy as np
from sklearn.cross_decomposition import CCA

class CCAReconstruction:
    def __init__(self, n_lag=None, n_components=None):
        self.n_lag = n_lag
        self.n_components = n_components

    def fit(self, rsp, stim):
        """
            rsp : Response instance containing data (S x N x L x R)
            stim : list of movies (len(stim) = S)
        """
        n_stim, n_cells, n_samples, n_trials = rsp.data.shape
        self.LY, self.LX, n_samples = stim[0].shape
        n_px = self.LY * self.LX

        # Average the response across trials.
        avg_rsp = np.mean(rsp.data, axis=3)

        # Remove the baseline response for all cells.
        self.baseline_rsps = np.zeros(n_cells)
        for i_n in range(n_cells):
            self.baseline_rsps[i_n] = avg_rsp[:,i_n,:].mean()
            avg_rsp[:,i_n,:] -= self.baseline_rsps[i_n]
        
        # Pad with zeros to fill out the responses at the end.
        avg_rsp = np.pad(avg_rsp, ((0,0), (0,0), (0,self.n_lag-1)),
                         mode='constant')

        # Create the X and Y matrices for CCA.
        X = np.zeros((n_stim * n_samples, n_cells * self.n_lag))
        Y = np.zeros((n_stim * n_samples, n_px))
        for i_s in range(n_stim):
            for i_t in range(n_samples):
                row = i_s * n_samples + i_t
                X[row,:] = avg_rsp[i_s,:,i_t:i_t+self.n_lag].flatten()
                Y[row,:] = stim[i_s][:,:,i_t].flatten()
        
        # Find CCA weights etc.
        self.cca = CCA(n_components = self.n_components)
        self.cca.fit(X, Y)

    def predict(self, rsp):
        """
            rsp : Response instance containing data (S x N x L x R)
            Return list of list of movies with len(list) = S
                    and
                   len(list[i]) = rsp.data.shape[3]
        """
        n_stim, n_cells, n_samples, n_trials = rsp.data.shape
        n_px = self.LY * self.LX
        reconstructed = []

        r = np.pad(rsp.data, ((0,0), (0,0), (0,self.n_lag-1), (0,0)),
                   mode='constant')
        
        # Remove baseline response for each cell.
        for i_n in range(n_cells):
            r[:,i_n,:,:] -= self.baseline_rsps[i_n]

        for i_s in range(n_stim):
            reconst = []
            for i_tr in range(n_trials):
                movie = np.zeros((self.LY, self.LX, n_samples))

                X = np.zeros((n_samples, n_cells * self.n_lag))
                for i_t in range(n_samples):
                    X[i_t,:] = r[i_s,:,i_t:i_t+self.n_lag,i_tr].flatten()

                Y_pred = self.cca.predict(X)
                for i_t in range(n_samples):
                    movie[:,:,i_t] = Y_pred[i_t,:].reshape((self.LY, self.LX))
                reconst.append(movie)
            reconstructed.append(reconst)
        return reconstructed
