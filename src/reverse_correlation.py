""" reverse_correlation.py
"""

import numpy as np

class OptimalPriorReverseCorrelation:
    def __init__(self, n_lag=None):
        self.n_lag = n_lag

    def fit(self, rsp, stim):
        """
            rsp : Response instance containing data (S x N x L x R)
            stim : list of movies (len(stim) = S)
        """
        assert rsp.data.shape[0] == len(stim)
        assert rsp.data.shape[2] == stim[0].shape[2]

        n_stim, n_cells, n_samples, n_trials  = rsp.data.shape
        self.LY, self.LX, n_samples = stim[0].shape
        n_px = self.LY * self.LX

        # Average the response across trials.
        avg_rsp = np.mean(rsp.data, axis=3)

        # Remove the baseline response for all cells.
        self.baseline_rsps = np.zeros(n_cells)
        for i_n in range(n_cells):
            self.baseline_rsps[i_n] = avg_rsp[:,i_n,:].mean()
            avg_rsp[:,i_n,:] -= self.baseline_rsps[i_n]

        # Create the R and S matrices for all stimulus movies.
        R = [np.zeros((n_cells * self.n_lag, n_samples))
             for i_s in range(n_stim)]
        S = [np.zeros((n_px, n_samples)) for i_s in range(n_stim)]
        for i_s in range(n_stim):
            for t in range(n_samples):
                S[i_s][:,t] = stim[i_s][:,:,t].flatten()

            for i_n in range(n_cells):
                for i_lag in range(self.n_lag):
                    row = i_n * self.n_lag + i_lag
                    R[i_s][row,i_lag:] = avg_rsp[i_s,i_n,:n_samples-i_lag]

        # Estimate reconstruction filters from all movies.
        C_RR = [np.dot(r, r.T) for r in R]
        C_RS = [np.dot(r, s.T) for r, s in zip(R, S)]
        G = [np.dot(np.linalg.inv(c_rr), c_rs) for c_rr, c_rs in zip(C_RR,C_RS)]
        self.G = np.mean(G, axis=0)

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
        for i_s in range(n_stim):
            reconst = []
            for i_tr in range(n_trials):
                movie = np.zeros((self.LY, self.LX, n_samples))
                R = np.zeros((n_cells, n_samples))
                for i_n in range(n_cells):
                    R[i_n,:] = rsp.data[i_s,i_n,:,i_tr]-self.baseline_rsps[i_n]

                for i_px in range(n_px):
                    row = i_px / self.LX
                    col = i_px % self.LY
                    g = self.G[:,i_px].reshape((n_cells, self.n_lag))
                    s = np.array([np.convolve(g[j_n,:], R[j_n,:], mode='same')
                                  for j_n in range(n_cells)])
                    movie[row,col,:] = np.sum(s, axis=0)

                reconst.append(movie)
            reconstructed.append(reconst)
        return reconstructed

class FlatPriorReverseCorrelation:
    def __init__(self, n_lag=None):
        self.n_lag = n_lag

    def fit(self, rsp, stim):
        """
            rsp : Response instance containing data (S x N x L x R)
            stim : list of movies (len(stim) = S)
        """
        assert rsp.data.shape[0] == len(stim)
        assert rsp.data.shape[2] == stim[0].shape[2]

        n_stim, n_cells, n_samples, n_trials  = rsp.data.shape
        self.LY, self.LX, n_samples = stim[0].shape
        n_px = self.LY * self.LX

        # Average the response across trials.
        avg_rsp = np.mean(rsp.data, axis=3)

        # Remove the baseline response for all cells.
        self.baseline_rsps = np.zeros(n_cells)
        for i_n in range(n_cells):
            self.baseline_rsps[i_n] = avg_rsp[:,i_n,:].mean()
            avg_rsp[:,i_n,:] -= self.baseline_rsps[i_n]

        # Create the R and S matrices for all stimulus movies.
        R = [np.zeros((n_cells, n_samples)) for i_s in range(n_stim)]
        S = [np.zeros((self.n_lag * n_px, n_samples)) for i_s in range(n_stim)]
        for i_s in range(n_stim):
            for i_n in range(n_cells):
                R[i_s][i_n,:] = avg_rsp[i_s,i_n,:]

            for t_lag in range(self.n_lag):
                for t in range(t_lag, n_samples):
                    t_eff = t - t_lag
                    S[i_s][t_lag*n_px:(t_lag+1)*n_px,t_eff] = \
                            stim[i_s][:,:,t_eff].flatten()

        # Estimate STRFs from all movies.
        C_SS = [np.dot(s, s.T) for s in S]
        C_SR = [np.dot(s, r.T) for r, s in zip(R, S)]
        H = [np.dot(np.linalg.inv(c_ss), c_sr) for c_ss, c_sr in zip(C_SS,C_SR)]
        self.H = np.mean(H, axis=0)
        self.F = np.dot(np.linalg.inv(np.dot(self.H, self.H.T)), self.H)

    def _S_to_movie(self, S):
        t_max = S.shape[1]
        n_px = self.LY * self.LX
        movie = np.zeros((self.LY, self.LX, t_max))

        for t in range(t_max):
            m = np.zeros((self.LY, self.LX))
            if t < self.n_lag:
                for i_t in range(t):
                    m += S[i_t*n_px:(i_t+1)*n_px,t]\
                            .reshape((self.LY, self.LX))
                m /= (t+1)
            else:
                for i_t in range(self.n_lag):
                    m += S[i_t*n_px:(i_t+1)*n_px,t]\
                            .reshape((self.LY, self.LX))
                m /= self.n_lag

            movie[:,:,t] = m
        return movie
        
    def predict(self, rsp):
        """
            rsp : Response instance containing data (S x N x L x R)
            Return list of list of movies with len(list) = S
                    and
                   len(list[i]) = rsp.data.shape[3]
        """
        n_stim, n_cells, n_samples, n_trials = rsp.data.shape
        reconstructed = []
        for i_s in range(n_stim):
            reconst = []
            for i_tr in range(n_trials):
                R = np.zeros((n_cells, n_samples))
                for i_n in range(n_cells):
                    R[i_n,:] = rsp.data[i_s,i_n,:,i_tr]-self.baseline_rsps[i_n]
                S_hat = np.dot(self.F, R)
                reconst.append(self._S_to_movie(S_hat))
            reconstructed.append(reconst)
        return reconstructed
