""" linear_regression.py
    Linear Regression to reconstruct stimulus from responses.
"""

import numpy as np

from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from ssim import ssim
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as hac

from correlation import signal_correlation, noise_correlation
from clustering import NeuronClustering

class LinearReconstruction:
    def __init__(self, model_name, model_type, n_clusters=None,
                 n_components=None, n_lag=None, regularisation=None):
        self.n_lag = n_lag
        self.model_name = model_name
        self.model_type = model_type
        self.n_clusters = n_clusters
        self.clustering = NeuronClustering(self.n_clusters, signal_correlation)
        
        if model_name == 'cca':
            self.n_components = n_components
            self.model = CCA(n_components=self.n_components)
        elif model_name == 'linear-regression':
            if regularisation is None:
                self.model = LinearRegression()
            elif regularisation == 'l1':
                self.model = Lasso()
            elif regularisation == 'l2':
                self.model = Ridge()
            elif regularisation == 'l1l2':
                self.model = ElasticNet()
            else:
                raise NotImplementedError

    def fit(self, rsp, stim):
        """
            rsp : Response instance containing data (S x N x L x R)
            stim : list of movies (len(stim) = S)
        """

        n_stim, n_cells, n_samples, n_trials = rsp.data.shape
        self.LY, self.LX, n_samples = stim[0].shape
        n_px = self.LY * self.LX

        self.clustering.fit(rsp.data)
        dn_rsp_data = self.clustering.divnorm(rsp.data)

        # Average the response across trials.
        avg_rsp = np.mean(dn_rsp_data, axis=3)

        # Remove the baseline response for all cells.
        self.baseline_rsps = np.zeros(n_cells)
        for i_n in range(n_cells):
            self.baseline_rsps[i_n] = avg_rsp[:,i_n,:].mean()
            avg_rsp[:,i_n,:] -= self.baseline_rsps[i_n]
        
        if self.model_type == 'reverse':
            # Pad with zeros to fill out the responses at the end.
            avg_rsp = np.pad(avg_rsp, ((0,0), (0,0), (0,self.n_lag-1)),
                             mode='constant')

            # Create the X and Y matrices.
            X = np.zeros((n_stim * n_samples, n_cells * self.n_lag))
            Y = np.zeros((n_stim * n_samples, n_px))
            for i_s in range(n_stim):
                for i_t in range(n_samples):
                    row = i_s * n_samples + i_t
                    X[row,:] = avg_rsp[i_s,:,i_t:i_t+self.n_lag].flatten()
                    Y[row,:] = stim[i_s][:,:,i_t].flatten()
        elif self.model_type == 'forward':
            # Pad movie with zeros to fill it out.
            p_stim = [np.pad(m,((0,0),(0,0),(self.n_lag-1,0)),mode='constant')
                      for m in stim]
            
            # Creating the X and Y matrices.
            X = np.zeros((n_stim * n_samples, n_cells))
            Y = np.zeros((n_stim * n_samples, n_px * self.n_lag))
            for i_s in range(n_stim):
                for i_t in range(n_samples):
                    row = i_s * n_samples + i_t
                    X[row,:] = avg_rsp[i_s,:,i_t]
                    Y[row,:] = p_stim[i_s][:,:,i_t:i_t+self.n_lag].flatten()
        
        self.model.fit(X, Y)

    def predict(self, rsp):
        """
            rsp : Response instance containing data (S x N x L x R)
            Return list of list of movies with len(list) = S
                    and
                   len(list[i]) = R
        """
        n_stim, n_cells, n_samples, n_trials = rsp.data.shape
        n_px = self.LY * self.LX
        reconstructed = []
        
        r = rsp.data
        r = self.clustering.divnorm(r)
        r = np.pad(r, ((0,0), (0,0), (0,self.n_lag-1), (0,0)),
                   mode='constant')

        # Remove baseline response for each cell.
        for i_n in range(n_cells):
            r[:,i_n,:,:] -= self.baseline_rsps[i_n]

        for i_s in range(n_stim):
            reconst = []
            for i_tr in range(n_trials):
                movie = np.zeros((self.LY, self.LX, n_samples))

                if self.model_type == 'reverse':
                    X = np.zeros((n_samples, n_cells * self.n_lag))
                    for i_t in range(n_samples):
                        X[i_t,:] = r[i_s,:,i_t:i_t+self.n_lag,i_tr].flatten()
                elif self.model_type == 'forward':
                    X = np.zeros((n_samples, n_cells))
                    for i_t in range(n_samples):
                        X[i_t,:] = r[i_s,:,i_t,i_tr]

                Y_pred = self.model.predict(X)
                for i_t in range(n_samples):
                    if self.model_type == 'reverse':
                        movie[:,:,i_t]=Y_pred[i_t,:].reshape((self.LY,self.LX))
                    elif self.model_type == 'forward':
                        window = Y_pred[i_t,:]\
                                .reshape((self.LY, self.LX, self.n_lag))
                        movie[:,:,i_t] = window[:,:,-1]
                reconst.append(movie)
            reconstructed.append(reconst)
        return reconstructed

    def reconstruction_quality(self, reconstructed_movies, actual_movies):
        """
            reconstructed_movies : output from self.predict
            actual_movies : ground truth
        """
        ssims = []
        for i in range(len(actual_movies)):
            x0 = actual_movies[i]
            T = x0.shape[2]
            sims = []
            for i_tr in range(len(reconstructed_movies[i])):
                x = reconstructed_movies[i][i_tr]
                sims.append([ssim(x0[:,:,i_t], x[:,:,i_t])
                             for i_t in range(T)])
            ssims.append(sims)
        return ssims
