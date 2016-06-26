""" nn_template.py
    Template-matching algorithm to classify a given V1 response as coming from
    one of the stimulus types seen in the training set responses.
    
    Old method:
    A k-means computation finds K centroids for the responses to each stimulus
    kind. It also stores the most frequent cluster index for each neuron for the
    training data, and when comparing the test point to the templates for each
    stimulus kind, only computes the distance to this most common cluster index
    for that neuron.
    NOTE : This may not be necessary, or good.

    New method:
    Find distance to all centroids for each stimulus kind. Use the argmin among
    mins as the nearest stimulus kind for the response.

    TODO : Hasn't been tested in a while : needs a look.
"""

# Basics
import numpy as np

# Clustering
import scipy.cluster.vq as scvq

class ClusterTemplateNN:
    def __init__(self, K=3):
        self.K = K
        pass

    def fit(self, tr):
        S, N, T, N_tr = tr.shape
        # rsps becomes (S,L,N*R)
        rsps = tr.swapaxes(1,2).reshape((S,T,N_tr*N))
        self.centroids = []
        for i in range(S):
            rsps_i = rsps[i,:,:].swapaxes(0,1) # rsps_i is (N*R,L)
            ce, _ = scvq.kmeans2(rsps_i, self.K, minit='points')
            self.centroids.append(ce)
        self.templates = np.array(self.centroids)

    def predict(self, te):
        S, N, T, R = te.shape
        
        labels = np.zeros((S, R, N)).astype(int)
        for i in range(N):
            for i_trial in range(R):
                for i_s in range(S):
                    r = te[i_s, i, :, i_trial]
                    dists = [min([np.linalg.norm(r - self.templates[j,k,:])
                                  for k in range(self.K)])
                             for j in range(S)]
                    labels[i_s, i_trial, i] = np.argmin(dists)
        return labels
