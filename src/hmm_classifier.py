""" hmm_classifier.py
    Uses HMMs to classify neuronal responses as coming from one of many video
    classes.

    Superseded by the HTK classifier in the project_root/hmm directory.
"""

import numpy as np
from hmmlearn.hmm import GaussianHMM

import warnings

class HMMClassifier:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, tr_seqs):
        self.tr_seqs = tr_seqs
        self.n_classes = len(tr_seqs)
        
        self.models = []
        for class_seqs in tr_seqs:
            lengths = [seq.shape[0] for seq in class_seqs]
            X = np.vstack(class_seqs)
            
            print X.shape, len(lengths)

            start_prob = np.ones(self.n_components)
            start_prob /= np.sum(start_prob)
            transmat = np.ones((self.n_components, self.n_components))
            for i in range(self.n_components):
                transmat[i,:] /= transmat[i,:].sum()

            trained = False
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                while not trained:
                    try:
                        model = GaussianHMM(n_components=self.n_components,
                                            covariance_type='diag', n_iter=50,
                                            startprob_prior=start_prob,
                                            transmat_prior=transmat)\
                                .fit(X, lengths)
                        trained = True
                    except RuntimeWarning as w:
                        print w
                        print start_prob
                        print transmat
                        print lengths
                        print X
                        start_prob = np.random.random(self.n_components)
                        transmat = np.random.random((self.n_components,
                                                     self.n_components))

            self.models.append(model)

    def predict(self, te_seqs):
        all_predictions = []

        for rsp_set in te_seqs:
            predictions = []
            for seq in rsp_set:
                scores = [model.score(seq) for model in self.models]
                predictions.append(np.argmax(scores))
            all_predictions.append(predictions)
        return all_predictions
