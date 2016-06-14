""" response.py
    General container for all recorded responses.
"""

import numpy as np

class Response:
    """ Data members:
        name : (string) something to call this response
        data : (4D ndarray) response data
               (S, N, L, R) : (stimulus index, neuron index,
                               sample index, trial index)
        attrs : (dict) other attributes that may be associated with this
    """
    def __init__(self, name, data_filename, attrs={}):
        """ name : self.name
            data_filename : the .npy file that holds the data
            attrs : additional attributes
        """
        self.name = str(name)
        self.data = np.load(data_filename)
        self.attrs = attrs
    
        assert len(self.data.shape) == 4
