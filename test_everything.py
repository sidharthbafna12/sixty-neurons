#!/usr/bin/env python

import os, sys

files = {'c' : './clustering.py', 't' : './tuning.py',
         'f' : './fourier.py', 'n' : './natural_movie_construction.py',
         'g' : './grating_construction.py'}

# Run as './test_everything.py [-cftng]'
if __name__ == "__main__":
    if len(sys.argv) == 1:
        args = 'cftng'
    else:
        args = sys.argv[1][1:]
    
    for f in files:
        if f in args:
            print 'Testing %s...' % files[f]
            os.system(files[f])
