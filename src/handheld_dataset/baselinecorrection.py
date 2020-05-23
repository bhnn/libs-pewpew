import numpy as np
import glob
import os
from pathlib import Path
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from tqdm import tqdm

def baseline_als_optimized(y, lam=102, p=0.1, niter=10):
    """
    Calculates the baseline correction of LIBS spectra and returns corrected
    spectra.
    :param y:       sample to process
    :param lam:     smoothness
    :param p:       asymmetry
    :param niter:   number of iterations
    """
    if np.max(y) <0:
        warnings.warn('LIBS shot is empty, no positive values')
    y = y[:,1].clip(min=0) #remove values < 0 and discard wavelengths
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = lam * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    new = y - z
    return new.clip(min=0)


path = r'/Users/jh/github'

def correct_baseline(path, datasetname):
    # create output directory
    if not os.path.exists(os.path.join(path, datasetname +'_corrected')):
        os.makedirs(os.path.join(path, datasetname +'_corrected'))

    files = sorted(glob.glob(os.path.join(path, datasetname, '*.npz')))

    for f in tqdm(files):
        with np.load(f) as npz_file:
            mutable_file = dict(npz_file)
            mutable_file['data'][:,1] = baseline_als_optimized(npz_file['data'])
            filename = f[-26:]
            output = os.path.join(path, datasetname +'_corrected', filename)
            np.savez_compressed(output, data=mutable_file['data'], labels=mutable_file['labels'])



correct_baseline(path, 'hh_12/test')
correct_baseline(path, 'hh_12/train')
correct_baseline(path, 'hh_all/test')
correct_baseline(path, 'hh_all/train')
