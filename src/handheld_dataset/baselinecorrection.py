import numpy as np
import glob
import os
from pathlib import Path
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import sys
import warnings

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



def correct_baseline(datasetpath):
    """
    Saves each dataset (handheld with 12 and 100 minerals) in a new folder with
    baseline corrected spectra to minimize the runtime of training the models
    :param datasetpath:         path to folder containing the dataset with train/test split
    :param datasetname:         name of the dataset
    :returns:                   Saves two new folders with baseline corrected spectra as train and test
    """

    # create output directory
    for folder in ['train', 'test']:
        if not os.path.exists(os.path.join(datasetpath, folder)):
            os.makedirs(os.path.join(datasetpath, folder))

        files = sorted(glob.glob(os.path.join(datasetpath, folder + '_uncorrected', '*.npz')))

        for f in tqdm(files):
            with np.load(f) as npz_file:
                mutable_file = dict(npz_file)
                mutable_file['data'][:,1] = baseline_als_optimized(npz_file['data'])
                filename = f[-26:]
                output = os.path.join(datasetpath, folder, filename)
                np.savez_compressed(output, data=mutable_file['data'], labels=mutable_file['labels'])


if __name__ == '__main__':

    with open('config/datasets.yaml') as cnf:
        dataset_configs = yaml.safe_load(cnf)
    try:
        hh_12_path = dataset_configs['hh_12_path']
        hh_all_path = dataset_configs['hh_all_path']
    except KeyError as e:
        print(f'Missing dataset config key: {e}')
        sys.exit(1)

    correct_baseline(hh_12_path)
    correct_baseline(hh_all_path)
