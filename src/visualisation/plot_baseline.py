import numpy as np
import glob
from os.path import join
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.sparse.linalg import spsolve

import pandas as pd
import os
from tqdm import tqdm
import yaml
import sys
import warnings


def only_baseline(y, lam=102, p=0.01, niter=10):
    """
    Calculates the baseline correction of LIBS spectra and returns baseline
    :param y:       sample to process
    :param lam:     smoothness
    :param p:       asymmetry
    :param niter:   number of iterations
    """
    if np.max(y) <0:
        warnings.warn('LIBS shot is empty, no positive values')
    y = y.clip(min=0) #remove values < 0 and discard wavelengths
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
    return z


def plot_spectra_with_baseline(repo_path):
    """
    Plots one spectrum of chalcopyrite and its calculated baseline
    :param repo_path:   path to repository
    :returns:           Saves plot in /data/visualisations/spectrum_baseline as png and pdf
    """

    with np.load(os.path.join(repo_path, 'data/chalcopyrit_uncorrected.npy')) as npy_file:
        x_values =  npy_file['data'][:,0]
        sample =    npy_file['data'][:,1]


    plt.figure(figsize=[8.5,5])
    plt.plot(x_values, sample, label='Normal spectra', color='#0d627a')
    plt.plot(x_values, only_baseline(sample), label='Baseline', color='#96cf6d')
    plt.xlabel('Wavelength', fontsize=14)
    plt.ylabel('Intensity', fontsize=14)
    plt.title('LIBS Spectrum - Mineral: Chalcopyrite', fontsize=16, pad=15)
    plt.legend(fontsize=13)
    plt.axhline(linewidth=0.5, color='black')

    plt.savefig(os.path.join(repo_path,'data/visualisations/spectrum_baseline.png'))
    plt.savefig(os.path.join(repo_path,'data/visualisations/spectrum_baseline.pdf'))



def plot_spectra_baselinecorrection(repo_path):
    """
    Plots one spectrum of chalcopyrite with baseline and the baseline corrected spectrum
    :param repo_path:   path to repository
    :returns:           Saves figure in libs-pewpew/data/visualisations/spectrum_corrected_baseline as png and pdf
    """

    with np.load(os.path.join(repo_path, 'data/chalcopyrit_uncorrected.npy')) as npy_file:
        x_values = npy_file['data'][:,0]
        sample = npy_file['data'][:,1]

    with np.load(os.path.join(repo_path, 'data/chalcopyrit_corrected.npy')) as npy_file:
        baselinecorrection = npy_file['data'][:,1]

    baseline = only_baseline(sample)

    plt.figure(figsize=[8.5,5])
    plt.plot(x_values, sample, label='Normal spectra', color= '#0d627a')
    plt.plot(x_values, baselinecorrection, label='Spectra with corrected baseline', color='#f2dc5c')
    plt.plot(x_values, baseline, label='Baseline', color='#96cf6d')
    plt.xlabel('Wavelength', fontsize=14)
    plt.ylabel('Intensity', fontsize=14)
    plt.title('LIBS Spectrum - Mineral: Chalcopyrite', fontsize= 16, pad=15)
    plt.legend(fontsize=13)
    plt.axhline(linewidth=0.5, color='black')
    plt.savefig(os.path.join(repo_path,'data/visualisations/spectrum_corrected_baseline.png'))
    plt.savefig(os.path.join(repo_path,'data/visualisations/spectrum_corrected_baseline.pdf'))



if __name__ == '__main__':

    with open('config/datasets.yaml') as cnf:
        dataset_configs = yaml.safe_load(cnf)
        try:
            repo_path = dataset_configs['repo_path']
        except KeyError as e:
            print(f'Missing dataset config key: {e}')
            sys.exit(1)

    plot_spectra_with_baseline(repo_path)
    plot_spectra_baselinecorrection(repo_path)
