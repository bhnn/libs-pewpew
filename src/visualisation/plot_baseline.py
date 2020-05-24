import numpy as np
import glob
from os.path import join
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.sparse.linalg import spsolve

import pandas as pd
import os
from tqdm import tqdm


path = r'/Users/jh/github'


def baseline_als_optimized(y, lam=102, p=0.01, niter=10):
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
    new = y - z
    return new.clip(min=0)

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


def plot_spectra_with_baseline():
    """
    Plots one spectrum of chalcopyrite and the calculated baseline
    Saves plot in libs-pewpew/data/spectrum_baseline.png
    """

    files = sorted(glob.glob(os.path.join(path, 'hh_6/test', '0026_006_005_017_00057.npz')))

    with np.load(files[0]) as npz_file_x:
        x_values = npz_file_x['data'][:,0]
    with np.load(files[0]) as npz_file:
        samples = npz_file['data'][:,1] #data without wavelength

    plt.figure(figsize=[8.5,5])
    plt.plot(x_values, samples, label='Normal spectra', color='#0d627a')
    plt.plot(x_values, only_baseline(samples), label='Baseline', color='#96cf6d')
    plt.xlabel('Wavelength', fontsize=14)
    plt.ylabel('Intensity', fontsize=14)
    plt.title('Average LIBS spectra of one measurepoint - Mineral: Chalcopyrite', fontsize=16, pad=15)
    plt.legend(fontsize=13)
    plt.axhline(linewidth=0.5, color='black')

    plt.savefig(os.path.join(path,'libs-pewpew/data/spectrum_baseline.png'))
    plt.show()

plot_spectra_with_baseline()

def plot_spectra_baselinecorrection():
    """
    Plots one spectrum of chalcopyrite with baseline and the baseline corrected spectrum
    Saves figure in libs-pewpew/data/spectrum_corrected_baseline.png
    """

    files = sorted(glob.glob(os.path.join(path, 'hh_6/test', '0026_006_005_017_00057.npz')))

    with np.load(files[0]) as npz_file_x:
        x_values = npz_file_x['data'][:,0]

    with np.load(files[0]) as npz_file:
        samples = npz_file['data'][:,1]


    baselinecorrection = baseline_als_optimized(samples)
    baseline = only_baseline(samples)

    plt.figure(figsize=[8.5,5])
    plt.plot(x_values, samples, label='Normal spectra', color= '#0d627a')
    plt.plot(x_values, baselinecorrection, label='Spectra with corrected baseline', color='#f2dc5c')
    plt.plot(x_values, baseline, label='Baseline', color='#96cf6d')
    plt.xlabel('Wavelength', fontsize=14)
    plt.ylabel('Intensity', fontsize=14)
    plt.title('Average LIBS spectra of one measurepoint - Mineral: Chalcopyrite', fontsize= 16, pad=15)
    plt.legend(fontsize=13)
    plt.axhline(linewidth=0.5, color='black')
    plt.savefig(os.path.join(path,'libs-pewpew/data/spectrum_corrected_baseline.png'))
    plt.show()


plot_spectra_baselinecorrection()
