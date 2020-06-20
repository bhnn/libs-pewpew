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

with open('config/datasets.yaml') as cnf:
    dataset_configs = yaml.safe_load(cnf)

path_hh12 = dataset_configs['hh_12_path']
path_syn = dataset_configs['synth_path']
path_repo = dataset_configs['repo_path']

minerals_12 = [ (26, 'Chalcopyrite'),
                (98, 'Tetrahedrite'),
                (19, 'Bornite'),
                (41, 'Cuprite'),
                (28, 'Chalcotrichite'),
                (97, 'Tenorite'),
                (88, 'Rosasite'),
                (11, 'Azurit'),
                (73, 'Malachite'),
                (86, 'Pseudomalachite'),
                (80, 'Olivenite'),
                (35, 'Cornetite')]

minerals_12_id = [a_tuple[0] for a_tuple in minerals_12]
mineral_names = [a_tuple[1] for a_tuple in minerals_12]


def save_average_minerals():
    """
    Saves a numpy file containing the average spectra of all 12 minerals in the hh_12 test dataset
    for the visualisation of the different spectra
    :returns: saves numpy file in libs-pewpew/data/average_spectra_handheld.npy
    """
    average_samples=[]
    for m_id in minerals_12_id:
        print(m_id)
        m_id = '{0:04d}'.format(m_id)
        files = sorted(glob.glob(os.path.join(path_hh12, 'test', m_id+'*.npz')))
        files.extend(sorted(glob.glob(os.path.join(path_hh12, 'train', m_id+'*.npz'))))

        sample = []
        for f in files:
            with np.load(f) as npz_file:
                data = npz_file['data'][:,1] #data without wavelength
                data = data / np.max(data) # normalisation
                data = data.clip(min=0) #remove values < 0
                sample.append(data)
                if len(sample) > 1:
                    sample = np.mean(sample, axis=0)
                    sample = [list(sample)]
        average_samples.append(sample[0])

    print(len(average_samples) == len(minerals_12_id))

    np.save(os.path.join(path_repo, 'data/average_spectra_handheld'), average_samples)


save_average_minerals()

def save_average_synthetics():
    """
    Saves a numpy file containing the average spectra of all 12 minerals in the synthetic eval dataset
    for the visualisation of the different spectra
    :returns: saves numpy file in libs-pewpew/data/average_spectra_synthetic
    """
    average_samples=[]
    for m_id in minerals_12_id:
        print(m_id)
        m_id = '{0:04d}'.format(m_id)
        files = sorted(glob.glob(os.path.join(path_syn, m_id+'*.npz')))
        sample = []
        for f in files:
            with np.load(f) as npz_file:
                data = npz_file['data'][:,1] #data without wavelength
                data = data / np.max(data) # normalisation
                data = data.clip(min=0) #remove values < 0
                sample.append(data)
                if len(sample) > 1:
                    sample = np.mean(sample, axis=0) #calculate average
                    sample = [list(sample)]
        average_samples.append(sample[0])

    print(len(average_samples) == len(minerals_12_id))

    np.save(os.path.join(path_repo, 'data/average_spectra_synthetic'), average_samples)

save_average_synthetics()
