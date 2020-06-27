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


def save_average_minerals(repo_path, hh_12_path, minerals_12_id):
    """
    Saves a numpy file containing the average spectra of all 12 minerals in the hh_12 dataset
    for the visualisation of the different spectra
    :param repo_path:       path to save files
    :param hh_12_path:      path to hh12 dataset
    :param minerals_12_id:  mineral IDs to iterate through files and calculate average spectra per ID
    :returns:               saves numpy file with average spectra in libs-pewpew/data/average_spectra_handheld.npy
    """

    average_samples=[]
    for m_id in minerals_12_id:
        print(m_id)
        m_id = '{0:04d}'.format(m_id)
        files = sorted(glob.glob(os.path.join(hh_12_path, 'test', m_id+'*.npz')))
        files.extend(sorted(glob.glob(os.path.join(hh_12_path, 'train', m_id+'*.npz'))))

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

    np.save(os.path.join(repo_path, 'data/average_spectra_handheld_test'), average_samples)



def save_average_synthetics(repo_path, syn_path, minerals_12_id):
    """
    Saves a numpy file containing the average spectra of all 12 minerals in the synthetic eval dataset
    for the visualisation of the different spectra
    :param repo_path:       path to save files
    :param syn_path:        path to synthetic dataset
    :param minerals_12_id:  mineral IDs to iterate through files and calculate average spectra per ID
    :returns:               saves numpy file with average spectra in libs-pewpew/data/average_spectra_synthetic.npy
    """
    average_samples=[]
    for m_id in minerals_12_id:
        print(m_id)
        m_id = '{0:04d}'.format(m_id)
        files = sorted(glob.glob(os.path.join(syn_path,'eval', m_id+'*.npz')))
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

    np.save(os.path.join(repo_path, 'data/average_spectra_synthetic_test'), average_samples)



if __name__ == '__main__':


    with open('config/datasets.yaml') as cnf:
        dataset_configs = yaml.safe_load(cnf)
        try:
            hh_12_path = dataset_configs['hh_12_path']
            syn_path = dataset_configs['synth_path']
            repo_path = dataset_configs['repo_path']
        except KeyError as e:
            print(f'Missing dataset config key: {e}')
            sys.exit(1)

    minerals_12_id = np.load(os.path.join(repo_path, 'data/mineral_infos/mineral_id_12.npy'))

    #save_average_minerals(repo_path, hh_12_path, minerals_12_id)
    save_average_synthetics(repo_path, syn_path, minerals_12_id)
