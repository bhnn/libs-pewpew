import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
import sys

def find_min_max(mineralid, datasetpath, datasetname, repo_path):
    """
    Finds min and max number of spectra per mineral in dataset
    :param mineralid:   mineralids of the dataset
    :param datasetpath: path to the dataset
    :param datasetname: datasetname for visualisation
    :param repo_path:   path to repository
    :returns:           saves visualisation in data/visualisations/datadistribution as pdf and png
    """

    num_spec = []
    for m_id in mineralid:
        m_id = '{0:04d}'.format(m_id)
        files = sorted(glob.glob(os.path.join(datasetpath,'train',m_id + '*.npz')))
        files.extend(sorted(glob.glob(os.path.join(datasetpath,'test', m_id+'*.npz'))))
        num_spec.append(len(files))

    print(datasetname)
    print('Minimum number of spectra per mineral: ',np.min(num_spec))
    print('Maximum number of spectra per mineral: ',np.max(num_spec))
    print('Number of minerals with more than 1000 spectra: ',sum([i>1000 for i in num_spec]))
    print('Number of minerals with less than 1000 spectra: ', sum([i<1000 for i in num_spec]))

    plt.hist(num_spec, bins=30, color='#0d627a')
    plt.title('Distribution of the dataset: '+ datasetname, fontsize=12)
    plt.xlabel('Number of spectra', fontsize=12)
    plt.ylabel('Number of minerals', fontsize=12)
    plt.savefig(os.path.join(repo_path, 'data/visualisations/datadistribution.pdf'))
    plt.savefig(os.path.join(repo_path, 'data/visualisations/datadistribution.png'))




if __name__ == '__main__':

    with open('config/datasets.yaml') as cnf:
        dataset_configs = yaml.safe_load(cnf)
        try:
            repo_path = dataset_configs['repo_path']
            hh_all_path = dataset_configs['hh_all_path']
            hh_all_name = dataset_configs['hh_all_str']
        except KeyError as e:
            print(f'Missing dataset config key: {e}')
            sys.exit(1)

    minerals_all_id = np.load(os.path.join(repo_path, 'data/mineral_infos/mineral_id_100.npy'))
    find_min_max(minerals_all_id, hh_all_path, hh_all_name, repo_path)
