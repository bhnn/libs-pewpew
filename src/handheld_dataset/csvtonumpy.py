import glob
import os
import shutil
import sys

import numpy as np
from tqdm import tqdm
import yaml


def convert_to_numpy(datasetpath):
    """
    Saves csv files as numpy files in the .npz format and deletes csv files.
    
    :param datasetpath:     path to folder containing all csv files produced by 'organise_hh_dataset.py'
    :returns:               files saved in .npz format, csv files deleted
     """

    files = sorted(glob.glob(os.path.join(datasetpath,'*.csv')))
    for f in tqdm(files, desc='npz-' + datasetpath):
        if os.path.getsize(f) > 250000: # > 250 kB
            data = np.loadtxt(f, skiprows=1, delimiter=',')
            mineral_id, mineral_class, mineral_subgroup, _, _ = f.split(os.sep)[-1].split('_')
            filename = f[:-4] # filename without .csv
            labels = np.asarray((mineral_class, mineral_subgroup, mineral_id)).astype(int)
            np.savez_compressed(filename, data=data, labels=labels)
            os.remove(f) #delete csv files
        else:
            os.remove(f)



def train_test_split(datasetpath, dataset_id):
    """
    Performs a train-test split, does not seperate parts of a measure point in train and another part in test, but
    splits the dataset by splitting the measure points (eg. 10 measurepoints = 7 in train and 3 in test).

    :param datasetpath:     path to folder of the dataset containing all .npz files
    :param dataset_id:      list containing the ids of minerals to iterate through the folder by mineral and split
    :returns:               train and test folders with "_uncorrected" because the spectra are not baseline corrected
    """


    train_samples = list()
    test_samples = list()
    train_labels = list()
    test_labels = list()
    split  = 0.33

    for m_id in tqdm(dataset_id, desc='split-' + datasetpath):
        m_id = f'{m_id:04}'
        files = sorted(glob.glob(os.path.join(datasetpath, m_id + '*.npz')))
        max_mp = int(files[-1][-13:-10])+1 # max measure points, plus 1 because it starts counting from 0

        for f in tqdm(files, leave=False):
            if int(f[-13:-10])+1 <= round(max_mp * split): # performs train-test split of measure points
                try:
                    os.makedirs(os.path.join(datasetpath, 'test_uncorrected'))
                except FileExistsError:
                    pass
                shutil.move(f, os.path.join(datasetpath, 'test_uncorrected'))
            else:
                try:
                    os.makedirs(os.path.join(datasetpath, 'train_uncorrected'))
                except FileExistsError:
                    pass
                shutil.move(f, os.path.join(datasetpath, 'train_uncorrected'))


if __name__ == '__main__':
    with open('config/datasets.yaml') as cnf:
        dataset_configs = yaml.safe_load(cnf)
        try:
            hh_all_path = dataset_configs['hh_all_path']
            hh_12_path = dataset_configs['hh_12_path']
            repo_path = dataset_configs['repo_path']
        except KeyError as e:
            print(f'Missing dataset config key: {e}')
            sys.exit(1)

    #load the mineral ids to iterate through filenames by mineral
    minerals_12_id = np.load(os.path.join(repo_path, 'data/mineral_infos/mineral_id_12.npy'))
    minerals_100_id = np.load(os.path.join(repo_path,'data/mineral_infos/mineral_id_100.npy'))

    #convert files from csv to npz first for hh_12 and hh_all
    convert_to_numpy(hh_12_path)
    convert_to_numpy(hh_all_path)

    #perform train test split for hh_12 and hh_all
    train_test_split(hh_12_path, minerals_12_id)
    train_test_split(hh_all_path, minerals_100_id)
