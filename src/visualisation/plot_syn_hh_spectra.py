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


def plot_syn_hh(repo_path, mineral_names):
    """
    Visualizes the average spectrum of 4 minerals from the handheld dataset and the synthetic dataset
    :param repo_path:       path to load average spectra files
    :param mineral_names:   Names of the minerals
    :returns:               8 subplots with the average spectrum for each of the minerals
                            Saves the plots in data/average_spectra_hh_syn as png and pdf
    """

    average_handheld =  np.load(os.path.join(repo_path, 'data/average_spectra_handheld.npy'))
    average_syn = np.load(os.path.join(repo_path, 'data/average_spectra_synthetic.npy'))
    x_values = np.arange(180, 961, 0.1)

    fig, axs = plt.subplots(4,2, sharex=True, figsize=(10,5))
    fig.subplots_adjust(bottom=0.2, top=0.9, left=0.07)

    axs[0,0].plot(x_values, average_handheld[0],   color= '#0d627a', label='Sulfides')
    axs[1,0].plot(x_values, average_syn[0],        color= '#0d627a')
    axs[2,0].plot(x_values, average_handheld[3],   color= '#f2dc5c',  label='Oxides')
    axs[3,0].plot(x_values, average_syn[3],        color= '#f2dc5c')
    axs[0,1].plot(x_values, average_handheld[6],   color= '#96cf6d',  label='Carbonates')
    axs[1,1].plot(x_values, average_syn[6],        color= '#96cf6d')
    axs[2,1].plot(x_values, average_handheld[10],   color= '#0c907d',  label='Phosphates')
    axs[3,1].plot(x_values, average_syn[10],        color= '#0c907d')


    plt.xlabel('Wavelength', fontsize=8)
    fig.suptitle('Average handheld and synthetic LIBS spectra of 4 minerals', fontsize= 13)
    fig.legend(title='Mineral classes',loc='lower center', bbox_to_anchor=(0.5, 0), ncol=4)

    for j in range(2):
        for i in range(4):
            axs[i,j].yaxis.set_visible(False) # Hide y axis
            axs[i,j].set_title(mineral_names[(i+j*4)], position=(0.87, 0.4), fontsize=10) #add mineralnames
    plt.savefig(os.path.join(repo_path,'data/visualisations/average_spectra_hh_syn.png'))
    plt.savefig(os.path.join(repo_path,'data/visualisations/average_spectra_hh_syn.pdf'))


if __name__ == '__main__':

    with open('config/datasets.yaml') as cnf:
        dataset_configs = yaml.safe_load(cnf)
        try:
            repo_path = dataset_configs['repo_path']
        except KeyError as e:
            print(f'Missing dataset config key: {e}')
            sys.exit(1)

    mineral_names = ['Chalcopyrite', 'Chalcopyrite \n synthetic', 'Cuprite','Cuprite \n synthetic', 'Rosasite', 'Rosasite \n synthetic', 'Olivenite', 'Olivenite \n synthetic']

    plot_syn_hh(repo_path, mineral_names)
