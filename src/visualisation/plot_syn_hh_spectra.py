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

path = dataset_configs['repo_path']

mineral_names = ['Chalcopyrite', 'Chalcopyrite \n synthetic', 'Cuprite','Cuprite \n synthetic', 'Rosasite', 'Rosasite \n synthetic', 'Olivenite', 'Olivenite \n synthetic']


def plot_syn_hh():
    """
    Visualizes the average spectrum of 4 minerals from the handheld dataset and the synthetic dataset
    :input:     average spectra
    :returns:   8 subplots with the average spectrum for each of the minerals
                Saves the plots in data/average_spectra_hh_syn as png and pdf
    """

    average_handheld =  np.load(os.path.join(path, 'data/average_spectra_handheld.npy'))
    average_syn = np.load(os.path.join(path, 'data/average_spectra_synthetic.npy'))
    x_values = np.arange(180, 961, 0.1)

    fig, axs = plt.subplots(4,2, sharex=True, figsize=(10,5))
    fig.subplots_adjust(bottom=0.15, top=0.92, left=0.07)

    axs[0,0].plot(x_values, average_handheld[0],   color= '#0d627a', label='Sulfides')
    axs[1,0].plot(x_values, average_syn[0],        color= '#0d627a')
    axs[2,0].plot(x_values, average_handheld[3],   color= '#f2dc5c',  label='Oxides')
    axs[3,0].plot(x_values, average_syn[3],        color= '#f2dc5c')
    axs[0,1].plot(x_values, average_handheld[6],   color= '#96cf6d',  label='Carbonates')
    axs[1,1].plot(x_values, average_syn[6],        color= '#96cf6d')
    axs[2,1].plot(x_values, average_handheld[10],   color= '#0c907d',  label='Phosphates')
    axs[3,1].plot(x_values, average_syn[10],        color= '#0c907d')


    plt.xlabel('Wavelength', fontsize=8)
    fig.suptitle('Average handheld and synthetic LIBS spectra of 4 minerals', fontsize= 10)
    fig.legend(title='Mineral classes',loc='lower center', bbox_to_anchor=(0.5, 0), ncol=4)

    for j in range(2):
        for i in range(4):
            print('J',j)
            print('I',i)
            axs[i,j].yaxis.set_visible(False) # Hide y axis
            axs[i,j].set_title(mineral_names[(i+j*4)], position=(0.85, 0.3), fontsize=10) #add mineralnames
    plt.savefig(os.path.join(r'/Users/jh/github/libs-pewpew/data/visualisations/average_spectra_hh_syn.png'))
    plt.savefig(os.path.join(r'/Users/jh/github/libs-pewpew/data/visualisations/average_spectra_hh_syn.pdf'))

plot_syn_hh()
