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
    :returns: saves numpy file in libs-pewpew/data/average_spectra
    """
    average_samples=[]
    for m_id in minerals_12_id:
        print(m_id)
        m_id = '{0:04d}'.format(m_id)
        files = sorted(glob.glob(os.path.join(path, 'hh_12/test_corrected', m_id+'*.npz')))
        min_mp = int(files[0][-13:-10])
        max_mp = int(files[-1][-13:-10])
        for mp in range(min_mp, max_mp+1):
            sample = []
            mp = '{0:03d}'.format(mp)
            mp = str(mp)
            files_mp = [f for f in files if f.find(mp,-13,-10) != -1]
            for f in files_mp:
                with np.load(f) as npz_file:
                    data = npz_file['data'][:,1] #data without wavelength
                    data = data / np.max(data) # normalisation
                    data = data.clip(min=0) #remove values < 0
                    sample.append(data)
                    if len(sample) > 1:
                        sample = np.mean(sample, axis=0)
                        sample = [list(sample)]
            average_samples.append(sample[0])

    np.save(os.path.join(path, 'libs-pewpew/data/average_spectra'), average_samples)


save_average_minerals()


def plot_several():
    """
    Plots 12 average spectra below one another and saves the plot
    :returns: Saves the plot in libs-pewpew/data/several_spectra.png
    """

    average_samples =  np.load(os.path.join(path, 'libs-pewpew/data/average_spectra.npy'))
    x_values = np.arange(180, 961, 0.1)

    fig, axs = plt.subplots(12, sharex=True, figsize=(7,7))
    fig.subplots_adjust(bottom=0.15, top=0.92)

    axs[0].plot(x_values, average_samples[0],   color= '#0d627a', label='Sulfides')
    axs[1].plot(x_values, average_samples[1],   color= '#0d627a')
    axs[2].plot(x_values, average_samples[2],   color= '#0d627a')
    axs[3].plot(x_values, average_samples[3],   color='#f2dc5c',  label='Oxides')
    axs[4].plot(x_values, average_samples[4],   color='#f2dc5c')
    axs[5].plot(x_values, average_samples[5],   color='#f2dc5c')
    axs[6].plot(x_values, average_samples[6],   color='#96cf6d',  label='Carbonates')
    axs[7].plot(x_values, average_samples[7],   color='#96cf6d')
    axs[8].plot(x_values, average_samples[8],   color='#96cf6d')
    axs[9].plot(x_values, average_samples[9],   color='#0c907d',  label='Phosphates')
    axs[10].plot(x_values, average_samples[10], color='#0c907d')
    axs[11].plot(x_values, average_samples[11], color='#0c907d')

    plt.xlabel('Wavelength', fontsize=8)
    fig.suptitle('Average LIBS spectra of 12 minerals', fontsize= 11)
    fig.legend(title='Mineral classes',loc='lower center', bbox_to_anchor=(0.5, 0), ncol=4)

    for i in range(12):
        axs[i].yaxis.set_visible(False) # Hide y axis
        axs[i].set_title(mineral_names[i], position=(0.85, 0.3), fontsize=10) #add mineralnames

    plt.savefig(os.path.join(r'/Users/jh/github/libs-pewpew/data/average_spectra_plot.png'))

plot_several()
