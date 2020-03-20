import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


path = r'/Volumes/Samsung_T5/LIBSqORE_Austausch'

def snv(input_data):
    """
    Standard normal variate
    Benutze ich jetzt grade nicht, ist unten im code auskommentiert. Mache minmax
    """
    # Define a new array and populate it with the corrected data
    input_data = np.array(input_data)
    data_snv = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Apply correction
        data_snv[i,:] = (input_data[i,:] - np.mean(input_data[i,:])) / np.std(input_data[i,:])
        data_snv[i,:] = data_snv[i,:].clip(min=0)
    return data_snv



def plot_std(datasetname, m_id):
    """
    Plots errorbar for one minerals
    """
    m_id = '{0:04d}'.format(m_id)
    files = sorted(glob.glob(os.path.join(path, datasetname,'train_data', m_id+'*.npz')))
    files.extend(sorted(glob.glob(os.path.join(path, datasetname,'test_data', m_id+'*.npz'))))
    samples = []
    with np.load(files[0]) as npz_file_x:
        x_values = npz_file_x['data'][:,0]

    for f in tqdm(files):
        with np.load(f) as npz_file:
            data = npz_file['data'][:,1] #data without wavelength
            data = data / np.max(data)
            data = data.clip(min=0) #remove values < 0
            samples.append(data)

    samples = np.array(samples)
    #samples = snv(samples)


    samples_mean = np.mean(samples, axis=0)
    samples_std = np.std(samples, axis=0)
    plt.errorbar(x=x_values, y=samples_mean, yerr=samples_std, fmt='.k')
    plt.title('Errorbar for mineral '+m_id)
    plt.show()



#plot_std('hh_6', 30) # mineral mit wenig daten
plot_std('hh_6', 11) # mineral mit viel daten





def plot_spectra(m_id, datasetname):
    """
    Plots average spectra of a mineral
    """

    m_id = '{0:04d}'.format(m_id)
    files = sorted(glob.glob(os.path.join(path, datasetname,'train_data', m_id+'*.npz')))
    files.extend(sorted(glob.glob(os.path.join(path, datasetname,'test_data', m_id+'*.npz'))))

    with np.load(files[0]) as npz_file_x:
        x_values = npz_file_x['data'][:,0]

    for f in tqdm(files):
        samples =[]
        with np.load(f) as npz_file:
            data = npz_file['data'][:,1] #data without wavelength
            data = data / np.max(data) # normalisation
            data = data.clip(min=0) #remove values < 0
            samples.append(data)
            if len(samples) > 1:
                samples = np.mean(samples, axis=0)
                samples = [list(samples)]

    #samples = snv(samples)
    plt.plot(x_values, samples[0])
    plt.title('LIBS Spectra with mineral ID: ' + m_id)
    plt.show()

#plot_spectra(m_id=11, datasetname='hh_6')
