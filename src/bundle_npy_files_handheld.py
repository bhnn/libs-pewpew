import os
from collections import defaultdict
# from pathlib import Path
import glob
import numpy as np
from tqdm import tqdm

split = 0.33

train_samples = list()
test_samples = list()
train_labels = list()
test_labels = list()
for m_id in [11, 19, 26, 28, 35, 41, 73, 80, 86, 88, 97, 98]:
    files = sorted(glob.glob(f'/home/ben/Desktop/ML/pretty_data/00{m_id}*.csv')) 
    max_mp = int(files[-1][46:49])
    for f in tqdm(files):
        if os.path.getsize(f) > 250000: # > 250 kB
            sample = np.loadtxt(f, skiprows=1, delimiter=',')
            labels = np.asarray([int(i) for i in f[33:45].split('_')])
            label_order = [1, 2, 0]
            if int(f[46:49]) <= int(max_mp * split):
                test_samples.append(sample)
                test_labels.append(labels[label_order])
            else:
                train_samples.append(sample)
                train_labels.append(labels[label_order])

test_samples = np.asarray(test_samples)
test_labels = np.asarray(test_labels)
train_samples = np.asarray(train_samples)
train_labels = np.asarray(train_labels)

print(train_samples.shape)
print(train_labels.shape)
print(test_samples.shape)
print(test_labels.shape)

final = np.asarray([train_samples, train_labels, test_samples, test_labels])
np.save('/home/ben/Desktop/ML/pretty_data/final', final)