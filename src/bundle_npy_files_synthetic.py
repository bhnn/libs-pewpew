import os
from collections import defaultdict
from pathlib import Path
import numpy as np

split = 0.33

minerals_by_group = defaultdict(list)
# load single mineral npy files into dictionary with mineral_id as key
for f in Path('/home/ben/Desktop/ML/synthetic_data').glob('*.npy'):
    sample = np.load(f, allow_pickle=True)
    minerals_by_group[sample[1][2]].append(sample)

train = list()
test  = list()
# split each array of minerals into train and test set for balanced distribution
for key in minerals_by_group.keys():
    total_amt = len(minerals_by_group[key])    
    train.extend(minerals_by_group[key][:int((1-split)*total_amt)])
    test.extend(minerals_by_group[key][int((1-split)*total_amt):])

train = np.asarray(train)
test = np.asarray(test)

# create np arrays with properly visible dimensions and transfer train and test set over
# used for working with tf.data.Dataset later
train_samples_visible_dim = np.zeros((train.shape[0], *train[0,0].shape))
train_labels_visible_dim  = np.zeros((train.shape[0], train[0,1].shape[0]))
test_samples_visible_dim = np.zeros((test.shape[0], *train[0,0].shape))
test_labels_visible_dim  = np.zeros((test.shape[0], train[0,1].shape[0]))

for i in range(len(train)):
    train_samples_visible_dim[i] = train[i, 0]
    train_labels_visible_dim[i]  = train[i, 1]

for i in range(len(test)):
    test_samples_visible_dim[i] = test[i, 0]
    test_labels_visible_dim[i]  = test[i, 1]

final = np.asarray([train_samples_visible_dim, train_labels_visible_dim.astype(int), test_samples_visible_dim, test_labels_visible_dim.astype(int)])
np.save('/home/ben/Desktop/ML/synthetic_data/final', final)