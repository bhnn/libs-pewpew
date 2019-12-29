import os
from collections import defaultdict
from pathlib import Path
import numpy as np

split = 0.33

minerals_by_group = defaultdict(list)
for f in Path('/home/ben/Desktop/ML/synthetic_data').glob('*.npy'):
    sample = np.load(f, allow_pickle=True)
    minerals_by_group[sample[1][2]].append(sample)

train = list()
test  = list()
for key in minerals_by_group.keys():
    total_amt = len(minerals_by_group[key])    
    train.extend(minerals_by_group[key][:int((1-split)*total_amt)])
    test.extend(minerals_by_group[key][int((1-split)*total_amt):])

train = np.array(train)
test = np.array(test)

final = np.array([train[:, 0], train[:, 1], test[:, 0], test[:, 1]])
np.save('/home/ben/Desktop/ML/synthetic_data/final', final)