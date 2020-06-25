import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from matplotlib.patches import Rectangle

from utils import has_sufficient_copper, normalise_minmax

train_samples, train_labels, test_samples, test_labels = np.load('/home/ben/Desktop/ML/pretty_data/final.npy', allow_pickle=True)
train_samples_norm = normalise_minmax(train_samples)
test_samples_norm = normalise_minmax(test_samples)
samples = np.vstack((train_samples_norm, test_samples_norm))
num_samples = len(samples)

# bei 30% über 15% kupfer content sein

overall_res = list()
for amount_t in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
    res_per_amount = list()
    for drop_t in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
        res = round(([has_sufficient_copper(s, amount_t, drop_t) for s in samples].count(True) / num_samples), 5)
        res_per_amount.append(res)
    overall_res.append(res_per_amount)

# print(overall_res, '\n')

# for i,amount_t in enumerate([.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]):
#     for j,drop_t in enumerate([.1, .2, .3, .4, .5, .6, .7, .8, .9]):
#         print(f'{amount_t:3}  {drop_t:3}  {overall_res[i][j]}')

fig = plt.figure(figsize = (10,7))
sn.heatmap(overall_res, annot=True, fmt='.4f', yticklabels=[.1, .2, .3, .4, .5, .6, .7, .8, .9], xticklabels=[.1, .2, .3, .4, .5, .6, .7, .8, .9])
plt.ylabel('amount_t')
plt.xlabel('drop_t')

plt.gca().add_patch(Rectangle((0, 2), 2, 1, fill=False, edgecolor='green', lw=3))

plt.show()
plt.savefig('img/copper.png', dpi=fig.dpi, bbox_inches='tight')
