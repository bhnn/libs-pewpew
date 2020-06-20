import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import os
import yaml
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

"""
Plots the results of mixture_mlp.py with 5 repetitions and 10 epochs for the three classification problems
run: python3 src/mixture_mlp.py -r 5 -e 10 -c 0
run: python3 src/mixture_mlp.py -r 5 -e 10 -c 1
run: python3 src/mixture_mlp.py -r 5 -e 10 -c 2
Figure saved in libs-pewpew/data/mixture_results as png and pdf
"""


with open('config/datasets.yaml') as cnf:
    dataset_configs = yaml.safe_load(cnf)

path = dataset_configs['repo_path']

target_0 = np.load(os.path.join(path, 'data/synthetic_influence_target_0.npy'))
target_1 = np.load(os.path.join(path, 'data/synthetic_influence_target_1.npy'))
target_2 = np.load(os.path.join(path, 'data/synthetic_influence_target_2.npy'))

mean0 = np.array([np.mean(y) for y in target_0])
mean1 = np.array([np.mean(y) for y in target_1])
mean2 = np.array([np.mean(y) for y in target_2])
std0 = np.array([np.std(y) for y in target_0])
std1 = np.array([np.std(y) for y in target_1])
std2 = np.array([np.std(y) for y in target_2])

x_values = np.arange(0,100.5, 5)

fig, axs = plt.subplots(3, sharex=True, sharey=True, figsize=(7,7))
fig.subplots_adjust(bottom=0.15, top=0.92, left=0.15)

plt.suptitle('Results for mixture of datasets', fontsize= 16)
plt.xlabel('Percentage of synthetic data added to handheld data', fontsize=10)
plt.ylabel('Accuracy', fontsize= 14)

axs[0].plot(x_values, mean0, label='Classes' , color= '#0d627a')
axs[0].fill_between(x_values, mean0+std0, mean0-std0, alpha=0.5, facecolor= '#0d627a')
axs[0].set_title('Classes', position=(0.9, 0.3), fontsize=10)
axs[0].set_ylabel('Accuracy', fontsize=10)

axs[1].plot(x_values, mean1, label='Groups'  , color= '#f2dc5c')
axs[1].fill_between(x_values, mean1+std1, mean1-std1, alpha=0.5, facecolor='#f2dc5c')
axs[1].set_title('Groups', position=(0.9, 0.3), fontsize=10)
axs[1].set_ylabel('Accuracy', fontsize=10)

axs[2].plot(x_values, mean2, label='Minerals', color= '#96cf6d')
axs[2].fill_between(x_values, mean2+std2, mean2-std2, alpha=0.5, facecolor='#96cf6d')
axs[2].set_title('Minerals', position=(0.9, 0.3), fontsize=10)
axs[2].set_ylabel('Accuracy', fontsize=10)



legend_elements = [Line2D([0], [0], color='black', lw=1, label='Average accuracy'),
                   Patch(facecolor='black', alpha=0.5,
                         label='Standard deviation')]

plt.legend(handles=legend_elements, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.6))

#plt.savefig(os.path.join(path, 'data/visualisations/mixture_results.pdf'))
#plt.savefig(os.path.join(path, 'data/visualisations/mixture_results.png'))

plt.show()
