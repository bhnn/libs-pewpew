import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import os
"""
Plots the results of mixture_mlp.py with 5 repetitions and 10 epochs for the three classification problems
run: python3 src/mixture_mlp.py -r 5 -e 10 -c 0
run: python3 src/mixture_mlp.py -r 5 -e 10 -c 1
run: python3 src/mixture_mlp.py -r 5 -e 10 -c 2
Figure saved in libs-pewpew/data/mixture_results.png
Change path before running
"""


path = r'/Users/jh/github/'

target_0 = np.load(os.path.join(path, 'libs-pewpew/data/synthetic_influence_target_0.npy'))
target_1 = np.load(os.path.join(path, 'libs-pewpew/data/synthetic_influence_target_1.npy'))
target_2 = np.load(os.path.join(path, 'libs-pewpew/data/synthetic_influence_target_2.npy'))

target_0_average = [np.mean(y) for y in target_0]
target_1_average = [np.mean(y) for y in target_1]
target_2_average = [np.mean(y) for y in target_2]

x_values = np.arange(0,100.5, 5)
plt.figure(figsize=[8.5,5])
plt.title('Results for mixture of datasets', fontsize= 16)
plt.xlabel('Percentage of synthetic data added to handheld data', fontsize=14)
plt.ylabel('Accuracy', fontsize= 14)
plt.plot(x_values, target_0_average, label='Classes' , color= '#0d627a')
plt.plot(x_values, target_1_average, label='Groups'  , color= '#f2dc5c')
plt.plot(x_values, target_2_average, label='Minerals', color= '#96cf6d')
plt.legend(fontsize=13)
plt.show()
plt.savefig(os.path.join(path, 'libs-pewpew/data/mixture_results.png'))
