import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from utils import normalise_minmax, transform_labels

train_samples, train_labels, test_samples, test_labels = np.load('/home/ben/Desktop/ML/synthetic_data/final.npy', allow_pickle=True)
# normalise each sample with its own np.max
train_samples_norm = normalise_minmax(train_samples)
test_samples_norm = normalise_minmax(test_samples)
# create onehot vectors out of labels
train_labels = transform_labels(train_labels, cell=0)
test_labels = transform_labels(test_labels, cell=0)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(train_samples_norm)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

df_targets = pd.DataFrame({'targets': train_labels[:,0].astype(int)})

finalDf = pd.concat([principalDf, df_targets], axis = 1)


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)

targets = [0,1,2,3]
colors = ['r', 'g', 'b','y']

for target, color in zip(targets,colors):
    indicesToKeep = finalDf['targets'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()