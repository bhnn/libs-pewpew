import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from utils import normalise_minmax, transform_labels

#from sklearn.preprocessing import StandardScaler

train_samples, train_labels, test_samples, test_labels = np.load('/home/ben/Desktop/ML/pretty_data/final.npy', allow_pickle=True)
# normalise each sample with its own np.max
train_samples_norm = normalise_minmax(train_samples)
test_samples_norm = normalise_minmax(test_samples)
# create onehot vectors out of labels
train_labels = transform_labels(train_labels, cell=0)
test_labels = transform_labels(test_labels, cell=0)

pca = PCA(n_components=3)
principalComponents = pca.fit_transform(train_samples_norm)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])

df_targets = pd.DataFrame({'targets': train_labels[:,0].astype(int)})

finalDf = pd.concat([principalDf, df_targets], axis = 1)

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
X = principalComponents
y = train_labels[:,0].astype(int)

# carbonates    - 11, 73, 88
# oxides        - 41, 28, 97
# phosphates    - 80, 86, 35
# sulfides      - 26, 98, 19

for name,label,color in [('Carbonates', 0, 'r'), ('Oxides', 1, 'g'), ('Phosphates', 2, 'b'), ('Sulfides', 3, 'y')]:
    # ax.text3D(X[y == label, 0].mean(),
    #           X[y == label, 1].mean() + 1.5,
    #           X[y == label, 2].mean(), name,
    #           horizontalalignment='center',
    #           bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    ax.scatter(X[y == label, 0], X[y == label, 1], X[y == label, 2], c=color, cmap='Spectral',
           edgecolor='k', alpha=0.4)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()

# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1) 
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 Component PCA', fontsize = 20)

# targets = [0,1,2,3,4]
# colors = ['r', 'g', 'b','y']

# for target, color in zip(targets,colors):
#     indicesToKeep = finalDf['targets'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#                , finalDf.loc[indicesToKeep, 'principal component 2']
#                , c = color
#                , s = 50
#                , alpha = 0.3)
# ax.legend(targets)
# ax.grid()
# plt.show()
