import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer
from sklearn import decomposition

# Get training Data
df = pd.read_csv('../input/train.csv')

xTrain = df.drop(['ID', 'From_Cache_la_Poudre'], axis=1)
yTrain = df['From_Cache_la_Poudre']

fig1 = plt.figure(1)
plt.clf()
ax = Axes3D(fig1, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()
plt.title('Wilderness Area Classification PCA 3-Dimensions')

pca = decomposition.PCA(n_components=3)
pca.fit(xTrain)
X = pca.transform(xTrain)
color = ['red', 'blue']
for name, label in [('Not Cache la Poudre', 0), ('Cache la Poudre', 1)]:
    ax.scatter(X[yTrain == label, 0], X[yTrain == label, 1], X[yTrain == label, 2], c=color[label], cmap=plt.cm.nipy_spectral, edgecolor='k', label=name)
plt.legend(loc=3)

fig2 = plt.figure(2)
pca = decomposition.PCA(n_components=2)
pca.fit(xTrain)
X = pca.transform(xTrain)

for name, label in [('Not Cache la Poudre', 0), ('Cache la Poudre', 1)]:
    plt.scatter(X[yTrain == label, 0], X[yTrain == label, 1], c=color[label], label=name)
plt.title('Wilderness Area Classification PCA 2-Dimensions')
plt.legend()
plt.show()