import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from sklearn import svm
import grakel.kernels.graphlet_sampling as graphlet
import pandas as pd
from sklearn.decomposition import PCA

base = "./"
PPI_file = os.path.join(base, "PPI.mat")
SHOCK_file = os.path.join(base, "SHOCK.mat")

PPI = scipy.io.loadmat(PPI_file)
SHOCK = scipy.io.loadmat(SHOCK_file)

#print(PPI.keys())
#print(SHOCK.keys())

#print(PPI["G"].shape)
#print(PPI["labels"].shape)

graphs = [i for i in range(86)]
#print(len(graphs))

data = PPI["G"][0]["am"]
labels = PPI["labels"]
for i in range(len(data)):
    graphs[i] = dict()
    graphs[i]["g"] = data[i]
    graphs[i]["l"] = labels[i][0]
    #print(data[i].shape)

#print(graphs[0])
#print(graphs[0]["g"].shape)
#print(graphs[0]["l"])

k=1
pca = PCA(n_components=k)
# print(np.array(pca.fit(graphs[0]["g"]).components_).flatten())


X = pd.DataFrame([np.array(pca.fit(i["g"]).components_).flatten() for i in graphs]).fillna(0)
#print(X)
Y = [i["l"] for i in graphs]

#print(X[0])
#print(Y)

# we create an instance of SVM and fit out data.
clf = svm.SVC(kernel=graphlet)
clf.fit(X, Y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].

h = .02  # step size in the mesh

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors='k')
plt.title('3-Class classification using Support Vector Machine with custom'
          ' kernel')
plt.axis('tight')
plt.show()