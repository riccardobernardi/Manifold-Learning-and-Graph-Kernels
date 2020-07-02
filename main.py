import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from sklearn import svm
from grakel.kernels import graphlet_sampling
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding

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

#############################################
######## WITHOUT Manifold and Graph Kernel
#############################################

k=1
pca = PCA(n_components=k)
# print(np.array(pca.fit(graphs[0]["g"]).components_).flatten())


X = pd.DataFrame([np.array(pca.fit(i["g"]).components_).flatten() for i in graphs]).fillna(0)
print(X.shape)
Y = [i["l"] for i in graphs]

#print(X[0])
#print(Y)

# we create an instance of SVM and fit out data.
clf = svm.SVC()
clf.fit(X,Y)
print(clf.score(X,Y))

#############################################
######## WITH Manifold and Graph Kernel
#############################################

k=1
embedding = LocallyLinearEmbedding(n_components=k)
# print(np.array(pca.fit(graphs[0]["g"]).components_).flatten())

X = pd.DataFrame([np.array(embedding.fit(i["g"]).embedding_).flatten() for i in graphs]).fillna(0)
print(X.shape)
Y = [i["l"] for i in graphs]

#print(X[0])
#print(Y)

# we create an instance of SVM and fit out data.
clf = svm.SVC()
clf.fit(X,Y)
print(clf.score(X,Y))





