
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from sklearn import svm
from grakel.kernels import graphlet_sampling
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
from grakel.graph import Graph
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from grakel.datasets import fetch_dataset
from grakel.kernels import ShortestPath
import networkx as nx

from time import time
from grakel.kernels import NeighborhoodSubgraphPairwiseDistance
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from grakel.datasets import fetch_dataset
from grakel.kernels import WeisfeilerLehman, VertexHistogram

base = "./"
PPI_file = os.path.join(base, "PPI.mat")
SHOCK_file = os.path.join(base, "SHOCK.mat")

PPI = scipy.io.loadmat(PPI_file)
SHOCK = scipy.io.loadmat(SHOCK_file)

graphs = [i for i in range(86)]


def from_adj_to_set(adj):
    ss = set()
    lab = {i:0 for i in range(len(adj))}
    w = dict()
    for idx,i in enumerate(adj):
        for jdx,j in enumerate(i):
            if j==1:
                ss.add((idx,jdx))
                ss.add((jdx, idx))

    for i in list(ss):
        if adj[i[0]][i[1]] == 1:
            w[i]=1
        else:
            w[i] = 0

    return ss,lab,w



def launch(function):
	try:

		name = function

		print("##########################################")
		print("----------PPI-",name,"-KERNEL")

		data = PPI["G"][0]["am"]

		graphs = [i for i in range(len(data))]

		labels = PPI["labels"]
		for i in range(len(data)):
			ss,lab,w = from_adj_to_set(data[i])
			graphs[i] = [ss,lab,w]

		G = graphs
		y = [i[0] for i in labels]

		G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.1, random_state=25061997)

		# Initialize neighborhood subgraph pairwise distance kernel
		gk = function()
		K_train = gk.fit_transform(G_train)
		K_test = gk.transform(G_test)

		print("----------with precomputed kernel")
		# Uses the SVM classifier to perform classification
		clf = SVC(kernel="precomputed")
		clf.fit(K_train, y_train)
		y_pred = clf.predict(K_test)

		# Computes and prints the classification accuracy
		acc = accuracy_score(y_test, y_pred)
		print("Accuracy:", str(round(acc*100, 2)) + "%")

		print("----------with linear kernel")
		# Uses the SVM classifier to perform classification
		clf = SVC(kernel="linear")
		clf.fit(K_train, y_train)
		y_pred = clf.predict(K_test)

		# Computes and prints the classification accuracy
		acc = accuracy_score(y_test, y_pred)
		print("Accuracy:", str(round(acc*100, 2)) + "%")

		print("----------with rbf kernel")
		# Uses the SVM classifier to perform classification
		clf = SVC(kernel="rbf", gamma="auto")
		clf.fit(K_train, y_train)
		y_pred = clf.predict(K_test)

		# Computes and prints the classification accuracy
		acc = accuracy_score(y_test, y_pred)
		print("Accuracy:", str(round(acc*100, 2)) + "%")


		print("##########################################")
		print("----------SHOCK-",name,"-KERNEL")

		data = SHOCK["G"][0]["am"]

		graphs = [i for i in range(len(data))]

		labels = SHOCK["labels"]
		for i in range(len(data)):
			ss,lab,w = from_adj_to_set(data[i])
			graphs[i] = [ss,lab,w]

		G = graphs
		y = [i[0] for i in labels]

		G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.1, random_state=25061997)

		# Initialize neighborhood subgraph pairwise distance kernel
		gk = function()
		K_train = gk.fit_transform(G_train)
		K_test = gk.transform(G_test)

		print("----------with precomputed kernel")

		# Uses the SVM classifier to perform classification
		clf = SVC(kernel="precomputed")
		clf.fit(K_train, y_train)
		y_pred = clf.predict(K_test)

		# Computes and prints the classification accuracy
		acc = accuracy_score(y_test, y_pred)
		print("Accuracy:", str(round(acc*100, 2)) + "%")

		print("----------with linear kernel")

		# Uses the SVM classifier to perform classification
		clf = SVC(kernel="linear")
		clf.fit(K_train, y_train)
		y_pred = clf.predict(K_test)

		# Computes and prints the classification accuracy
		acc = accuracy_score(y_test, y_pred)
		print("Accuracy:", str(round(acc*100, 2)) + "%")

		print("----------with rbf kernel")

		# Uses the SVM classifier to perform classification
		clf = SVC(kernel="rbf", gamma="auto")
		clf.fit(K_train, y_train)
		y_pred = clf.predict(K_test)

		# Computes and prints the classification accuracy
		acc = accuracy_score(y_test, y_pred)
		print("Accuracy:", str(round(acc*100, 2)) + "%")
	except Exception as e:
		print(e.with_traceback())
