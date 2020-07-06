import copy
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

from networkx.algorithms import dominating_set

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
from grakel.kernels.graphlet_sampling import GraphletSampling
from grakel.kernels.random_walk import RandomWalk

from sklearn.manifold import LocallyLinearEmbedding


def from_adj_to_set(adj):
	ss = set()
	lab = {i: 0 for i in range(len(adj))}
	w = dict()
	for idx, i in enumerate(adj):
		for jdx, j in enumerate(i):
			if j == 1:
				ss.add((idx, jdx))
				ss.add((jdx, idx))

	for i in list(ss):
		if adj[i[0]][i[1]] == 1:
			w[i] = 1
		else:
			w[i] = 0

	return ss, lab, w

def from_set_to_adj(set):
	edges = list(set[0])
	nn = max(edges)[0]
	adj = [[0 for _ in range(nn)] for _ in range(nn)]

	for i in edges:
		adj[i[0]-1][i[1]-1] = 1

	return np.array(adj)







########## LOAD THE MATRICES
base = "./"
PPI_file = os.path.join(base, "PPI.mat")
SHOCK_file = os.path.join(base, "SHOCK.mat")

PPI = scipy.io.loadmat(PPI_file)
SHOCK = scipy.io.loadmat(SHOCK_file)

########## LOAD THE PPI

data = PPI["G"][0]["am"]

G_PPI_adj = copy.deepcopy(data)

graphs = [i for i in range(len(data))]

labels = PPI["labels"]
for i in range(len(data)):
	ss, lab, w = from_adj_to_set(data[i])
	graphs[i] = [ss, lab, w]

G = graphs
y = [i[0] for i in labels]

y_PPI_adj = copy.deepcopy(y)

G_PPI = copy.deepcopy(G)
y_PPI = copy.deepcopy(y)

########## LOAD THE SHOCK

data = SHOCK["G"][0]["am"]

G_SHOCK_adj = copy.deepcopy(data)

graphs = [i for i in range(len(data))]

labels = SHOCK["labels"]
for i in range(len(data)):
	ss, lab, w = from_adj_to_set(data[i])
	graphs[i] = [ss, lab, w]

G = graphs
y = [i[0] for i in labels]

y_SHOCK_adj = copy.deepcopy(y)

G_SHOCK = copy.deepcopy(G)
y_SHOCK = copy.deepcopy(y)

G_train_PPI, G_test_PPI, y_train_PPI, y_test_PPI = train_test_split(G_PPI, y_PPI, test_size=0.1, random_state=25061997)
G_train_SHOCK, G_test_SHOCK, y_train_SHOCK, y_test_SHOCK = train_test_split(G_SHOCK, y_SHOCK, test_size=0.1, random_state=25061997)

def jaccard(A,B):
	return len(list(A & B))/len(list(A | B))

def pairwiseDSGK(G_name_adj):
	kernel_sim = [[0 for _ in range(len(G_name_adj))] for _ in range(len(G_name_adj))]

	for i in range(len(G_name_adj)):
		for j in range(len(G_name_adj)):
			g1 = nx.from_numpy_matrix(from_set_to_adj(G_name_adj[i]))
			g2 = nx.from_numpy_matrix(from_set_to_adj(G_name_adj[j]))
			kernel_sim[i][j] = jaccard(set(dominating_set(g1)),set(dominating_set(g2)))

	return copy.deepcopy(kernel_sim)

def pairwiseDSGKtest(G_name_adj, train):
	kernel_sim = [[0 for _ in range(len(train))] for _ in range(len(G_name_adj))]

	for i in range(len(G_name_adj)):
		for j in range(len(train)):
			g1 = nx.from_numpy_matrix(from_set_to_adj(G_name_adj[i]))
			g2 = nx.from_numpy_matrix(from_set_to_adj(train[j]))
			kernel_sim[i][j] = jaccard(set(dominating_set(g1)),set(dominating_set(g2)))

	return copy.deepcopy(kernel_sim)

class DomSetGraKer():
	def __init__(self):
		self.train_graphs = None

	def fit_transform(self, graphs):
		self.train_graphs = graphs
		return pairwiseDSGK(graphs)

	def transform(self, graphs):
		return pairwiseDSGKtest(graphs, self.train_graphs)

results_PPI = []
results_SHOCK = []