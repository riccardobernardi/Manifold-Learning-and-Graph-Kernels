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
from numpy.linalg import norm
from numexpr.necompiler import evaluate
from sklearn.decomposition import PCA

from pygraham import *

from sklearn import manifold
from sklearn.metrics import pairwise_distances

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

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

# import redis
# r = redis.Redis()
# r.mset({"Croatia": "Zagreb", "Bahamas": "Nassau"})
# r.get("Bahamas")


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


def dominant_set(A, x=None, epsilon=1.0e-4):
	"""Compute the dominant set of the similarity matrix A with the
	replicator dynamics optimization approach. Convergence is reached
	when x changes less than epsilon.
	See: 'Dominant Sets and Pairwise Clustering', by Massimiliano
	Pavan and Marcello Pelillo, PAMI 2007.
	"""
	if x is None:
		x = np.ones(A.shape[0]) / float(A.shape[0])

	distance = epsilon * 2
	while distance > epsilon:
		x_old = x.copy()
		# x = x * np.dot(A, x) # this works only for dense A
		ss = A.dot(x)
		x = evaluate("x * ss")  # this works both for dense and sparse A
		ss = x.sum()
		x = evaluate("x / ss")
		distance = norm( evaluate("x - x_old") )

	return x

class DomSetGraKer():
	def __init__(self):
		self.train_graphs = None

	def similarity(self,g1,g2):
		return jaccard(set(dominating_set(g1)), set(dominating_set(g2)))

	def similarity2(self,g1adj,g2adj):
		ds1list = [i for i,x in enumerate(list(dominant_set(g1adj))) if x>0]
		ds2list = [i for i,x in enumerate(list(dominant_set(g2adj))) if x>0]
		ds1adj = []
		ds2adj = []
		for i in ds1list:
			a = []
			for j in ds1list:
				a += [g1adj[i][j]]
			ds1adj += [a]
		for i in ds2list:
			a = []
			for j in ds2list:
				a += [g2adj[i][j]]
			ds2adj += [a]

		a, b, c = from_adj_to_set(ds1adj)
		d, e, f = from_adj_to_set(ds2adj)

		# tmp = ShortestPath(normalize=True).fit_transform([[a, b, c], [d, e, f]])[0][1]
		tmp = WeisfeilerLehman(n_iter=4, normalize=True).fit_transform([[a, b, c], [d, e, f]])[0][1]

		return tmp

	def fit_transform(self, graphs):
		self.train_graphs = graphs
		kernel_sim = [[0 for _ in range(len(graphs))] for _ in range(len(graphs))]

		for i in range(len(graphs)):
			for j in range(i, len(graphs)):
				#g1 = nx.from_numpy_matrix(from_set_to_adj(graphs[i]))
				#g2 = nx.from_numpy_matrix(from_set_to_adj(graphs[j]))
				g1adj = from_set_to_adj(graphs[i])
				g2adj = from_set_to_adj(graphs[j])
				ss = self.similarity2(g1adj,g2adj)
				kernel_sim[i][j] = ss
				kernel_sim[j][i] = ss

		return copy.deepcopy(kernel_sim)

	def transform(self, graphs):
		kernel_sim = [[0 for _ in range(len(self.train_graphs))] for _ in range(len(graphs))]

		for i in range(len(graphs)):
			for j in range(i, len(self.train_graphs)):
				#g1 = nx.from_numpy_matrix(from_set_to_adj(graphs[i]))
				#g2 = nx.from_numpy_matrix(from_set_to_adj(self.train_graphs[j]))
				#kernel_sim[i][j] = self.similarity2(g1,g2)
				g1adj = from_set_to_adj(graphs[i])
				g2adj = from_set_to_adj(self.train_graphs[j])
				ss = self.similarity2(g1adj, g2adj)
				kernel_sim[i][j] = ss
				kernel_sim[j][i] = ss

		return copy.deepcopy(kernel_sim)

results_PPI = []
results_SHOCK = []