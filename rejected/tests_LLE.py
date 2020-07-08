from utils import *

print("##########################################")
print("----------SHOCK-ST-KERNEL")

# Uses the shortest path kernel to generate the kernel matrices
gk = ShortestPath(normalize=True)
K = gk.fit_transform(G_SHOCK)
D = pairwise_distances(K, metric='euclidean',n_jobs=4)

n_neighbors = 15
n_components = 2
iso_prj_D = manifold.Isomap(n_neighbors, n_components).fit_transform(D)
print(iso_prj_D.shape)

clf = SVC(kernel="linear", C = 1.0)
scores_ln = cross_val_score(clf, iso_prj_D, y_SHOCK, cv = 10, n_jobs= 8)

print(np.mean(scores_ln))