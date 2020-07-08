from sklearn.metrics import pairwise_distances

from utils import *

def launch(ker, results_PPI, results_SHOCK, red=False):
	# Uses the shortest path kernel to generate the kernel matrices
	gk = None
	if ker == "SPK":
		gk = ShortestPath(normalize=True)
	if ker == "WLK":
		gk = WeisfeilerLehman(n_iter=4, normalize=True)
	if ker == "STK":
		gk = NeighborhoodSubgraphPairwiseDistance(r=3, d=2)
	if ker == "DSGK":
		gk = DomSetGraKer()

	for j in ["PPI", "SHOCK"]:
		print("##########################################")
		if red:
			print("----------"+j+"-" + ker + "-KERNEL-RED")
		else:
			print("----------" + j + "-" + ker + "-KERNEL")

		if j=="PPI":
			K = gk.fit_transform(G_PPI)
			D = pairwise_distances(K, metric='euclidean', n_jobs=4)
			y = y_PPI
		else:
			K = gk.fit_transform(G_SHOCK)
			D = pairwise_distances(K, metric='euclidean', n_jobs=4)
			y=y_SHOCK

		if red:
			n_neighbors = 15
			n_components = 2
			iso_prj_D = manifold.Isomap(n_neighbors, n_components).fit_transform(D)

		for i in ["precomputed","linear","rbf"]:
			if red and (i=="precomputed"):
				continue
			start = time()
			print("----------with "+i+" kernel")

			# Uses the SVM classifier to perform classification
			if i=="rbf":
				clf = SVC(kernel=i, gamma="auto", C = 1.0)
			else:
				clf = SVC(kernel=i, C = 1.0)

			if red:
				scores_ln = cross_val_score(clf, iso_prj_D, y, cv=10, n_jobs=8)
			else:
				strat_k_fold = StratifiedKFold(n_splits = 10, shuffle = True) #10
				scores_ln = cross_val_score(clf, D, y, cv = strat_k_fold, n_jobs= 8)
			print(str(np.min(scores_ln)) + " - " +str(np.mean(scores_ln))+ " - " + str(np.max(scores_ln)) + " - "+ str(np.std(scores_ln)))
			acc= np.mean(scores_ln)

			if j=="PPI":
				if red:
					results_PPI += [(ker+"-"+i+"-RED", "Acc: "+str(str(round(acc*100, 2)))+ "%")]
				else:
					results_PPI += [(ker + "-" + i, "Acc: " + str(str(round(acc * 100, 2))) + "%")]
			else:
				if red:
					results_SHOCK += [(ker+"-"+i+"-RED", "Acc: "+str(str(round(acc*100, 2)))+ "%")]
				else:
					results_SHOCK += [(ker + "-" + i, "Acc: " + str(str(round(acc * 100, 2))) + "%")]

	return results_PPI, results_SHOCK
