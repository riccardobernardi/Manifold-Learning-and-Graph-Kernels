from sklearn.metrics import pairwise_distances

from utils import *

def launch(ker, results_PPI, results_SHOCK):
	print("##########################################")
	print("----------PPI-"+ker+"-KERNEL")

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
		if j=="PPI":
			K = gk.fit_transform(G_PPI)
			D = pairwise_distances(K, metric='euclidean', n_jobs=4)
			y = y_PPI
		else:
			K = gk.fit_transform(G_SHOCK)
			D = pairwise_distances(K, metric='euclidean', n_jobs=4)
			y=y_SHOCK
		for i in ["precomputed","linear","rbf"]:
			start = time()
			print("----------with "+i+" kernel")

			# Uses the SVM classifier to perform classification
			if i=="rbf":
				clf = SVC(kernel=i, gamma="auto")
			else:
				clf = SVC(kernel=i)

			strat_k_fold = StratifiedKFold(n_splits = 10, shuffle = True) #10
			scores_ln = cross_val_score(clf, D, y, cv = strat_k_fold, n_jobs= 8)
			print(str(np.min(scores_ln)) +" - "+str(np.mean(scores_ln))+ " - " + str(np.max(scores_ln)) + " - "+ str(np.std(scores_ln)))
			acc= np.mean(scores_ln)

			if j=="PPI":
				results_PPI += [(ker+"-"+i, "Acc: "+str(str(round(acc*100, 2)))+ "%")]
			else:
				results_SHOCK += [(ker + "-" + i, "Acc: " + str(str(round(acc * 100, 2))) + "%")]

	return results_PPI, results_SHOCK
