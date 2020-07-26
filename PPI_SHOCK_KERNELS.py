from sklearn.metrics import pairwise_distances

from utils import *

def launch(ker, red, results_PPI, results_SHOCK, n_iter=4,n_neighbors = 15, n_components = 2):
	# Uses the shortest path kernel to generate the kernel matrices
	gk = None
	if ker == "SPK":
		gk = ShortestPath(normalize=True)
	if ker == "WLK":
		gk = WeisfeilerLehman(n_iter=n_iter, normalize=True)
		ker = "WLK"+str(n_iter)
	if ker == "STK":
		gk = NeighborhoodSubgraphPairwiseDistance(r=3, d=2)
	if ker == "DSGK":
		gk = DomSetGraKer(n_iter=n_iter)
		ker = "DSGK"+str(n_iter)

	if red == "NOP":
		red = "no-RED"

	for j in ["PPI", "SHOCK"]:
		print("##########################################")
		print("----------"+j+"-" + ker + "-" + red)

		if j=="PPI":
			K = gk.fit_transform(G_PPI)
			D = pairwise_distances(np.array(K), metric='euclidean', n_jobs=4)
			y = y_PPI
		else:
			K = gk.fit_transform(G_SHOCK)
			D = pairwise_distances(K, metric='euclidean', n_jobs=4)
			y=y_SHOCK

		iso_prj_D = D

		if red == "NOP":
			red="no-RED"

		if red == "ISO":
			iso_prj_D = manifold.Isomap(n_neighbors, n_components).fit_transform(D)

		if red == "PCA":
			iso_prj_D = PCA(n_neighbors, n_components).fit_transform(D)

		if red == "LLE":
			iso_prj_D = manifold.LocallyLinearEmbedding(n_components).fit_transform(D)

		if red == "SE":
			iso_prj_D = manifold.SpectralEmbedding(n_components).fit_transform(D)

		if red == "MDS":
			iso_prj_D = manifold.MDS(n_components).fit_transform(D)

		if red == "TSNE":
			iso_prj_D = manifold.TSNE(n_components).fit_transform(D)

		for i in ["precomputed","linear","rbf"]:
			if (red!="no-RED") and (i=="precomputed"):
				continue
			start = time()
			print("----------with "+i+" kernel")

			# Uses the SVM classifier to perform classification
			if i=="rbf":
				clf = SVC(kernel=i, gamma="auto", C = 1.0)
			else:
				clf = SVC(kernel=i, C = 1.0)

			if red!="no-RED":
				scores_ln = cross_val_score(clf, iso_prj_D, y, cv=10, n_jobs=8)
			else:
				strat_k_fold = StratifiedKFold(n_splits = 10, shuffle = True) #10
				scores_ln = cross_val_score(clf, D, y, cv = strat_k_fold, n_jobs= 8)
			print(str(np.min(scores_ln)) + " - " +str(np.mean(scores_ln))+ " - " + str(np.max(scores_ln)) + " - "+ str(np.std(scores_ln)))

			#acc= np.mean(scores_ln)

			pd_name = ker+"-"+i+"-"+red
			#pd_acc = "Acc: "+str(str(round(acc*100, 2)))+ "%"
			pd_acc = "Acc: min " + str(np.min(scores_ln))[:4] + " - avg " +str(np.mean(scores_ln))[:4]+ " - max " + str(np.max(scores_ln))[:4] + " - std "+ str(np.std(scores_ln))[:5]

			if j == "PPI":
				plt.scatter(iso_prj_D[:, 0], iso_prj_D[:, 1],c=y_PPI)
				plt.savefig(os.path.join("./images", j+"-"+pd_name + '.png'))
				plt.show()
			else:
				plt.scatter(iso_prj_D[:, 0], iso_prj_D[:, 1], c=y_SHOCK)
				plt.savefig(os.path.join("./images", j+"-" + pd_name + '.png'))
				plt.show()

			if j=="PPI":
				results_PPI += [(pd_name, pd_acc)]
			else:
				results_SHOCK += [(pd_name, pd_acc)]

	return results_PPI, results_SHOCK
