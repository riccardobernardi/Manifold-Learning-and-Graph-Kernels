from sklearn.metrics import pairwise_distances

from utils import *

print("##########################################")
print("----------PPI-SP-KERNEL")

# Uses the shortest path kernel to generate the kernel matrices
gk = ShortestPath(normalize=True)
K = gk.fit_transform(G_PPI)
D = pairwise_distances(K, metric='euclidean',n_jobs=4)
# K_train = gk.fit_transform(G_train_PPI)
# K_test = gk.transform(G_test_PPI)

start = time()
print("----------with precomputed kernel")
# Uses the SVM classifier to perform classification
clf = SVC(kernel="precomputed")
# clf.fit(K_train, y_train_PPI)
# y_pred = clf.predict(K_test)

strat_k_fold = StratifiedKFold(n_splits = 10, shuffle = True) #10
scores_ln = cross_val_score(clf, D, y_PPI, cv = strat_k_fold, n_jobs= 8)
print(str(np.min(scores_ln)) +" - "+str(np.mean(scores_ln))+ " - " + str(np.max(scores_ln)) + " - "+ str(np.std(scores_ln)))


# Computes and prints the classification accuracy
# acc = accuracy_score(y_test_PPI, y_pred)
# print("Accuracy:", str(round(acc*100, 2)) + "%")
#
# results_PPI += [("STK-KERNEL-precomputed", "Acc: "+str(str(round(acc*100, 2)))+ "%", str(time()-start))]
#
# start = time()
print("----------with linear kernel")
# Uses the SVM classifier to perform classification
clf = SVC(kernel="linear")
# clf.fit(K_train, y_train_PPI)
# y_pred = clf.predict(K_test)

strat_k_fold = StratifiedKFold(n_splits = 10, shuffle = True) #10
scores_ln = cross_val_score(clf, D, y_PPI, cv = strat_k_fold, n_jobs= 8)
print(str(np.min(scores_ln)) +" - "+str(np.mean(scores_ln))+ " - " + str(np.max(scores_ln)) + " - "+ str(np.std(scores_ln)))


# Computes and prints the classification accuracy
# acc = accuracy_score(y_test_PPI, y_pred)
# print("Accuracy:", str(round(acc*100, 2)) + "%")
#
# results_PPI += [("STK-KERNEL-linear", "Acc: "+str(str(round(acc*100, 2)))+ "%", str(time()-start))]

# start = time()
print("----------with rbf kernel")
# Uses the SVM classifier to perform classification
clf = SVC(kernel="rbf", gamma="auto")
# clf.fit(K_train, y_train_PPI)
# y_pred = clf.predict(K_test)

# Computes and prints the classification accuracy
# acc = accuracy_score(y_test_PPI, y_pred)
# print("Accuracy:", str(round(acc*100, 2)) + "%")
#
# results_PPI += [("STK-KERNEL-RBF", "Acc: "+str(str(round(acc*100, 2)))+ "%", str(time()-start))]

strat_k_fold = StratifiedKFold(n_splits = 10, shuffle = True) #10
scores_ln = cross_val_score(clf, D, y_PPI, cv = strat_k_fold, n_jobs= 8)
print(str(np.min(scores_ln)) +" - "+str(np.mean(scores_ln))+ " - " + str(np.max(scores_ln)) + " - "+ str(np.std(scores_ln)))


print("##########################################")
print("----------SHOCK-ST-KERNEL")

# Uses the shortest path kernel to generate the kernel matrices
gk = ShortestPath(normalize=True)
K = gk.fit_transform(G_SHOCK)
D = pairwise_distances(K, metric='euclidean',n_jobs=4)
# K_train = gk.fit_transform(G_train_SHOCK)
# K_test = gk.transform(G_test_SHOCK)
# K_train = pairwise_distances(K_train, metric='euclidean',n_jobs=4)
# K_test = pairwise_distances(K_test, metric='euclidean',n_jobs=4)

# start = time()
print("----------with precomputed kernel")

# Uses the SVM classifier to perform classification
clf = SVC(kernel="precomputed")
# clf.fit(K_train, y_train_SHOCK)
# y_pred = clf.predict(K_test)


strat_k_fold = StratifiedKFold(n_splits = 10, shuffle = True) #10
scores_ln = cross_val_score(clf, D, y_SHOCK, cv = strat_k_fold, n_jobs= 8)
print(str(np.min(scores_ln)) +" - "+str(np.mean(scores_ln))+ " - " + str(np.max(scores_ln)) + " - "+ str(np.std(scores_ln)))

# Computes and prints the classification accuracy
# acc = accuracy_score(y_test_SHOCK, y_pred)
# print("Accuracy:", str(round(acc*100, 2)) + "%")
#
# results_SHOCK += [("STK-KERNEL-precomputed", "Acc: "+str(str(round(acc*100, 2)))+ "%", str(time()-start))]

# start = time()
print("----------with linear kernel")

# Uses the SVM classifier to perform classification
clf = SVC(kernel="linear")
# clf.fit(K_train, y_train_SHOCK)
# y_pred = clf.predict(K_test)

strat_k_fold = StratifiedKFold(n_splits = 10, shuffle = True) #10
scores_ln = cross_val_score(clf, D, y_SHOCK, cv = strat_k_fold, n_jobs= 8)
print(str(np.min(scores_ln)) +" - "+str(np.mean(scores_ln))+ " - " + str(np.max(scores_ln)) + " - "+ str(np.std(scores_ln)))


# Computes and prints the classification accuracy
# acc = accuracy_score(y_test_SHOCK, y_pred)
# print("Accuracy:", str(round(acc*100, 2)) + "%")
#
# results_SHOCK += [("STK-KERNEL-linear", "Acc: "+str(str(round(acc*100, 2)))+ "%", str(time()-start))]
#
# start = time()
print("----------with rbf kernel")

# Uses the SVM classifier to perform classification
clf = SVC(kernel="rbf", gamma="auto")
# clf.fit(K_train, y_train_SHOCK)
# y_pred = clf.predict(K_test)

strat_k_fold = StratifiedKFold(n_splits = 10, shuffle = True) #10
scores_ln = cross_val_score(clf, D, y_SHOCK, cv = strat_k_fold, n_jobs= 8)
print(str(np.min(scores_ln)) +" - "+str(np.mean(scores_ln))+ " - " + str(np.max(scores_ln)) + " - "+ str(np.std(scores_ln)))


# Computes and prints the classification accuracy
# acc = accuracy_score(y_test_SHOCK, y_pred)
# print("Accuracy:", str(round(acc*100, 2)) + "%")
#
# results_SHOCK += [("STK-KERNEL-RBF", "Acc: "+str(str(round(acc*100, 2)))+ "%", str(time()-start))]
#

