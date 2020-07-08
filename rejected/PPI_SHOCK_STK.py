from utils import *

print("##########################################")
print("----------PPI-SUBGRAPH-KERNEL")

# Initialize neighborhood subgraph pairwise distance kernel
gk = NeighborhoodSubgraphPairwiseDistance(r=3, d=2)
K_train = gk.fit_transform(G_train_PPI)
K_test = gk.transform(G_test_PPI)

start = time()
print("----------with precomputed kernel")
# Uses the SVM classifier to perform classification
clf = SVC(kernel="precomputed")
clf.fit(K_train, y_train_PPI)
y_pred = clf.predict(K_test)

# Computes and prints the classification accuracy
acc = accuracy_score(y_test_PPI, y_pred)
print("Accuracy:", str(round(acc*100, 2)) + "%")

results_PPI += [("STK-KERNEL-precomputed-LLE", "Acc: "+str(str(round(acc*100, 2)))+ "%", str(time()-start))]

start = time()
print("----------with linear kernel")
# Uses the SVM classifier to perform classification
clf = SVC(kernel="linear")
clf.fit(K_train, y_train_PPI)
y_pred = clf.predict(K_test)

# Computes and prints the classification accuracy
acc = accuracy_score(y_test_PPI, y_pred)
print("Accuracy:", str(round(acc*100, 2)) + "%")

results_PPI += [("STK-KERNEL-linear-LLE", "Acc: "+str(str(round(acc*100, 2)))+ "%", str(time()-start))]

start = time()
print("----------with rbf kernel")
# Uses the SVM classifier to perform classification
clf = SVC(kernel="rbf", gamma="auto")
clf.fit(K_train, y_train_PPI)
y_pred = clf.predict(K_test)

# Computes and prints the classification accuracy
acc = accuracy_score(y_test_PPI, y_pred)
print("Accuracy:", str(round(acc*100, 2)) + "%")

results_PPI += [("STK-KERNEL-RBF-LLE", "Acc: "+str(str(round(acc*100, 2)))+ "%", str(time()-start))]

print("##########################################")
print("----------SHOCK-ST-KERNEL")

# Initialize neighborhood subgraph pairwise distance kernel
gk = NeighborhoodSubgraphPairwiseDistance(r=3, d=2)
K_train = gk.fit_transform(G_train_SHOCK)
K_test = gk.transform(G_test_SHOCK)

start = time()
print("----------with precomputed kernel")

# Uses the SVM classifier to perform classification
clf = SVC(kernel="precomputed")
clf.fit(K_train, y_train_SHOCK)
y_pred = clf.predict(K_test)

# Computes and prints the classification accuracy
acc = accuracy_score(y_test_SHOCK, y_pred)
print("Accuracy:", str(round(acc*100, 2)) + "%")

results_SHOCK += [("STK-KERNEL-precomputed-LLE", "Acc: "+str(str(round(acc*100, 2)))+ "%", str(time()-start))]

start = time()
print("----------with linear kernel")

# Uses the SVM classifier to perform classification
clf = SVC(kernel="linear")
clf.fit(K_train, y_train_SHOCK)
y_pred = clf.predict(K_test)

# Computes and prints the classification accuracy
acc = accuracy_score(y_test_SHOCK, y_pred)
print("Accuracy:", str(round(acc*100, 2)) + "%")

results_SHOCK += [("STK-KERNEL-linear-LLE", "Acc: "+str(str(round(acc*100, 2)))+ "%", str(time()-start))]

start = time()
print("----------with rbf kernel")

# Uses the SVM classifier to perform classification
clf = SVC(kernel="rbf", gamma="auto")
clf.fit(K_train, y_train_SHOCK)
y_pred = clf.predict(K_test)

# Computes and prints the classification accuracy
acc = accuracy_score(y_test_SHOCK, y_pred)
print("Accuracy:", str(round(acc*100, 2)) + "%")

results_SHOCK += [("STK-KERNEL-RBF-LLE", "Acc: "+str(str(round(acc*100, 2)))+ "%", str(time()-start))]


