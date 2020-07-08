from utils import *

print("##########################################")
print("----------PPI-WL-KERNEL-LLE")

# Uses the Weisfeiler-Lehman subtree kernel to generate the kernel matrices
gk = WeisfeilerLehman(n_iter=4, normalize=True)
K_train = gk.fit_transform(G_train_PPI)
K_test = gk.transform(G_test_PPI)

embedding = LocallyLinearEmbedding(n_components=3)
K_train = embedding.fit_transform(K_train)
embedding = LocallyLinearEmbedding(n_components=3)
K_test = embedding.fit_transform(K_test)

start = time()
print("----------with precomputed kernel")
# Uses the SVM classifier to perform classification
clf = SVC(kernel="precomputed")
clf.fit(K_train, y_train_PPI)
y_pred = clf.predict(K_test)

# Computes and prints the classification accuracy
acc = accuracy_score(y_test_PPI, y_pred)
print("Accuracy:", str(round(acc*100, 2)) + "%")

results_PPI += [("WLK-KERNEL-precomputed-LLE", "Acc: "+str(str(round(acc*100, 2)))+ "%", str(time()-start))]

start = time()
print("----------with linear kernel")
# Uses the SVM classifier to perform classification
clf = SVC(kernel="linear")
clf.fit(K_train, y_train_PPI)
y_pred = clf.predict(K_test)

# Computes and prints the classification accuracy
acc = accuracy_score(y_test_PPI, y_pred)
print("Accuracy:", str(round(acc*100, 2)) + "%")

results_PPI += [("WLK-KERNEL-linear-LLE", "Acc: "+str(str(round(acc*100, 2)))+ "%", str(time()-start))]

start = time()
print("----------with rbf kernel")
# Uses the SVM classifier to perform classification
clf = SVC(kernel="rbf", gamma="auto")
clf.fit(K_train, y_train_PPI)
y_pred = clf.predict(K_test)

# Computes and prints the classification accuracy
acc = accuracy_score(y_test_PPI, y_pred)
print("Accuracy:", str(round(acc*100, 2)) + "%")

results_PPI += [("WLK-KERNEL-RBF-LLE", "Acc: "+str(str(round(acc*100, 2)))+ "%", str(time()-start))]

print("##########################################")
print("----------SHOCK-WL-KERNEL-LLE")

# Uses the Weisfeiler-Lehman subtree kernel to generate the kernel matrices
gk = WeisfeilerLehman(n_iter=4, normalize=True)
K_train = gk.fit_transform(G_train_SHOCK)
K_test = gk.transform(G_test_SHOCK)

embedding = LocallyLinearEmbedding(n_components=3)
K_train = embedding.fit_transform(K_train)
embedding = LocallyLinearEmbedding(n_components=3)
K_test = embedding.fit_transform(K_test)

start = time()
print("----------with precomputed kernel")

# Uses the SVM classifier to perform classification
clf = SVC(kernel="precomputed")
clf.fit(K_train, y_train_SHOCK)
y_pred = clf.predict(K_test)

# Computes and prints the classification accuracy
acc = accuracy_score(y_test_SHOCK, y_pred)
print("Accuracy:", str(round(acc*100, 2)) + "%")

results_SHOCK += [("WLK-KERNEL-precomputed-LLE", "Acc: "+str(str(round(acc*100, 2)))+ "%", str(time()-start))]

start = time()
print("----------with linear kernel")

# Uses the SVM classifier to perform classification
clf = SVC(kernel="linear")
clf.fit(K_train, y_train_SHOCK)
y_pred = clf.predict(K_test)

# Computes and prints the classification accuracy
acc = accuracy_score(y_test_SHOCK, y_pred)
print("Accuracy:", str(round(acc*100, 2)) + "%")

results_SHOCK += [("WLK-KERNEL-linear-LLE", "Acc: "+str(str(round(acc*100, 2)))+ "%", str(time()-start))]

start = time()
print("----------with rbf kernel")

# Uses the SVM classifier to perform classification
clf = SVC(kernel="rbf", gamma="auto")
clf.fit(K_train, y_train_SHOCK)
y_pred = clf.predict(K_test)

# Computes and prints the classification accuracy
acc = accuracy_score(y_test_SHOCK, y_pred)
print("Accuracy:", str(round(acc*100, 2)) + "%")

results_SHOCK += [("WLK-KERNEL-RBF-LLE", "Acc: "+str(str(round(acc*100, 2)))+ "%", str(time()-start))]


