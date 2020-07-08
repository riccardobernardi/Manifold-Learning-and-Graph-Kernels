from utils import *

print("##########################################")
print("----------PPI-DSGK-KERNEL")

# Uses the shortest path kernel to generate the kernel matrices
gk = DomSetGraKer()
K_train = gk.fit_transform(G_train_PPI)
K_test = gk.transform(G_test_PPI)

print("----------with precomputed kernel")

# Uses the SVM classifier to perform classification
clf = SVC(kernel="precomputed")
clf.fit(K_train, y_train_PPI)
y_pred = clf.predict(K_test)

# Computes and prints the classification accuracy
acc = accuracy_score(y_test_PPI, y_pred)
print("Accuracy:", str(round(acc*100, 2)) + "%")

results_PPI += [("DSGK-KERNEL-precomputed", "Acc: "+str(str(round(acc*100, 2)))+ "%")]

print("----------with linear kernel")

# Uses the SVM classifier to perform classification
clf = SVC(kernel="linear")
clf.fit(K_train, y_train_PPI)
y_pred = clf.predict(K_test)

# Computes and prints the classification accuracy
acc = accuracy_score(y_test_PPI, y_pred)
print("Accuracy:", str(round(acc*100, 2)) + "%")

results_PPI += [("DSGK-KERNEL-linear", "Acc: "+str(str(round(acc*100, 2)))+ "%")]

print("----------with RBF kernel")

# Uses the SVM classifier to perform classification
clf = SVC(kernel="rbf", gamma="auto")
clf.fit(K_train, y_train_PPI)
y_pred = clf.predict(K_test)

# Computes and prints the classification accuracy
acc = accuracy_score(y_test_PPI, y_pred)
print("Accuracy:", str(round(acc*100, 2)) + "%")

results_PPI += [("DSGK-KERNEL-RBF", "Acc: "+str(str(round(acc*100, 2)))+ "%")]



print("##########################################")
print("----------SHOCK-DSGK-KERNEL")


# Uses the shortest path kernel to generate the kernel matrices
gk = DomSetGraKer()
K_train = gk.fit_transform(G_train_SHOCK)
K_test = gk.transform(G_test_SHOCK)

print("----------with precomputed kernel")

# Uses the SVM classifier to perform classification
clf = SVC(kernel="precomputed")
clf.fit(K_train, y_train_SHOCK)
y_pred = clf.predict(K_test)

# Computes and prints the classification accuracy
acc = accuracy_score(y_test_SHOCK, y_pred)
print("Accuracy:", str(round(acc*100, 2)) + "%")

results_SHOCK += [("DSGK-KERNEL-precomputed", "Acc: "+str(str(round(acc*100, 2)))+ "%")]

print("----------with linear kernel")

# Uses the SVM classifier to perform classification
clf = SVC(kernel="linear")
clf.fit(K_train, y_train_SHOCK)
y_pred = clf.predict(K_test)

# Computes and prints the classification accuracy
acc = accuracy_score(y_test_SHOCK, y_pred)
print("Accuracy:", str(round(acc*100, 2)) + "%")

results_SHOCK += [("DSGK-KERNEL-linear", "Acc: "+str(str(round(acc*100, 2)))+ "%")]

print("----------with RBF kernel")

# Uses the SVM classifier to perform classification
clf = SVC(kernel="rbf", gamma="auto")
clf.fit(K_train, y_train_SHOCK)
y_pred = clf.predict(K_test)

# Computes and prints the classification accuracy
acc = accuracy_score(y_test_SHOCK, y_pred)
print("Accuracy:", str(round(acc*100, 2)) + "%")

results_SHOCK += [("DSGK-KERNEL-RBF", "Acc: "+str(str(round(acc*100, 2)))+ "%")]

