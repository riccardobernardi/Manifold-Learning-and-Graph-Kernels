from utils import *

print("##########################################")
print("----------PPI-SP-KERNEL")

# Uses the shortest path kernel to generate the kernel matrices
gk = ShortestPath(normalize=True)
K_train = gk.fit_transform(G_train_PPI)

embedding = LocallyLinearEmbedding(n_components=3)
X_transformed = embedding.fit_transform(K_train)

#print(X_transformed.shape)

#plt.scatter(X_transformed[:,0],X_transformed[:,1], c= y_train_PPI)
#plt.show()

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X_transformed[:,0],X_transformed[:,1],X_transformed[:,2], c=y_train_PPI)
plt.show()

# K_test = gk.transform(G_test_PPI)
#
# print("----------with precomputed kernel")
#
# # Uses the SVM classifier to perform classification
# clf = SVC(kernel="precomputed")
# clf.fit(K_train, y_train_PPI)
# y_pred = clf.predict(K_test)
#
# # Computes and prints the classification accuracy
# acc = accuracy_score(y_test_PPI, y_pred)
# print("Accuracy:", str(round(acc*100, 2)) + "%")
#
# print("----------with linear kernel")
#
# # Uses the SVM classifier to perform classification
# clf = SVC(kernel="linear")
# clf.fit(K_train, y_train_PPI)
# y_pred = clf.predict(K_test)
#
# # Computes and prints the classification accuracy
# acc = accuracy_score(y_test_PPI, y_pred)
# print("Accuracy:", str(round(acc*100, 2)) + "%")
#
# print("----------with RBF kernel")
#
# # Uses the SVM classifier to perform classification
# clf = SVC(kernel="rbf", gamma="auto")
# clf.fit(K_train, y_train_PPI)
# y_pred = clf.predict(K_test)
#
# # Computes and prints the classification accuracy
# acc = accuracy_score(y_test_PPI, y_pred)
# print("Accuracy:", str(round(acc*100, 2)) + "%")
#
