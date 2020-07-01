import os
import scipy.io

base = "./"
PPI = os.path.join(base, "PPI.mat")
SHOCK = os.path.join(base, "SHOCK.mat")

PPI_mat = scipy.io.loadmat(PPI)
SHOCK_mat = scipy.io.loadmat(SHOCK)

print(PPI_mat.keys())
print(SHOCK_mat.keys())