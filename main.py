import os
import scipy.io

base = "./"
PPI_file = os.path.join(base, "PPI.mat")
SHOCK_file = os.path.join(base, "SHOCK.mat")

PPI = scipy.io.loadmat(PPI_file)
SHOCK = scipy.io.loadmat(SHOCK_file)

print(PPI.keys())
print(SHOCK.keys())

print(PPI["G"].shape)
print(PPI["labels"].shape)