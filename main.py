from utils import results_PPI
from utils import results_SHOCK
import pandas as pd
from PPI_SHOCK_KERNELS import launch


def save():
	with open("results.txt", "w") as ff:
		tmp1 = pd.DataFrame(results_PPI, columns=["method", "PPI_score"])
		tmp2 = pd.DataFrame(results_SHOCK, columns=["method", "SHOCK_score"])

		tmp1["SHOCK_score"] = tmp2["SHOCK_score"]

		ss = tmp1.to_markdown()

		ff.write(ss)

def printa():
	tmp1 = pd.DataFrame(results_PPI, columns=["method", "PPI_score"])
	tmp2 = pd.DataFrame(results_SHOCK, columns=["method", "SHOCK_score"])

	tmp1["SHOCK_score"] = tmp2["SHOCK_score"]

	print(tmp1.to_markdown())


# for i in ["PCA", "NOP", "ISO", "LLE"]:
# 	results_PPI, results_SHOCK = launch("SPK", i, results_PPI, results_SHOCK)
# 	save()
# 	results_PPI, results_SHOCK = launch("WLK", i, results_PPI, results_SHOCK)
# 	save()
# 	results_PPI, results_SHOCK = launch("STK", i, results_PPI, results_SHOCK)
# 	save()
# 	results_PPI, results_SHOCK = launch("DSGK", i, results_PPI, results_SHOCK)
# 	save()

# for i in ["PCA", "NOP", "ISO", "LLE"]:
# 	results_PPI, results_SHOCK = launch("WLK", i, results_PPI, results_SHOCK, n_iter= 1, n_neighbors = 10, n_components = 5)
# 	save()

for i in range(2,20,2):
	results_PPI, results_SHOCK = launch("WLK", "ISO", results_PPI, results_SHOCK, n_iter= 1, n_neighbors = i, n_components = 2)
	save()


save()
printa()

