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


for i in ["NOP", "ISO", "LLE", "TSNE"]:
	results_PPI, results_SHOCK = launch("SPK", i, results_PPI, results_SHOCK)
	save()
	results_PPI, results_SHOCK = launch("WLK", i, results_PPI, results_SHOCK)
	save()
	results_PPI, results_SHOCK = launch("STK", i, results_PPI, results_SHOCK)
	save()
	results_PPI, results_SHOCK = launch("DSGK", i, results_PPI, results_SHOCK)
	save()



save()
printa()

