
from utils import results_PPI
from utils import results_SHOCK
import pandas as pd
from PPI_SHOCK_KERNELS import launch

for i in ["NOP", "ISO", "LLE", "SE", "MDS", "TSNE", ]:
	results_PPI, results_SHOCK = launch("SPK", i, results_PPI, results_SHOCK)
	results_PPI, results_SHOCK = launch("WLK", i, results_PPI, results_SHOCK)
	results_PPI, results_SHOCK = launch("STK", i, results_PPI, results_SHOCK)
	results_PPI, results_SHOCK = launch("DSGK", i, results_PPI, results_SHOCK)

results_PPI = pd.DataFrame(results_PPI,columns=["method","PPI_score"])
results_SHOCK = pd.DataFrame(results_SHOCK,columns=["method","SHOCK_score"])

results_PPI["SHOCK_score"] = results_SHOCK["SHOCK_score"]

# print(results_PPI)

print(results_PPI.to_markdown())
