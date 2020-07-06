
from utils import results_PPI
from utils import results_SHOCK
import pandas as pd


import PPI_SHOCK_SPK
# import PPI_SHOCK_WLK
# import PPI_SHOCK_STK
# import PPI_LLE
import DSGK

results_PPI = pd.DataFrame(results_PPI,columns=["method","PPI_score"])
results_SHOCK = pd.DataFrame(results_SHOCK,columns=["method","SHOCK_score"])

results_PPI["SHOCK_score"] = results_SHOCK["SHOCK_score"]

print(results_PPI)
