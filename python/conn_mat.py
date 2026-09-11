import pandas as pd
import os

cwd = os.getcwd()
# find all represented provinces in the case data
case_file = os.path.join(cwd, "behavioral_ebola", "data", "insp_sitrep__cumulative_confirmed_cases__daily.csv")
DRC_cases = pd.read_csv(case_file, usecols=["nom"])
repped_provinces = list(set(DRC_cases["nom"]))
repped_provinces.sort()
print(len(repped_provinces))

# load inflow and outflow data
inflow_file = os.path.join(cwd, "behavioral_ebola", "data", "flowminder__inflow_202604__static.matrix.csv")
outflow_file = os.path.join(cwd, "behavioral_ebola", "data", "flowminder__outflow_202604__static.matrix.csv")
inflow_df = pd.read_csv(inflow_file, usecols=repped_provinces)