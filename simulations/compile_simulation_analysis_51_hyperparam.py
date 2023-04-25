import os 
import pickle 
import numpy as np 
from scipy.stats import sem
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 
import itertools 
import argparse

import warnings
warnings.filterwarnings("ignore")

def create_dir(newpath):
    try:
        os.mkdir(newpath)
        print("Directory " , newpath ,  " Created ") 
    except FileExistsError:
        print("Directory " , newpath ,  " already exists")

parser = argparse.ArgumentParser()
parser.add_argument("--DATA_DIR", action="store", required=True,
                    type=str, help="Set this to the path where data should be/is written (require's a lot of space for full sim-study)")
args = parser.parse_args()
DATA_DIR = args.DATA_DIR 
K_t = 20

create_dir(os.path.join(DATA_DIR,"results"))

## simulation parameters 
Nd = 50 
Nsamps = 50

n_marginal_basis = 11 ## Fourier basis rank
#K_t = 20
alpha = 7e-1
## Model params 
Km = 25  
mb = 15    

dfs_mean = []
dfs_se = []
for sigma2_e in (0.5, 10.):

	SNR = "Low" if sigma2_e == 10. else "High"
	rt_dir = "md_%s__nmode_3__K_%s__alpha_%s"%(n_marginal_basis, K_t, alpha)
	samp_dir = "sigma2_%s__nd_%s__N_%s" % (sigma2_e, Nd, Nsamps)

	with open(os.path.join(DATA_DIR, "analysis_mpb", rt_dir, samp_dir, "results_hyperparam_select.pkl"), "rb") as pklfile:
		results = pickle.load(pklfile)

	## get corresponds (in terms of DF) TPB fit in same format
	results_tpb = [("TPB", mise_tpb_rep[0]) for mise_tpb_rep in np.load(os.path.join(DATA_DIR, "analysis_tpb", rt_dir, samp_dir, "results_hyperparam_select.npy"))][:-1]
	results.extend(results_tpb)

	df = pd.DataFrame(results, columns=["Method", "MISE"])

	dfs_mean.append(df.groupby(["Method"]).mean()["MISE"])
	dfs_se.append(df.groupby(["Method"]).agg(sem)["MISE"])

df_mean = pd.merge(dfs_mean[0], dfs_mean[1], left_index=True, right_index=True, suffixes=("_HighSNR", "_LowSNR"))
df_se = pd.merge(dfs_se[0], dfs_se[1], left_index=True, right_index=True, suffixes=("_HighSNR", "_LowSNR"))
print("----- Data For Table 1 -----")
print()
print("-----Monte-Carlo Mean Results -----")
print()
print(df_mean.round(4).transpose().to_latex())
print()
print("-----Accompanying Standard Errors  -----")
print()
print(df_se.round(4).transpose().to_latex())
