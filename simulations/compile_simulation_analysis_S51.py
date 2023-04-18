import os 
import pickle 
import numpy as np 
from scipy.stats import sem
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 
import itertools 
import argparse

def create_dir(newpath):
    try:
        os.mkdir(newpath)
        print("Directory " , newpath ,  " Created ") 
    except FileExistsError:
        print("Directory " , newpath ,  " already exists")

n_marginal_basis_dims = (11,)
K_t = (10, 20)

Nds = (30, 50) 
sigma2s = (0.5, 10.)
Nsamps = (5, 50)
#Nreps = 100 
Nreps = 2 ## set this very small to save space 

Km = (8, 15, 25) 
lambda_grid = (1e-12, 1e-11, 1e-10, 1e-8, 1e-6, 1e-4)
mb_dims_m = (15, 25)  

dof_map = {(mb, K):(3*mb*K) for mb in mb_dims_m for K in Km}
tpb_mb = (7, 8, 9, 11, 12, 13)

## Create multidex 
indices = [(n_marginal_basis, K_true, sigma2, N, nd) for n_marginal_basis in n_marginal_basis_dims
													 for K_true in K_t 
													 for sigma2 in sigma2s
													 for N in Nsamps
													 for nd in Nds]
index = pd.MultiIndex.from_tuples(indices, names=["mb_true", "K_true", "sigma2", "N", "nd"])

def vs_FCP_TPA():
	dict_results = {}
	dict_results_tpb = {}
	columns_mpf = list(itertools.chain(*[["FCP_TPA_%s_%s" % (mb, K),"MARGARITA_%s_%s" % (mb, K)]  for mb in mb_dims_m
																		for K in Km]))
	df_mpf = pd.DataFrame(0, columns=columns_mpf, index=index)
	df_time = pd.DataFrame(0, columns=columns_mpf, index=index)
	for n_marginal_basis in n_marginal_basis_dims:
		for K_true in K_t:
			dist_dir = "md_%s__nmode_3__K_%s__alpha_0.7"%(n_marginal_basis, K_true)
			for sigma2 in sigma2s:
				for N in Nsamps:
					for nd in Nds:
						samp_dir = "sigma2_%s__nd_%s__N_%s" % (sigma2, nd, N)
						for replindex in range(Nreps):
							if os.path.exists(os.path.join(DATA_DIR, "analysis_mpb", dist_dir, samp_dir, "results_%s.pkl"%replindex)):
								with open(os.path.join(DATA_DIR, "analysis_mpb", dist_dir, samp_dir, "results_%s.pkl"%replindex), "rb") as pklfile:
									data = pickle.load(pklfile)
								for mb in mb_dims_m:
									for K in Km:
										mise_fpc_tpa = []; time_fcp_tpa = []
										mise_f_admm = []; time_f_admm = []
										for lambda_ in lambda_grid:
											ise_fpc_tpa, ise_f_admm, t_fpc_tpa, t_f_admm = data[(mb, K, lambda_)]
											mise_fpc_tpa.append(np.mean(ise_fpc_tpa))
											mise_f_admm.append(np.mean(ise_f_admm))
											time_fcp_tpa.append(t_fpc_tpa)
											time_f_admm.append(t_f_admm)

										df_mpf.loc[(n_marginal_basis, K_true, sigma2, N, nd), "FCP_TPA_%s_%s" % (mb, K)] += np.min(mise_fpc_tpa)
										df_mpf.loc[(n_marginal_basis, K_true, sigma2, N, nd), "MARGARITA_%s_%s" % (mb, K)] += np.min(mise_f_admm)
										df_time.loc[(n_marginal_basis, K_true, sigma2, N, nd), "FCP_TPA_%s_%s" % (mb, K)] += np.mean(time_fcp_tpa)
										df_time.loc[(n_marginal_basis, K_true, sigma2, N, nd), "MARGARITA_%s_%s" % (mb, K)]  += np.mean(time_f_admm)

										if (K_true, sigma2, N, nd, mb, K) not in dict_results:
											dict_results[(K_true, sigma2, N, nd, mb, K)] = []
										dict_results[(K_true, sigma2, N, nd, mb, K)].append((np.min(mise_fpc_tpa), np.min(mise_f_admm), np.mean(time_fcp_tpa), np.mean(time_f_admm)))

	df_mpf = df_mpf/Nreps
	df_time = df_time/Nreps
	df_mpf.to_csv(os.path.join(DATA_DIR,"results","analysis_margarita_fcptpa_mise.csv"))
	df_time.to_csv(os.path.join(DATA_DIR,"results","results/analysis_margarita_fcptpa_time.csv"))
	
	## Creat figures 
	K_true = 20; sigma2 = 10.
	df_dict = {}
	for N in Nsamps:
		for nd in Nds:
			df_dict[(N, nd)] = []
			for mb in mb_dims_m:
				for K in Km:
					datavec = dict_results[(K_true, sigma2, N, nd, mb, K)]
					for ix in range(len(datavec)):
						df_dict[(N, nd)].append(("FPC-TPA", "%s,%s"%(mb, K), datavec[ix][0], datavec[ix][2]))
						df_dict[(N, nd)].append(("MARGARITA", "%s,%s"%(mb, K), datavec[ix][1], datavec[ix][3]))
	## Plots 
	sns.set(font_scale=1.3)
	f, ax = plt.subplots(2, 2, figsize=(18,12))
	for i, N in enumerate(Nsamps):
		for j, nd in enumerate(Nds):
			df = pd.DataFrame(df_dict[(N, nd)], columns=["Method",r'Ranks $(m_{d},K_{fit})$',"MISE","Time (s)"])
			ax_ij = sns.boxplot(x=r'Ranks $(m_{d},K_{fit})$', y="MISE", hue="Method",
                 	data=df, palette="Set3", ax=ax[i,j], showmeans=True)
			ax_ij.set_title(r'$N=%s$ $n_{d}=%s$'%(N, nd))
			ax_ij.set_yscale('log')
	plt.savefig(os.path.join(DATA_DIR,"results","margarita_fcptpa_mise_comparison.png"))

	f, ax = plt.subplots(2, 2, figsize=(18,12))
	for i, N in enumerate(Nsamps):
		for j, nd in enumerate(Nds):
			df = pd.DataFrame(df_dict[(N, nd)], columns=["Method",r'Ranks $(m_{d},K_{fit})$',"MISE","Time (s)"])
			ax_ij = sns.boxplot(x=r'Ranks $(m_{d},K_{fit})$', y="Time (s)", hue="Method",
                 	data=df, palette="Set3", ax=ax[i,j], showmeans=True)
			ax_ij.set_title(r'$N=%s$ $n_{d}=%s$'%(N, nd))
			ax_ij.set_yscale('log')


	plt.savefig(os.path.join(DATA_DIR,"results", "margarita_fcptpa_time_comparison.png"))

def vs_TPB():
	mpf_dict = {}
	tpb_dict = {}
	for n_marginal_basis in n_marginal_basis_dims:
		for K_true in K_t:
			dist_dir = "md_%s__nmode_3__K_%s__alpha_0.7"%(n_marginal_basis, K_true)
			for sigma2 in sigma2s:
				for N in Nsamps:
					for nd in Nds:
						samp_dir = "sigma2_%s__nd_%s__N_%s" % (sigma2, nd, N)
						for replindex in range(Nreps):
							with open(os.path.join(DATA_DIR, "analysis_mpb", dist_dir, samp_dir, "results_%s.pkl"%replindex), "rb") as pklfile:
								data = pickle.load(pklfile)
							for mb in mb_dims_m:
								for K in Km:
									mise_f_admm = []
									for lambda_ in lambda_grid:
										ise_fpc_tpa, ise_f_admm, t_fpc_tpa, t_f_admm = data[(mb, K, lambda_)]
										mise_f_admm.append(np.mean(ise_f_admm))
									
									if (K_true, sigma2, N, nd, dof_map[(mb, K)]) not in mpf_dict:
										mpf_dict[(K_true, sigma2, N, nd, dof_map[(mb, K)])] = []
									mpf_dict[(K_true, sigma2, N, nd, dof_map[(mb, K)])].append(np.min(mise_f_admm))

							tpb_data = np.load(os.path.join(DATA_DIR, "analysis_tpb", dist_dir, samp_dir, "%s.npy"%replindex))
							tpb_mise = np.mean(tpb_data, 1)[:,3]

							for i, tmb in enumerate(tpb_mb):
								if (K_true, sigma2, N, nd, tmb**3) not in tpb_dict:
									tpb_dict[(K_true, sigma2, N, nd, tmb**3)] = []
								tpb_dict[(K_true, sigma2, N, nd, tmb**3)].append(tpb_mise[i])

	K_true = 20; sigma2 = 10.
	df_dict = {}
	for N in Nsamps:
		for nd in Nds:
			df_dict[(N, nd)] = []
			for mpfdf in np.unique(list(dof_map.values())):
				for mpf_mise in mpf_dict[(K_true, sigma2, N, nd, mpfdf)]:
					df_dict[(N, nd)].append(("MARGARITA", mpfdf, mpf_mise))
			for tpbdf in (tmb**3 for tmb in tpb_mb):
				for tpb_mise in tpb_dict[(K_true, sigma2, N, nd, tpbdf)]:
					df_dict[(N, nd)].append(("TPB", tpbdf, tpb_mise))

					
	sns.set(font_scale=1.3)
	f, ax = plt.subplots(2, 2, figsize=(18,12))
	for i, N in enumerate(Nsamps):
		for j, nd in enumerate(Nds):
			df = pd.DataFrame(df_dict[(N, nd)], columns=["Method","Degrees of Freedom","MISE"])
			ax_ij = sns.boxplot(x="Degrees of Freedom", y="MISE", hue="Method",data=df, palette="Set1", ax=ax[i,j], showmeans=True)
			ax_ij.set_title(r'$N=%s$ $n_{d}=%s$'%(N, nd))
			ax_ij.set_yscale('log')
	plt.savefig(os.path.join(DATA_DIR,"results", "margarita_tpb_dof_mise_comparison.png"))

parser = argparse.ArgumentParser()
parser.add_argument("--DATA_DIR", action="store", required=True,
                    type=str, help="Set this to the path where data should be/is written (require's a lot of space for full sim-study)")
args = parser.parse_args()
DATA_DIR = args.DATA_DIR 
create_dir(os.path.join(DATA_DIR,"results"))

vs_FCP_TPA()
vs_TPB()

