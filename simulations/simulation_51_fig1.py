import numpy as np 
import tensorly as tl
import sys
sys.path.append("../mfda/")
from tensor_decomposition import MARGARITA, fCP_TPA
from marginal_product_basis import MPB
from utility import discrete_laplacian_1D
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.representation.basis import Fourier, BSpline, Tensor
from skfda.misc.operators import LinearDifferentialOperator, gram_matrix
from skfda.preprocessing.smoothing import BasisSmoother
from skfda import FDataGrid

from collections import namedtuple 
import time 
import os 
import itertools 
import pickle 
import scipy 
from scipy.stats import ortho_group, sem
import pandas as pd 
import functools 
import operator 

import argparse

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

np.random.seed(0)
svdtuple = namedtuple("SVD", ["U", "s", "Vt"]) 

####UTILITY I/O####
def create_dir(newpath):
	try:
		os.mkdir(newpath)
		print("Directory " , newpath ,  " Created ") 
	except FileExistsError:
		print("Directory " , newpath ,  " already exists")

def pkl2npy(dirpath, Nreps):
	for nrep in range(Nreps):
		fname = "%s.pkl"%nrep
		with open(os.path.join(dirpath, fname), "rb") as pklfile:
			data = pickle.load(pklfile)
		Y_true = data.get("Y")
		Coefs = data.get("Coefs")
		xgrids = data.get("xgrids")
		Epsilon = data.get("Epsilon")
		Clst = data["distr"]["Clst"]
		C_tensor = np.zeros((3, Clst[0].shape[0], Clst[0].shape[1]))
		for d in range(3):
			C_tensor[d,:,:] = Clst[d]
		np.save(os.path.join(dirpath, "Y_" + fname.split(".")[0] + ".npy"), Y_true)
		np.save(os.path.join(dirpath, "Coefs_" + fname.split(".")[0] + ".npy"), Coefs)
		np.save(os.path.join(dirpath, "Epsilon_" + fname.split(".")[0] + ".npy"), Epsilon)
		np.save(os.path.join(dirpath, "Ctensor_" + fname.split(".")[0] + ".npy"), C_tensor)

def createIPMatrices(fourier_basis, bspline_basis):
	J_eta = bspline_basis.gram_matrix()
	J_phi = fourier_basis.gram_matrix()
	J_phieta = fourier_basis.inner_product_matrix(bspline_basis)
	np.save(os.path.join(DATA_DIR, "inner_product_mats", "J_fb_phi_%s.npy"%fourier_basis.n_basis), J_phi)
	np.save(os.path.join(DATA_DIR, "inner_product_mats", "J_bs_eta_%s.npy"%bspline_basis.n_basis), J_eta)
	np.save(os.path.join(DATA_DIR, "inner_product_mats", "J_fbbs_phieta_%s_%s.npy"%(fourier_basis.n_basis, bspline_basis.n_basis)), 
								J_phieta)

####UTILITY METRICS####
def MISE_evaluation(Clst_true, Coef_true, basis_true, K_true, Jlst_true, Jlst_cross,
					Clst_repr, basis_repr, Smat_repr, K_repr, Jlst_repr): 
	N = Smat_repr.shape[0]
	MISE = np.zeros(N)
	for i in range(N):
		f2 = 0; g2 = 0; fg = 0
		for k in range(K_true):
			for j in range(K_true):
				f2 += Coef_true[i,k]*Coef_true[i,j]*np.prod([Clst_true[d][:,k:(k+1)].T@Jlst_true[d]@Clst_true[d][:,j:(j+1)]  
												   for d in range(3)])
		for k in range(K_repr):
			for j in range(K_repr):
				g2 += Smat_repr[i,k]*Smat_repr[i,j]*np.prod([Clst_repr[d][:,k:(k+1)].T@Jlst_repr[d]@Clst_repr[d][:,j:(j+1)]  
												   for d in range(3)]) 
		for k in range(K_true):
			for j in range(K_repr):
				fg += Coef_true[i,k]*Smat_repr[i,j]*np.prod([Clst_true[d][:,k:(k+1)].T@Jlst_cross[d]@Clst_repr[d][:,j:(j+1)]  
												   for d in range(3)])
		MISE[i] = f2 - 2*fg + g2
	return MISE

####UTILITY SIMULATOR####
##RANDOM TENSOR DISTRIBUTION
def random_tensor_distribution(n_marginal_basis=20, K=50, nmode=3, alpha=1e-1):
	## This should only need to be run twice, once for each sbst considered
	_id = "md_%s__nmode_%s__K_%s__alpha_%s.pkl"%(n_marginal_basis, nmode, K, alpha)
	fourier_basis = Fourier((0,1), n_basis=n_marginal_basis, period=1)
	V = ortho_group.rvs(K)
	Sigma_S = V @ np.diag(np.exp(-alpha*np.arange(0,K))) @ V.T
	mu_S = np.zeros(K)
	Clst = [np.random.normal(loc=0.0, scale=0.3, size=(n_marginal_basis, K)) for d in range(nmode)]
	results = {"Clst":Clst, "Sigma_S":Sigma_S, "mu_S":mu_S, "K":K, "marginal_basis":fourier_basis}
	with open(os.path.join(DATA_DIR, "random_tensor_distributions", _id), "wb") as pklfile:
		pickle.dump(results, pklfile)

##SIMULATOR
def simulate_data(Clst, skfda_basis, Sigma_S, K, Ns, N, sigma2, nmode):
	## Link replications together by common naming convention on the outname 
	n1, n2, n3 = Ns
	xgrids = [np.linspace(0, 1, n1),
			  np.linspace(0, 1, n2),
			  np.linspace(0, 1, n3)]
	Phis = [np.squeeze(skfda_basis.evaluate(xgrids[d])).T for d in range(nmode)]
	## Sample random coefficients 
	Coefs = np.random.multivariate_normal(np.zeros(K), Sigma_S, N) ## N x K (3)
	## Construct tensor basis system 
	tensor_lst = [np.prod(np.ix_(*[Phis[d]@Clst[d][:, k] for d in range(nmode)])) for k in range(K)]
	tensor_basis = np.zeros(tuple([K] + list(tensor_lst[0].shape)))
	for k in range(K):
		tensor_basis[k, :, :, :] = tensor_lst[k]
	## Simulate data 
	Y = np.zeros((n1, n2, n3, N))
	for n in range(N):
		Y[:, :, :, n] = tl.tenalg.mode_dot(tensor_basis, Coefs[n,:], 0)
	Epsilon = np.random.normal(0, np.sqrt(sigma2), size=Y.shape)
	return Y, Coefs, xgrids, Epsilon

def create_replication(N, nd, sigma2, dirname, rt_file):
	n1, n2, n3 = nd, nd, nd
	rt_dir = rt_file.strip(".pkl")
	create_dir(os.path.join(DATA_DIR, "simulated_data"))
	create_dir(os.path.join(DATA_DIR, "simulated_data", rt_dir))
	create_dir(os.path.join(DATA_DIR, "simulated_data", rt_dir, dirname))
	## Read and extract data 
	with open(os.path.join(DATA_DIR, "random_tensor_distributions", rt_file), "rb") as pklfile: #"md_%s__nmode_%s__K_%s.pkl"
		results = pickle.load(pklfile)
	Clst = results.get("Clst")
	Sigma_S = results.get("Sigma_S")
	K = results.get("K")
	skfda_basis = results.get("marginal_basis")
	Y, Coefs, xgrids, Epsilon = simulate_data(Clst, skfda_basis, Sigma_S, K, (n1, n2, n3), N, sigma2, 3)  
	return {"Y":Y, "Coefs":Coefs, "xgrids":xgrids, "Epsilon":Epsilon, "distr":results}

####SIMULATION STUDY FROM MAIN TEXT#### 
def simulation_study_1(mb_dims_model, K, lambdas_fpc_tpa, lambdas_f_marga, nmode, data):
	## set algorithm parameters  
	maxiter = (100, 100)
	tol_inner = (1e-3, 1e-3)
	tol_outer = 1e-3
	initialize = "random"
	bspline_basis = [BSpline(n_basis=mb_dims_model[d], order=4) for d in range(nmode)]
	## Get true data qunatities 
	Y_true = data.get("Y")
	Coefs = data.get("Coefs")
	xgrids = data.get("xgrids")
	Epsilon = data.get("Epsilon")
	Y_noisey = Y_true + Epsilon
	N = Y_noisey.shape[-1]
	true_dist = data.get("distr")
	Clst_true = true_dist.get("Clst")
	K_true = true_dist.get("K")
	marg_basis_true = true_dist.get("marginal_basis")
	basis_true = [marg_basis_true, marg_basis_true, marg_basis_true]
	cross_fname = "bspline_%s_%s_%s__fb_%s_%s_%s.pkl"%(mb_dims_model[0], mb_dims_model[1], mb_dims_model[2],
													   marg_basis_true.n_basis, marg_basis_true.n_basis, marg_basis_true.n_basis)
	cross_ip = os.path.join(DATA_DIR, "inner_product_mats", cross_fname)
	if os.path.exists(cross_ip):
		with open(cross_ip, "rb") as pklfile:
			Jlst_true, Jlst_cross = pickle.load(pklfile)
	else:
		Jlst_true = [basis_true[d].gram_matrix() for d in range(nmode)]
		Jlst_cross = [basis_true[d].inner_product_matrix(bspline_basis[d]) for d in range(nmode)]
		with open(cross_ip, "wb") as pklfile:
			pickle.dump((Jlst_true, Jlst_cross), pklfile)
	## Evaluation 
	## 1) FCP-TPA (decompose-then-smooth)
	PenMat = [discrete_laplacian_1D(Y_noisey.shape[d]) for d in range(nmode)]
	start = time.time()
	factors_1, Smat_1, scalars_1 = fCP_TPA(Y_noisey, PenMat, lambdas_fpc_tpa, K, 
										   max_iter=maxiter[0], tol=tol_outer, init=initialize)
	elapsed_1 = time.time() - start
	mpb_fcp_tpa = MPB.from_evaluations(bspline_basis, factors_1, xgrids=xgrids)
	Clst_1 = mpb_fcp_tpa.coefficients
	Smat_scaled_1 = np.multiply(Smat_1, scalars_1)
	## 2) MARGARITA 
	## Compute basis evaluation matrices and SVDs
	Phis = [np.squeeze(bspline_basis[d].evaluate(xgrids[d])).T for d in range(nmode)]
	Svds = [svdtuple(*np.linalg.svd(Phis[d], full_matrices=False)) for d in range(nmode)]
	## Compute inner product + differential operator-based roughness penalty matrix
	Jlst = [bspline_basis[d].gram_matrix() for d in range(nmode)]
	D2 = LinearDifferentialOperator(2)
	Rlst = [gram_matrix(D2, bspline_basis[d]) for d in range(nmode)] 
	## Perform the n-mode coordinate transformations into the spline coefficient space 
	G = tl.tenalg.multi_mode_dot(Y_noisey, [svdt.U.T for svdt in Svds], list(range(nmode)))
	Vs = [Svds[d].Vt.T for d in range(nmode)]
	Dinvs = [np.diag(1./Svds[d].s) for d in range(nmode)]
	Tlst_bcd = [Dinvs[d]@Vs[d].T@Rlst[d]@Vs[d]@Dinvs[d] for d in range(nmode)]
	start = time.time()
	Ctilde_2, Smat_2, scalars_2, FLAG_C, FLAG_N = MARGARITA(G, Tlst_bcd, lambdas_f_marga, K, 
									 max_iter=maxiter, tol_inner=tol_inner, 
									 tol_outer=tol_outer,  regularization="l2", init=initialize, 
									verbose=False)
	elapsed_2 = time.time() - start
	## Map factors to bpsline coordinate space
	Clst_2 = [Svds[d].Vt.T @ np.diag(1/Svds[d].s) @ Ctilde_2[d] for d in range(nmode)] 
	Smat_scaled_2 = np.multiply(Smat_2, scalars_2)
	## MISE 
	ise_1 = MISE_evaluation(Clst_true, Coefs, basis_true, K_true, Jlst_true, Jlst_cross,
							Clst_1, bspline_basis, Smat_scaled_1, K, Jlst)
	ise_2 = MISE_evaluation(Clst_true, Coefs, basis_true, K_true, Jlst_true, Jlst_cross,
							Clst_2, bspline_basis, Smat_scaled_2, K, Jlst)
	return ise_1, ise_2, elapsed_1, elapsed_2

parser = argparse.ArgumentParser()
parser.add_argument("--DATA_DIR", action="store", required=True,
					type=str, help="Set this to the path where data should be/is written (note: this require's a lot of space for full sim-study)")
parser.add_argument("--Nreps", action="store", type=int, default = 100,
				help="Number of simulation replications to run. Note, each replication writes approximately 2GB of data to DATA_DIR.")

args = parser.parse_args()
DATA_DIR = args.DATA_DIR 
Nreps = args.Nreps

## Sample parameters 
nmode = 3 
Nds = (30, 50) 
sigma2 = 10.
Nsamps = (5, 50)

## Distribution parameters 
n_marginal_basis = 11 ## Fourier basis rank
K_t = 20
alpha = 7e-1
## Model params 
Km = (8, 15, 25) 
lambda_grid = (1e-12, 1e-11, 1e-10, 1e-8, 1e-6, 1e-4)
mb_dims_m = (15, 25)     

create_dir(os.path.join(DATA_DIR, "inner_product_mats"))
## Define function distribution data
create_dir(os.path.join(DATA_DIR, "random_tensor_distributions"))
random_tensor_distribution(n_marginal_basis=n_marginal_basis, K=K_t, nmode=nmode, alpha=alpha)
rt_dir = "md_%s__nmode_3__K_%s__alpha_%s"%(n_marginal_basis, K_t, alpha)
create_dir(os.path.join(DATA_DIR, "analysis_mpb"))
create_dir(os.path.join(DATA_DIR, "analysis_mpb", rt_dir))
## Simulate replicated datasets corresponding to the desired states of nature  
rt_file = "md_%s__nmode_3__K_%s__alpha_%s.pkl" % (n_marginal_basis, K_t, alpha)

totstates = np.prod((len(Nsamps),
					 Nreps,
					len(Nds),
					len(mb_dims_m),
					len(Km),
					len(lambda_grid)))
st_cnt = 0
for N in Nsamps:
	for nd in Nds:
		samp_dir = "sigma2_%s__nd_%s__N_%s" % (sigma2, nd, N)
		create_dir(os.path.join(DATA_DIR, "analysis_mpb", rt_dir, samp_dir))
		for rep in range(Nreps):
			repl_data = create_replication(N, nd, sigma2, samp_dir, rt_file)
			results = {}
			for mb in mb_dims_m:
				for K in Km:
					for lambda_ in lambda_grid:
						lambdas_fpc_tpa = lambda_*np.ones(nmode)
						lambdas_f_marga = lambda_*np.ones(nmode+1)
						mb_dims_model = [mb]*3  
						ise_fpc_tpa, ise_f_marga, t_fpc_tpa, t_f_marga = simulation_study_1(mb_dims_model, K, 
																					lambdas_fpc_tpa, lambdas_f_marga, 
																					nmode, repl_data)
						results[(mb, K, lambda_)] = (ise_fpc_tpa, ise_f_marga, t_fpc_tpa, t_f_marga)
						st_cnt += 1
						print("Completed eval", st_cnt, "out of ", totstates)
			with open(os.path.join(DATA_DIR, "analysis_mpb", rt_dir, samp_dir, "results_%s.pkl"%rep), "wb") as pklfile:
				pickle.dump(results, pklfile)

## make Figure 1  
indices = [(n_marginal_basis, K_t, sigma2, N, nd) for N in Nsamps for nd in Nds]
index = pd.MultiIndex.from_tuples(indices, names=["mb_true", "K_true", "sigma2", "N", "nd"])

dict_results = {}
columns_mpf = list(itertools.chain(*[["FCP_TPA_%s_%s" % (mb, K),"MARGARITA_%s_%s" % (mb, K)]  for mb in mb_dims_m
                                                                    for K in Km]))
df_mpf = pd.DataFrame(0, columns=columns_mpf, index=index)
df_time = pd.DataFrame(0, columns=columns_mpf, index=index)

dist_dir = "md_%s__nmode_3__K_%s__alpha_0.7"%(n_marginal_basis, K_t)
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

                        df_mpf.loc[(n_marginal_basis, K_t, sigma2, N, nd), "FCP_TPA_%s_%s" % (mb, K)] += np.min(mise_fpc_tpa)
                        df_mpf.loc[(n_marginal_basis, K_t, sigma2, N, nd), "MARGARITA_%s_%s" % (mb, K)] += np.min(mise_f_admm)
                        df_time.loc[(n_marginal_basis, K_t, sigma2, N, nd), "FCP_TPA_%s_%s" % (mb, K)] += np.mean(time_fcp_tpa)
                        df_time.loc[(n_marginal_basis, K_t, sigma2, N, nd), "MARGARITA_%s_%s" % (mb, K)]  += np.mean(time_f_admm)

                        if (K_t, sigma2, N, nd, mb, K) not in dict_results:
                            dict_results[(K_t, sigma2, N, nd, mb, K)] = []
                        dict_results[(K_t, sigma2, N, nd, mb, K)].append((np.min(mise_fpc_tpa), np.min(mise_f_admm), np.mean(time_fcp_tpa), np.mean(time_f_admm)))

df_mpf = df_mpf/Nreps
df_time = df_time/Nreps
df_dict = {}
for N in Nsamps:
    for nd in Nds:
        df_dict[(N, nd)] = []
        for mb in mb_dims_m:
            for K in Km:
                datavec = dict_results[(K_t, sigma2, N, nd, mb, K)]
                for ix in range(len(datavec)):
                    df_dict[(N, nd)].append(("FCP-TPA", "%s,%s"%(mb, K), datavec[ix][0], datavec[ix][2]))
                    df_dict[(N, nd)].append(("MARGARITA", "%s,%s"%(mb, K), datavec[ix][1], datavec[ix][3]))

create_dir(os.path.join(DATA_DIR,"results"))

## Plots 
my_pal = {"MARGARITA": "w", "FCP-TPA": "0.5"}
sns.set(font_scale=1.3)
f, ax = plt.subplots(2, 2, figsize=(18,12))
for i, N in enumerate(Nsamps):
    for j, nd in enumerate(Nds):
        df = pd.DataFrame(df_dict[(N, nd)], columns=["Method",r'Ranks $(m_{d},K_{fit})$',"MISE","Time (s)"])
        ax_ij = sns.boxplot(x=r'Ranks $(m_{d},K_{fit})$', y="MISE", hue="Method",
                data=df, palette=my_pal, ax=ax[i,j], showmeans=True,
                meanprops={"markerfacecolor":"k","markeredgecolor":"k"})
        ax_ij.set_title(r'$N=%s$ $n_{d}=%s$'%(N, nd))
        ax_ij.set_yscale('log')

plt.savefig(os.path.join(DATA_DIR,"results","margarita_fcptpa_mise_comparison.pdf"))





				   
