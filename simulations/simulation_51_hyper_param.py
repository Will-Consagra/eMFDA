import numpy as np 
import tensorly as tl
import sys
sys.path.append("../mfda/")
from tensor_decomposition import MARGARITA, fCP_TPA_GCV
from marginal_product_basis import MPB
from hyperparam_selection import tfold_cross_val
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

def create_replications(Nreps, N, nd, sigma2, dirname, rt_file):
	n1, n2, n3 = nd, nd, nd
	rt_dir = rt_file.strip(".pkl")
	create_dir(os.path.join(DATA_DIR, "simulated_data", rt_dir))
	create_dir(os.path.join(DATA_DIR, "simulated_data", rt_dir, dirname))
	## Read and extract data 
	with open(os.path.join(DATA_DIR, "random_tensor_distributions", rt_file), "rb") as pklfile: #"md_%s__nmode_%s__K_%s.pkl"
		results = pickle.load(pklfile)
	Clst = results.get("Clst")
	Sigma_S = results.get("Sigma_S")
	K = results.get("K")
	skfda_basis = results.get("marginal_basis")
	for nrep in range(Nreps):
		Y, Coefs, xgrids, Epsilon = simulate_data(Clst, skfda_basis, Sigma_S, K, (n1, n2, n3), N, sigma2, 3)
		with open(os.path.join(DATA_DIR, "simulated_data", rt_dir, dirname, "%s.pkl"%nrep), "wb") as pklfile:
			pickle.dump({"Y":Y, "Coefs":Coefs, "xgrids":xgrids, "Epsilon":Epsilon, "distr":results}, pklfile)

####SIMULATION STUDIES#### 
def simulation_study_1(mb_dims_model, K, lambda_grid, nmode, replindex, rt_dir, samp_dir, Nreps):
	## set algorithm parameters  
	maxiter = (200, 100)
	tol_inner = (1e-3, 1e-3)
	tol_outer = 1e-3
	initialize = "random"
	bspline_basis = [BSpline(n_basis=mb_dims_model[d], order=4) for d in range(nmode)]
	## Run replicated analysis 
	repfname = "%s.pkl" % replindex
	with open(os.path.join(DATA_DIR, "simulated_data", rt_dir, samp_dir, repfname), "rb") as pklfile:
		data = pickle.load(pklfile) 
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
	## Compute inner product + differential operator-based roughness penalty matrix
	Jlst = [bspline_basis[d].gram_matrix() for d in range(nmode)]
	D2 = LinearDifferentialOperator(2)
	Rlst = [gram_matrix(D2, bspline_basis[d]) for d in range(nmode)]
	## Evaluation 
	## 1) FCP-TPA (decompose-then-smooth)
	PenMat = [discrete_laplacian_1D(Y_noisey.shape[d]) for d in range(nmode)]
	times_fpc_tpa = []
	mise_fcp_tpa = []
	lambdas_fpc_tpa = [lambda_grid for d in range(nmode)]
	start = time.time()
	factors_1, Smat_1, scalars_1 = fCP_TPA_GCV(Y_noisey, PenMat, lambdas_fpc_tpa, K, 
										   max_iter=maxiter[0], tol=tol_outer, init=initialize)
	elapsed_1 = time.time() - start
	## MISE 
	mpb_fcp_tpa = MPB.from_evaluations(bspline_basis, factors_1, xgrids=xgrids)
	Clst_1 = mpb_fcp_tpa.coefficients
	Smat_scaled_1 = np.multiply(Smat_1, scalars_1)
	ise_1 = MISE_evaluation(Clst_true, Coefs, basis_true, K_true, Jlst_true, Jlst_cross,
							Clst_1, bspline_basis, Smat_scaled_1, K, Jlst)
	## 2) MARGARITA 
	## Compute basis evaluation matrices and SVDs
	Phis = [np.squeeze(bspline_basis[d].evaluate(xgrids[d])).T for d in range(nmode)]
	Svds = [svdtuple(*np.linalg.svd(Phis[d], full_matrices=False)) for d in range(nmode)]
	## Cross-validation to select smoothing parameters 
	start = time.time()
	cross_val_results = tfold_cross_val(Y_noisey, K, Phis, Rlst, lambda_grid, lambda_grid, 
							nfold=5, reg_type="l2", verbose=False)
	cross_val_results_tuple = [(k,v) for k, v in cross_val_results.items()]
	cross_val_selection = sorted(cross_val_results_tuple, key=lambda e: e[1], reverse=False)[0]
	lam_c, lam_s = cross_val_selection[0]        
	## Perform the n-mode coordinate transformations into the spline coefficient space 
	G = tl.tenalg.multi_mode_dot(Y_noisey, [svdt.U.T for svdt in Svds], list(range(nmode)))
	Vs = [Svds[d].Vt.T for d in range(nmode)]
	Dinvs = [np.diag(1./Svds[d].s) for d in range(nmode)]
	Tlst_bcd = [Dinvs[d]@Vs[d].T@Rlst[d]@Vs[d]@Dinvs[d] for d in range(nmode)]
	print("Selected params:", lam_c, lam_s)
	lambdas_bcd = [lam_c]*nmode + [lam_s]
	Ctilde_2, Smat_2, scalars_2, FLAG_C, FLAG_N = MARGARITA(G, Tlst_bcd, lambdas_bcd, K, 
										 max_iter=maxiter, tol_inner=tol_inner, 
										 tol_outer=tol_outer,  regularization="l2", init=initialize, 
										verbose=False)
	#Gerr_c = np.inf
	#for rix in range(5):
	#	np.random.seed(rix)
	#	Ctilde, Smat, scalars, FLAG_C, FLAG_N = MARGARITA(G, Tlst_bcd, lambdas_bcd, K, 
	#									 max_iter=maxiter, tol_inner=tol_inner, 
	#									 tol_outer=tol_outer,  regularization="l2", init=initialize, 
	#									verbose=False)
	#	Smat_scaled = np.multiply(Smat, scalars)
	#	Ghat = np.zeros(tuple(list(mb_dims_model) + [N]))
	#	for k in range(K):
	#		Ghat = Ghat + np.prod(np.ix_(*[Ctilde[d][:, k] for d in range(nmode)]+[Smat_scaled[:,k]]))
	#	Gerr = tl.norm(Ghat-G)
	#	if Gerr < Gerr_c:
	#		Ctilde_2 = Ctilde
	#		Smat_2 = Smat
	#		scalars_2 = scalars
	#		Gerr_c = Gerr
	#elapsed_2 = time.time() - start
	elapsed_2 = time.time() - start
	## Map factors to bpsline coordinate space
	Clst_2 = [Svds[d].Vt.T @ np.diag(1/Svds[d].s) @ Ctilde_2[d] for d in range(nmode)] 
	Smat_scaled_2 = np.multiply(Smat_2, scalars_2)
	ise_2 = MISE_evaluation(Clst_true, Coefs, basis_true, K_true, Jlst_true, Jlst_cross,
							Clst_2, bspline_basis, Smat_scaled_2, K, Jlst)
	return ise_1, ise_2, elapsed_1, elapsed_2

####MAIN#### 

parser = argparse.ArgumentParser()
parser.add_argument("--MODE", choices=["simulate","analyze"],
					type=str, help="Simulate or analyze the data.")
parser.add_argument("--DATA_DIR", action="store", required=True,
					type=str, help="Set this to the path where data should be/is written (require's a lot of space for full sim-study)")
parser.add_argument("--Nreps", action="store", type=int, default = 2,
				help="Number of simulation replications to run. Note, each replication writes approximately 2GB of data to DATA_DIR.")
parser.add_argument("--Kt", action="store", type=int, default = 20,
				help="True rank.")
parser.add_argument("--sigma2_e", action="store", type=float, default = 10.,
				help="Measurement noise variance.")

args = parser.parse_args()

## set this to the path where data should be written (require's a lot of space for full sim-study)
MODE = args.MODE
DATA_DIR = args.DATA_DIR 
Nreps = args.Nreps
K_t = args.Kt
sigma2_e = args.sigma2_e
#replindex = args.rep 

## Sample parameters 
nmode = 3 
Nd = 50 
#sigma2_e = 10.
#sigma2_e = 0.5
Nsamps = 50

## Distribution parameters 
n_marginal_basis = 11 ## Fourier basis rank
#K_t = 20
alpha = 7e-1
## Model params 
Km = 25  
mb = 15    
mb_dims_model = [mb]*3  
lambda_grid = (1e-12, 1e-11, 1e-10, 1e-8, 1e-6)
## Create necessary directories 
create_dir(DATA_DIR)
create_dir(os.path.join(DATA_DIR,"random_tensor_distributions"))
create_dir(os.path.join(DATA_DIR, "simulated_data"))
create_dir(os.path.join(DATA_DIR, "inner_product_mats"))
create_dir(os.path.join(DATA_DIR, "analysis_mpb"))

if MODE == "simulate":
	## Simulate data
	random_tensor_distribution(n_marginal_basis=n_marginal_basis, K=K_t, nmode=nmode, alpha=alpha)
	## Simulate replicated datasets corresponding to the desired states of nature  
	rt_file = "md_%s__nmode_3__K_%s__alpha_%s.pkl" % (n_marginal_basis, K_t, alpha)
	dirname = "sigma2_%s__nd_%s__N_%s" % (sigma2_e, Nd, Nsamps)
	create_replications(Nreps, Nsamps, Nd, sigma2_e, dirname, rt_file)

	##Prepare data for sandwhich smoother in simulation.R
	rt_dir = "md_%s__nmode_3__K_%s__alpha_%s" % (n_marginal_basis, K_t, alpha)
	samp_dir = "sigma2_%s__nd_%s__N_%s" % (sigma2_e, Nd, Nsamps)
	pkl2npy(os.path.join(DATA_DIR, "simulated_data", rt_dir, samp_dir), Nreps)

	tpb_mb = (7, 8, 9, 11, 12, 13)
	for mb in tpb_mb:
		fourier_basis = Fourier((0,1), n_basis=n_marginal_basis, period=1)
		bspline_basis = BSpline(n_basis=mb, order=4)
		createIPMatrices(fourier_basis, bspline_basis)

## pass in replication to analyze 
elif MODE == "analyze":
	rt_dir = "md_%s__nmode_3__K_%s__alpha_%s"%(n_marginal_basis, K_t, alpha)
	create_dir(os.path.join(DATA_DIR, "analysis_mpb", rt_dir))
	samp_dir = "sigma2_%s__nd_%s__N_%s" % (sigma2_e, Nd, Nsamps)
	create_dir(os.path.join(DATA_DIR, "analysis_mpb", rt_dir, samp_dir))
	results = []
	for replindex in range(Nreps):
		#try:
		ise_fpc_tpa, ise_marg, t_fpc_tpa, t_marg= simulation_study_1(mb_dims_model, Km, lambda_grid, 
																							nmode, replindex, rt_dir, samp_dir, Nreps)
		results.append(("FCP-TPA", np.mean(ise_fpc_tpa), t_fpc_tpa))
		results.append(("MARGARITA",  np.mean(ise_marg), t_marg))
		#except Exception as e:
		#	print(e)

		print("Finished replication: %s"%replindex)
		print(("FCP-TPA", np.mean(ise_fpc_tpa), t_fpc_tpa))
		print(("MARGARITA",  np.mean(ise_marg), t_marg))

	with open(os.path.join(DATA_DIR, "analysis_mpb", rt_dir, samp_dir, "results_hyperparam_select.pkl"), "wb") as pklfile:
		pickle.dump(results, pklfile)


