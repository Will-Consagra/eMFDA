import datetime 
import numpy as np 
import tensorly as tl

from tensorly.tenalg.proximal import soft_thresholding
from tensorly.decomposition import parafac
from tensorly.decomposition._cp import initialize_cp
from tensorly.tenalg import inner
from scipy import linalg 

from functools import reduce 

#### Algorithm 1 (Allen, 2013) ####
def fCP_TPA(Y, PenMat, lambdas, K, max_iter=100, tol=1e-8, init="random"):
	"""
	Implementation of functional CP-TPA algorithm from Allen, 2013 (Algorithm 1).

	Arguments:
			Y: n_1 x ... x n_p X N raw data tensor 
			PenMat: lenght P list of n_dxn_d matrices defining the smoothing structure to be appleid to factors
			lambdas: length p list of marginal roughness penalties
			K: Rank of decomposition 
			max_iter: maximum number of iteration in the inner ALS algorithm
			tol: tolerance to exit inner loop 
			init: {"svd", "random"} 
	Returns:			
			Clst: List of numpy arrays of rank-1 factors of Y
			Smat: numpy array of subject coefficients 
			scalars: numpy array of scaling factors  
	"""
	P = len(Y.shape[:-1])
	N = Y.shape[-1]
	rangeP = list(range(P))
	nlst = Y.shape[:-1]
	## Initialize factors 
	weights, factors = initialize_cp(Y, K, init=init)
	## Get inverse smoothermat for regularization 
	SmootherMatInv = [(np.eye(nlst[p])+lambdas[p]*PenMat[p]) for p in rangeP]
	SmootherMat = [np.linalg.inv(SmootherMatInv[p]) for p in range(P)]
	## Create ndarrays to hold factors 
	Clst = [np.zeros((nlst[p], K)) for p in rangeP]
	Smat = np.zeros((Y.shape[-1], K))
	scalars = np.zeros(K)
	Yhat = Y
	## Perform decomposition
	for k in range(K):
		i = 0 
		delta_residual_norm = np.inf 
		residual_norm_prev = np.inf 
		C_i_norms = [np.sqrt(factors[p][:, k].reshape((1, nlst[p]))@factors[p][:,k].reshape((nlst[p], 1))).item() for p in rangeP]
		C_i = [factors[p][:, k].reshape((nlst[p], 1))/C_i_norms[p] for p in rangeP]
		S_i = factors[-1][:, k].reshape((N, 1)) 
		while (i <= max_iter) and (delta_residual_norm > tol):
			## Update smooth factors
			for p in range(P):
				if p == 0:
					mode_less_p = rangeP[1:]
				elif p == P:
					mode_less_p = rangeP[0:-1]
				else:
					mode_less_p = rangeP[0:p] + rangeP[(p+1):]
				C_less_p = [C_i[pp] for pp in mode_less_p]
				C_p_new = SmootherMat[p] @ tl.tenalg.multi_mode_dot(Yhat, C_less_p + [S_i], modes=mode_less_p + [P], transpose=True).reshape((nlst[p], 1))
				normalize_p = np.prod([(C_i[pp].T @ SmootherMatInv[pp] @ C_i[pp]).item() for pp in mode_less_p])
				C_i[p] = C_p_new / normalize_p
			## Update subject factor; right now the accuracy of this form is uncertain.
			S_new = tl.tenalg.multi_mode_dot(Yhat, C_i, modes=rangeP, transpose=True).reshape((N, 1))
			S_i = S_new / np.linalg.norm(S_new)
			## Scale the norms of smooth factors 
			C_i_norms = [np.sqrt(C_i[p].T@C_i[p]).item() for p in rangeP]
			C_i = [C_i[p]/C_i_norms[p] for p in rangeP]
			## Make identifiability constant  
			scalar_K = tl.tenalg.multi_mode_dot(Yhat, C_i + [S_i], list(range(P+1)), transpose=True).item()
			## Check for convergence 
			residual = Yhat - scalar_K*reduce(np.multiply.outer, [C_i[p].ravel() for p in rangeP] + [S_i.ravel()])
			residual_norm = np.sqrt(inner(residual, residual))
			delta_residual_norm = np.abs(residual_norm_prev - residual_norm)
			residual_norm_prev = residual_norm
			i += 1
		## Remove rank-1 factor 
		Yhat = residual
		## Save factors 
		for p in rangeP:
			Clst[p][:, k] = C_i[p].ravel()
		Smat[:, k] = S_i.ravel()
		scalars[k] = scalar_K
	return Clst, Smat, scalars

#### Algorithm 1 (Consagra, 2021) ####
def update_C(G_d, W_d, Gram_matrix, T_d, lambda_d):
	"""
	Parameters:
		G_d: nd.array (prod(mlst[-d]*N), m_d) folded data matrix 
		W_d: nd.array (prod(mlst[-d])*N, K) Khatri-Rao product
		Gram_matrix: nd.array (K, K) W_d.T @ W_d
		T_d: (m_d, m_d) contraint matrix for d-marginal dimension system 
		lambda_d: float, regularization strenght 
	returns:
		C_d: (m_d, K) update of factor matrix 
	"""
	WtG = W_d.T @ G_d
	C_d_T = linalg.solve_sylvester(Gram_matrix, lambda_d*T_d, WtG)
	C_d_update = C_d_T.T
	return C_d_update

def update_S(G_D1, W_D1, Gram_matrix, S_current, U_dual, lambda_D1, tol, max_iter, prox_oper_l, rho_min=1e-6):
	"""
	Parameters:
		G_D1: nd.array (prod(mlst), N) folded data matrix 
		W_D1: nd.array (prod(mlst), K) Khatri-Rao product 
		Gram_matrix: nd.array (K,K) W_D1.T @ W_D1
		S_current: nd.array (N, K) current factor matrix
		U_dual: nd.array (N, K) Current value of dual variable 
		lambda_D1: float, regularization strength 
		tol: tuple, stopping criteria, (tol_abs, tol_relative)
		max_iter: int, stopping criteria
		prox_oper_l: function, proximity operator for convex coefficient penalty l()
		rho_min: float, minimum value for rho-normalization parameter to protect against numerical instability.
	Returns: 
		S_hat: nd.array (N, K) update of subject mode factors
	"""
	K = Gram_matrix.shape[0]
	N = S_current.shape[0]
	rho = np.max((np.trace(Gram_matrix)/K, rho_min))
	GI_inv = np.linalg.inv(Gram_matrix + rho*np.eye(K)) 
	WtG = W_D1.T @ G_D1
	r_crit = np.inf; s_crit = np.inf; i = 0; Z_0 = np.zeros(S_current.T.shape)
	FLAG = 0 
	while (not FLAG) and (i <= max_iter):
		## Update Z 
		Z = GI_inv @ (WtG + rho*(S_current + U_dual).T)
		## Update S 
		S_current = prox_oper_l(Z.T - U_dual, lambda_D1, rho)
		## Update U_dual 
		U_dual = U_dual + S_current - Z.T
		## Compute primal and dual residual and evaluate convergence 
		primal_residual = S_current - Z.T
		dual_residual = rho*(Z - Z_0)
		r_crit = np.linalg.norm(primal_residual, ord="fro")
		s_crit = np.linalg.norm(dual_residual, ord="fro")
		tol_primal = np.sqrt(N)*tol[0] + tol[1]*np.max((np.linalg.norm(S_current, ord="fro"), np.linalg.norm(Z, ord="fro")))
		tol_dual = np.sqrt(N)*tol[0] + tol[1]*np.linalg.norm(U_dual, ord="fro")
		Z_0 = Z
		i += 1
		if (r_crit < tol_primal) and (s_crit < tol_dual):
			FLAG = 1
	return S_current, U_dual

def l1_proximity_operator(Zbar, lambda_d, rho):
	"""
	The so-called proximity operator of the function (lambda/rho)*|| ||_1
	Parameters: 
		Zbar: nd.array, (N,K) Z_0.T - U_dual
		lambda_d: float, user specified regularization penalty 
		rho: regularization for proximity operator 
	Returns:
		S_update: (N, K), update based on proximity operator (pseudo projection)
	"""
	S_update = soft_thresholding(Zbar, lambda_d/rho)
	return S_update

def l2_proximity_operator(Zbar, lambda_d, rho):
	"""
	The so-called proximity operator of the function (lambda/rho)*|| ||_F^2
	Parameters: 
		Zbar: nd.array, (N,K) Z_0.T - U_dual
		lambda_d: float, user specified regularization penalty 
		rho: regularization for proximity operator 
	Returns:
		S_update: (N, K), update based on proximity operator (pseudo projection)
	"""
	S_update = (rho/(rho+lambda_d))*Zbar
	return S_update

def MARGARITA(G, Tlst, lambdas, K, max_iter=(100, 100), tol_inner=(1e-3, 1e-3),
			tol_outer=1e-8, regularization="l1", init="random", tol_num=0.2, verbose=False):
	"""
	Block coodinate descent with ADMM subproblem solver for regularized multidimensional optimal basis problem.
		Arguments: 
			G: m_1 x...x m_D xN data tensor in the "tilde space"; i.e. Y X_1 U_1' X_2 U_2'  ... X_D U_D'
			Tlst: length D list of T_d constraint matrices 
			lambdas: length D+1 list of marginal roughness penalties 
			K: Rank of basis 
			max_iter: tuple, maximum number of iterations for inner and outer loops, at 0 and 1 positions respectively
			tol_inner: tuple, tolerance to exit inner loop (eps_abs, eps_relative)
			tol_outer: float, tolerance to exit outer loop 
			regularization: string, choice of regularization of coefficients S. For now, must be 'l1' or 'l2'.
			tol_num: float, tolerance indicating convergence issues resulting from numercial instability
			init: {"svd", "random"}
			verbose: boolean, specify verbosity 
		Returns:
			C_i: List of rank-1 factors matrices: Ctilde_d
			S_i: Normalized subject coefficients w.r.t. marginal product basis functions 
			scale: Scale factor 
			FLAG_C: boolean indicating if convergence tolerance for BCD was met 
			FLAG_N: boolean indicating if algorithm exited due to evidence of numerical instability 
	"""
	## Set parameters and intitialize factors 
	if regularization == "l1":
		prox_oper_l = l1_proximity_operator
	elif regularization == "l2":
		prox_oper_l = l2_proximity_operator
	else:
		raise ValueError("Varaible 'regularization' must be either 'l1' or 'l2'")
	max_iter_inner, max_iter_outer = max_iter
	D = len(G.shape[:-1])
	N = G.shape[-1]
	Ds = list(range(D))
	weights, factors = initialize_cp(G, K, init=init)
	mlst = [Tlst[d].shape[1] for d in Ds]
	U_i = np.zeros((N, K))
	C_i = [factors[d]/np.linalg.norm(factors[d], ord="fro") for d in Ds]
	S_i = factors[-1]/np.linalg.norm(factors[-1], ord="fro")
	## Perform rank K decomposition 
	residual_norm = np.inf 
	FLAG_C = False
	FLAG_N = False
	itr = 0
	delta_factor_norms = np.zeros((max_iter_outer+1, D+1))
	delta_factor_norms[0,:] = [np.inf]*(D+1)
	while (itr <= max_iter_outer-1) and (not FLAG_C) and (not FLAG_N):
		C_i_init = [np.copy(c) for c in C_i]
		S_i_init = np.copy(S_i)
		for d in Ds:
			G_d = tl.unfold(G, d).T
			W_d = tl.kr([C_i[j] for j in Ds if j != d] + [S_i])
			Gram_matrix_d = reduce(np.multiply, [C_i[j].T@C_i[j] for j in Ds if j != d] + [S_i.T@S_i])
			C_d = update_C(G_d, W_d, Gram_matrix_d, Tlst[d], lambdas[d])
			C_i[d] = C_d/np.linalg.norm(C_d, ord="fro")
		G_D1 = tl.unfold(G, D).T
		W_D1 = tl.kr(C_i)
		Gram_matrix_D1 = reduce(np.multiply, [C_i[j].T@C_i[j] for j in Ds])
		S_i, U_i = update_S(G_D1, W_D1, Gram_matrix_D1, S_i, U_i, lambdas[-1], tol_inner, max_iter_inner, prox_oper_l)
		scale = np.linalg.norm(S_i, ord="fro")
		S_i = S_i /scale
		delta_factor_norms[itr+1,:] = [np.linalg.norm(C_i[d] - C_i_init[d], ord="fro") for d in Ds] + [np.linalg.norm(S_i - S_i_init)]
		FLAG_C = np.all(delta_factor_norms[itr+1,:] < tol_outer)
		FLAG_N = np.any(delta_factor_norms[itr+1,:] - delta_factor_norms[itr,:] > tol_num)
		itr += 1
		if verbose:
			print("outer iteration", itr, "delta factor norms:", delta_factor_norms[itr,:])

	return C_i, S_i, scale, FLAG_C, FLAG_N
