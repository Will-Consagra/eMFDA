
import numpy as np
import tensorly as tl 
from sklearn.decomposition import PCA
from scipy import linalg 

def discrete_laplacian_1D(n, dx=1):
	"""
	Assumes uniform spacing dx,
	"""
	return (1./(dx**2))*np.diag(np.ones(n)*-2) + np.diag(np.ones(n-1), k=-1) + np.diag(np.ones(n-1), k=1)

def trapeziodal_rule_1D(n, dx=1):
	"""
	Assumes uniform spacing dx.
	"""
	A = 2*np.eye(n)
	A[0,0] = 1; A[-1,-1] = 1
	return 0.5*dx*A

def FPCA(S, J_psi, R_psi, lam=1e-8):
	"""
	Implementation of roughness penalized FPCA from Silverman 1996, adapted to marginal product basis functions 
	Arguments:
		S: (ndarray) N x K matrix of subject coefficients w.r.t. the marginal product basis functions
		J_psi: (ndarray) KxK L2 inner product matrix of marginal product basis functions
		R_psi: (ndarray) KxK L2 inner product matrix of linear differential operator penalty functional applied to marginal product basis functions 
		lam: (float) regularization strength
	Returns:
		B: KxK matrix of coefficient expansion of the eigenfunctions 
		gamma: (K,) array of associated eigenvalues 
	Notes:
	"""
	S_centered = S - np.mean(S, 0)
	K = J_psi.shape[0]
	M_lambda = J_psi + lam*R_psi
	L = np.linalg.cholesky(M_lambda)
	SL = np.linalg.inv(L.T)
	D = SL.T @ J_psi @ S_centered.T
	D_pca = PCA()
	D_pca.fit(D.T)
	B = SL @ D_pca.components_.T
	for k in range(B.shape[1]):
		B[:, k] = B[:, k] / np.sqrt(B[:, k].reshape((1,K))@B[:, k].reshape((K,1))).item()
	gamma = D_pca.singular_values_
	return B, gamma


#### Random Projections ####
def OBRP(Y, mb_dims_model, q):
	"""
	Construct optimal marginal basis systems via Random Projections. 
	Uses power iterations to control tail. See Erichson, 2020.
		Arguments:
			Y: np.ndarry n1 x ... x nD x N data tensor 
			mb_dims_model: lenth D list of integers m_d specifying the ranks of the marginal basis systems
			q: int, number of power iterations 
		Returns:
			Us: length D list of n_d x m_d  optimal basis systems 
	"""
	Us = []
	D = len(Y.shape) - 1
	y_unfoldings = [tl.unfold(tl.tensor(Y), d) for d in range(D)]
	## For now, just use random normal vectors. Other distributions may have better performance, see Halko 2011 section 4.5. 
	Ws = [np.random.normal(0,1,size=(y_unfoldings[d].shape[1], mb_dims_model[d])) for d in range(D)]
	for d in range(D):
		a_d = y_unfoldings[d]
		Ws_d = Ws[d]
		y_d = a_d @ Ws_d
		if q > 0:
			for i in np.arange(1, q+1):
				y_d, ll1 = linalg.lu(y_d, permute_l=True, check_finite=False, overwrite_a=True)
				z_d , ll2 = linalg.lu(a_d.T @ y_d, permute_l=True, check_finite=False, overwrite_a=True)
				y_d = a_d @ z_d
		Q_d, ll2 = linalg.qr(y_d,  mode="economic", check_finite=False, overwrite_a=True ) 
		Us.append(Q_d.T)
	return Us

def AOBRP(Y, tols, q, r=10, maxrank=None):
	"""
	Construct optimal marginal basis systems via Random projections with adpative marginal ranks.
		Arguments:
			Y: np.ndarry n1 x ... x nD x N data tensor 
			tols: lenth D list of floats m_d specifying the approximation error bound ||(I-U_dU_d^\prime)Y_d||_{F}^2 < tol[d]
			q: int, number of power iterations 
			r: int, Error bound holds with probability at least 1 - m_d*10^{-r}
			maxrank: int or None, threshold on the maximum marginal ranks 
		Returns:
			Us: length D list of n_d x m_d  optimal basis systems 

	"""
	Us = []
	D = len(Y.shape) - 1
	if maxrank is None:
		maxranks = Y.shape[:-1]
	else:
		maxranks = [maxrank]*D
	y_unfoldings = [tl.unfold(tl.tensor(Y), d) for d in range(D)]
	Ws = [np.random.normal(0,1,size=(y_unfoldings[d].shape[1], y_unfoldings[d].shape[0])) for d in range(D)]
	for d in range(D):
		a_d = y_unfoldings[d]
		Ws_d = Ws[d]
		y_d = a_d @ Ws_d
		## Perform power iterations to increase spectral decay rate 
		if q > 0:
			for i in np.arange(1, q+1):
				y_d, ll1 = linalg.lu(y_d, permute_l=True, check_finite=False, overwrite_a=True)
				z_d , ll2 = linalg.lu(a_d.T @ y_d, permute_l=True, check_finite=False, overwrite_a=True)
				y_d = a_d @ z_d
		## Implementation of adaptive randomized range finder, algorithm 4.2 of Halko, 2011
		n_d, N_prod_nd = a_d.shape
		y_ds = y_d[:, 0:r]
		norm_yds = np.array([np.linalg.norm(y_ds[:,i]) for i in range(r)])
		I_d = np.eye(n_d)
		j = 0; Q_d = np.zeros((n_d, n_d)) ## max # of basis vectors is n_d
		tol_d = tols[d]*np.linalg.norm(a_d, "fro")/(10*np.sqrt(2/np.pi));
		while np.max(norm_yds[j+1:j+r]) > tol_d and (r+j < n_d) and (j < maxranks[d]):
			if j > 0:
				y_ds[:,j] = (I_d - Q_d[:,0:j]@Q_d[:,0:j].T) @ y_ds[:,j]
			Q_d[:, j] = y_ds[:,j]/np.linalg.norm(y_ds[:,j])
			y_new = y_d[:,r+j]
			y_new = (I_d - Q_d[:,0:j+1]@Q_d[:,0:j+1].T)@y_new
			y_ds = np.append(y_ds, y_new.reshape(-1,1), 1)
			norm_yds = np.append(norm_yds, np.linalg.norm(y_new))
			for i in range(j+1, j+r):
				y_ds[:,i] = y_ds[:,i] - np.dot(Q_d[:,j], y_ds[:,i])
			j += 1
			print("Max norm",  np.max(norm_yds[j+1:j+r]),"tol_d", tol_d, "j=",j)
		Q_d = Q_d[:,0:j]
		Us.append(Q_d.T)
	return Us
