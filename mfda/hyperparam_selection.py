import numpy as np
import tensorly as tl
import tensor_decomposition
from collections import namedtuple

svdtuple = namedtuple("SVD", ["U", "s", "Vt"])

def tfold_splt(nfold, Nsamps):
    indices = np.arange(Nsamps)
    np.random.shuffle(indices)
    ho_indices = np.array_split(indices, nfold)
    tr_indices = []
    for e in ho_indices:
        mask_eval = np.ones(Nsamps, bool)
        mask_eval[e] = False
        tr_indices.append(indices[mask_eval])
    return ho_indices, tr_indices

#### Algorithm 2 `Efficient Multidimensional Functional Data Analysis Using Marginal Product Basis Systems' ####
def tfold_cross_val(Y, K, Phis, Rlst, lambda_c_grid, lambda_s_grid, nfold=5, reg_type="l2", maxiter=(200, 100), 
					tol_inner=(1e-3, 1e-3), tol_outer=1e-3, initialize = "random", verbose=False):
    nmode = len(Y.shape[:-1]) 
    ## Compute basis evaluation matrices and SVDs
    Svds = [svdtuple(*np.linalg.svd(Phis[d], full_matrices=False)) for d in range(nmode)]
    ## Map to tilde space 
    Vs = [Svds[d].Vt.T for d in range(nmode)]
    Dinvs = [np.diag(1./Svds[d].s) for d in range(nmode)]
    Tlst_bcd = [Dinvs[d]@Vs[d].T@Rlst[d]@Vs[d]@Dinvs[d] for d in range(nmode)]
    ## BCD parameters     
    max_iter_inner, max_iter_outer = maxiter
    param_grid = [(lam_c, lam_s) for lam_c in lambda_c_grid for lam_s in lambda_s_grid]
    ## construct folds 
    ho_indices, tr_indices = tfold_splt(nfold, Y.shape[-1])
    ## save cross-validation results 
    cross_val_results = {}
    for p_setting in range(len(param_grid)):
        lambda_c, lambda_s = param_grid[p_setting]
        penalty_params = tuple([lambda_c]*nmode + [lambda_s])
        SSE_per_fold = np.zeros(nfold)
        for nf in range(nfold):
            Y_tr = Y[...,tr_indices[nf]]
            Y_ho = Y[...,ho_indices[nf]]
            G = tl.tenalg.multi_mode_dot(Y_tr, [svdt.U.T for svdt in Svds], list(range(nmode)))
            Ctilde, Smat, scalars, FLAG_C, FLAG_N = tensor_decomposition.MARGARITA(G, Tlst_bcd, penalty_params, K, 
						                                         max_iter=maxiter, tol_inner=tol_inner, 
						                                         tol_outer=tol_outer,  regularization=reg_type, 
						                                        init=initialize, verbose=False)
            Clst = [Svds[d].Vt.T @ np.diag(1/Svds[d].s) @ Ctilde[d] for d in range(nmode)] 
            Smat_scaled = np.multiply(Smat, scalars)            
            Zeta_tensor = np.zeros(tuple(list(Y.shape[:-1]) + [Smat_scaled.shape[1]]))
            for k in range(Smat_scaled.shape[1]): ## write below to be a general outer product
                Zeta_tensor[...,k] = np.prod(np.ix_(*[Phis[d] @ Clst[d][:,k] for d in range(nmode)]))

            ## compute MSE on validation set 
            Y_ho_D1 = tl.unfold(Y_ho, nmode).T
            Z_D1 = tl.unfold(Zeta_tensor, nmode).T
            Gram_matrix_Zeta = Z_D1.T @ Z_D1
            S_init = np.zeros((Y_ho.shape[-1], Smat_scaled.shape[1])) ##could give this a "warm-start" with a simple linear regression
            U_dual_init = np.zeros(S_init.shape)
            S_ho, _ = tensor_decomposition.update_S(Y_ho_D1, Z_D1, Gram_matrix_Zeta, S_init, U_dual_init, lambda_s, tol_inner, max_iter_inner, reg_type)
            Y_ho_hat = tl.tenalg.mode_dot(Zeta_tensor, S_ho, nmode)
            SSE = tl.norm(Y_ho - Y_ho_hat)**2

            SSE_per_fold[nf] = SSE

        cross_val_results[(lambda_c, lambda_s)] = np.mean(SSE_per_fold) 

        if verbose:
            print("Finished CV for lambda_c=%s, lambda_s=%s. Value: %s"%(lambda_c, lambda_s, cross_val_results[(lambda_c, lambda_s)]))

    return cross_val_results
