import numpy as np 
from typing import Iterable, Tuple
from skfda.representation.basis import Basis 
from skfda.representation.grid import FDataGrid
from skfda.misc.operators import LinearDifferentialOperator, gram_matrix

class MPB(Basis):
	r"""
	Marginal product basis system.

	Attributes: 
	
	"""

	def __init__(self, basis_list: Iterable[Basis], 
						coefficient_list: Iterable[np.ndarray] = None):

		self._basis_list = tuple(basis_list)
		self._dim = len(self._basis_list)
		if not all(b.dim_codomain == 1 for b in self._basis_list):
			raise ValueError("The basis functions must be univariate and scalar valued")

		if coefficient_list is not None:
			self._coefficient_list = tuple(coefficient_list)
			n_basis = self._coefficient_list[0].shape[1]
		else:
			n_basis = 1
			self._coefficient_list = tuple([np.zeros(self._basis_list[d].n_basis, n_basis) for d in range(self._dim)])

		super().__init__(
			domain_range=[b.domain_range[0] for b in basis_list],
			n_basis=n_basis,
			)

	@classmethod
	def from_evaluations(cls,
						basis_list: Iterable[Basis],
        				basis_evals: Iterable[np.ndarray],
        				xgrids: Iterable[np.ndarray] = None
        				):

		ndim = len(basis_list)

		if xgrids is None:
			xgrids = [np.linspace(*basis_list[d].domain_range[0], basis_evals[d].shape[0]) for d in range(ndim)]

		coefficient_list = [FDataGrid(basis_evals[d].T, xgrids[d]).to_basis(basis_list[d]).coefficients.T for d in range(ndim)]

		return cls(basis_list, coefficient_list=coefficient_list)

	@property
	def basis_list(self) -> Tuple[Basis, ...]:
		return self._basis_list

	@property
	def dim_domain(self) -> int:
		return len(self._domain_range)

	@property
	def coefficients(self) -> Tuple[np.ndarray, ...]:
		return self._coefficient_list
	
	def _evaluate(self, eval_points: np.ndarray) -> np.ndarray:

		matrix = np.zeros((self.n_basis, eval_points.shape[0], self.dim_codomain))

		basis_evaluations = np.multiply.reduce(
							[
							np.squeeze(b(eval_points[:, i:i + 1])).T @ self._coefficient_list[i] 
							for i, b in enumerate(self._basis_list)
							]) 

		matrix[:,:,0] = basis_evaluations.T

		return matrix

	def _gram_matrix(self) -> np.ndarray:

		gram_matrices = [b.gram_matrix() for b in self._basis_list]
		Clist = self._coefficient_list
		K = self.n_basis
		GM = np.zeros((K, K))
		for k1 in range(K):
			for k2 in range(k1, K):
				GM[k1, k2] = np.prod([(
									Clist[d][:, k1:k1+1].T@ gram_matrices[d] @ Clist[d][:, k2:k2+1]).item() 
									for d in range(self._dim)
									])
		ix_low_tri = np.tril_indices(K, -1)
		GM[ix_low_tri] = GM.T[ix_low_tri] 
		return GM

	def roughness_matrix(self) -> np.ndarray:
		K = self.n_basis
		nmode = self._dim
		Clist = self._coefficient_list

		D2 = LinearDifferentialOperator(2)
		J = [b.gram_matrix() for b in self._basis_list]
		J_D2 = [gram_matrix(D2, b) for b in self._basis_list]
		J_cross = [b.inner_product_matrix(b.derivative().derivative()) for b in self._basis_list]
		
		R_psi = np.zeros((K, K))
		for i in range(K):
			for j in range(i, K):
				for d in range(nmode):
					for dprime in range(nmode):
						if d == dprime:
							Prod_less_d =  np.prod([
													(Clist[dtilde][:,j:j+1].T @ J[dtilde] @ Clist[dtilde][:,i:i+1]).item()
													for dtilde in range(nmode) if dtilde not in (d,)
													])
							R_psi[i,j] = R_psi[i,j] + Prod_less_d * (Clist[d][:,j:j+1].T @ J_D2[d] @ Clist[d][:,i:i+1]).item()
						else:
							Prod_d_dprime = (Clist[d][:,j:j+1].T @ J_cross[d] @ Clist[d][:,i:i+1]).item() * \
									 (Clist[dprime][:,i:i+1].T @ J_cross[dprime] @ Clist[dprime][:,j:j+1]).item()
							if nmode > 2:
								Prod_less_d_dprime = np.prod([
															(Clist[dtilde][:,j:j+1].T @ J[dtilde] @ Clist[dtilde][:,i:i+1]).item() 
															for dtilde in range(nmode) if dtilde not in (d, dprime)
															])
								R_psi[i,j] = R_psi[i,j] + Prod_less_d_dprime * Prod_d_dprime
							else:
								R_psi[i,j] =  R_psi[i,j] + Prod_d_dprime
		ix_low_tri = np.tril_indices(K, -1)
		R_psi[ix_low_tri] = R_psi.T[ix_low_tri] 
		return R_psi
