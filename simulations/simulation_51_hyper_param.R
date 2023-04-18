
library(reticulate)
library(hero)
np <- import("numpy")

helper <- function(d, Ctensor, J_phieta, k, ilst) {
  C_d = Ctensor[d,,]
  J_phieta_d = J_phieta
  id = ilst[d]
  md = nrow(C_d)
  result = 0
  for (jd in 1:md) {
    result = result + C_d[jd, k]*J_phieta_d[jd, id]
  }
  return(result)
}

ISE <- function(S, Ctensor, J_phi, J_eta, J_phieta, Chat, i) {
  # \int (f - g)^2 
  nmode = 3 
  K = dim(Ctensor)[3]
  N = dim(S)[1]
  m_phi = sapply(1:nmode, function(d) nrow(J_phi))
  m_eta = sapply(1:nmode, function(d) nrow(J_eta))
  # (I) = \int f^2 
  f2 = 0 
  for (k in 1:K) {
    for (j in 1:K) {
      f2 = f2 + S[i,k]*S[i,j]*prod(sapply(1:nmode, function(d) t(as.matrix(Ctensor[d,,k]))%*%J_phi%*%as.matrix(Ctensor[d,,j])))
    }
  }
  ## (II) = \int g^2 
  g2 = 0
  for (i1 in 1:m_eta[1]) {
    for (i2 in 1:m_eta[2]) {
      for (i3 in 1:m_eta[3]) {
        
        for (j1 in 1:m_eta[1]) {
          for (j2 in 1:m_eta[2]) {
            for (j3 in 1:m_eta[3]) {
              g2 = g2 + Chat[i1, i2, i3]*Chat[j1, j2, j3]*J_eta[i1, j1]*J_eta[i2, j2]*J_eta[i3, j3]
            }
          }
        }
        
      }
    }
  }
  ## (III) = \int fg 
  fg = 0 
  for (i1 in 1:m_eta[1]) {
    for (i2 in 1:m_eta[2]) {
      for (i3 in 1:m_eta[3]) {
        
        temp_result = 0 
        for (k in 1:K) {
          temp_result = temp_result + S[i,k]*prod(sapply(1:nmode, helper, Ctensor=Ctensor, J_phieta=J_phieta, k=k, ilst=c(i1,i2,i3))) 
        }
        fg = fg + Chat[i1, i2, i3]*temp_result
        
      }
    }
  }
  integral = f2 -2*fg + g2
  #return(integral)
  return(c(f2, fg, g2, integral))
}

set.seed(0)

args = commandArgs(trailingOnly = TRUE)
DATA_DIR = args[1]
Nreps = as.integer(args[2])
Kt = as.numeric(args[3])

## simulation parameters 
mb = 11
#Kt = 20
#Kt = 10
N = 50
nd = 50
sigma2s = c(0.5,10.0)
mbf = 11 
rt_dir = paste0("md_", mb, "__nmode_3__K_", Kt, "__alpha_0.7")
for (sigma2 in sigma2s) {
      if (sigma2 == 10) {
        samp_dir = paste0("sigma2_", sigma2, ".0__nd_", nd, "__N_", N)
      } else {
        samp_dir = paste0("sigma2_", sigma2, "__nd_", nd, "__N_", N) 
      }
      analysis = array(0, dim=c(Nreps, 2))
      for (replindex in 0:(Nreps-1)) {
        ## Read in all required data 
        dirname = file.path(DATA_DIR, "simulated_data", rt_dir, samp_dir)
        Y_true = np$load(paste0(dirname, "/Y_", replindex, ".npy"))
        Smat = np$load(paste0(dirname, "/Coefs_", replindex, ".npy"))
        Epsilon = np$load(paste0(dirname, "/Epsilon_", replindex, ".npy"))
        Ctensor = np$load(paste0(dirname, "/Ctensor_", replindex, ".npy"))
        J_phi = np$load(file.path(DATA_DIR, "inner_product_mats", paste0("J_fb_phi_", mb,".npy")))
        J_eta = np$load(file.path(DATA_DIR, "inner_product_mats", paste0("J_bs_eta_", mbf,".npy")))
        J_phieta = np$load(file.path(DATA_DIR, "inner_product_mats", paste0("J_fbbs_phieta_", mb, "_", mbf,".npy")))
        ## Set TPB model parameters 
        phi_1 = bspline(nbasis = mbf, norder=4)
        phi_2 = bspline(nbasis = mbf, norder=4)
        phi_3 = bspline(nbasis = mbf, norder=4)
        marginal_grids = list(seq(0, 1, len = nd), 
                              seq(0, 1, len = nd), 
                              seq(0, 1, len = nd))
        ## Run analysis 
        Y = Y_true + Epsilon
        mise_repl =  rep(0, N)
        time_repl = rep(0, N)
        for (n in 1:N) {
          Y_n = Y[,,,n]
          t1 = proc.time()
          obj = prepare(Y_n, marginal_grids, list(phi_1, phi_2, phi_3))
          obj = enhance(obj)
          sandmod = hero(obj)
          Yhat_n = sandmod$fitted
          elapsed = proc.time() - t1
          Chat_n = sandmod$coefficients 
          res_n = ISE(Smat, Ctensor, J_phi, J_eta, J_phieta, Chat_n, n)
          mise_repl[n] = res_n[4]
          time_repl[n] = elapsed
        }
        write(replindex, stdout())
        analysis[replindex+1,] = c(mean(mise_repl), sum(time_repl))
      }
      ## Write analysis 
      dir.create(file.path(DATA_DIR, "analysis_tpb"))
      dir.create(file.path(DATA_DIR, "analysis_tpb", rt_dir))
      dir.create(file.path(DATA_DIR, "analysis_tpb", rt_dir, samp_dir))
      np$save(file.path(DATA_DIR, "analysis_tpb", rt_dir, samp_dir, paste0("results_hyperparam_select.npy")), analysis)
}
    
