import ot
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "../tools/partitioned_networks/src"))
import coot
from coot import dot, H, eta
import numpy as np

def _solve_fgw_balanced(M, C1, C2, w1, w2, eps = None, alpha = 1/2, solver = "emd", pi = None, **kwargs):
    # solve Fused Gromov-Wasserstein problem
    if solver == "sinkhorn":
        return ot.gromov.entropic_fused_gromov_wasserstein(M, C1, C2, w1, w2, alpha = alpha, epsilon = eps, solver = "PGD")
    elif solver == "emd":
        return ot.gromov.fused_gromov_wasserstein(M, C1, C2, w1, w2, alpha = alpha, epsilon = eps)
    elif solver == "ipot":
        return ot.gromov.entropic_fused_gromov_wasserstein(M, C1, C2, w1, w2, alpha = alpha, epsilon = eps, solver = "PPA")
    return None

def TPOT(X1, X2,
         w1, w2, v1, v2, 
         C1, C2, C_pd, 
         alpha, beta, 
         eps_s = 0.05, eps_f = 0.05, 
         solver = "ipot", 
         u_s0 = None, v_s0 = None, u_f0 = None, v_f0 = None, 
         pi_s0 = None, pi_f0 = None, 
         iter = 100, print_iter = 25, rel_tol = 1e-9, abs_tol = 1e-9, verbose = False, **solver_args):
    def _obj(pi_s, pi_f):
        # just return <L, pi_s * pi_f> for now, no regularization terms 
        obj_coot = dot(eta(_X1, _X2, pi_f.sum(-1), pi_f.sum(0)) - _X1 @ pi_f @ _X2.T, pi_s) 
        obj_ot = dot(C_pd, pi_f)
        obj_gw = dot(eta(C1, C2, pi_s.sum(-1), pi_s.sum(0)) - C1 @ pi_s @ C2.T, pi_s)
        return beta*obj_coot + (alpha * obj_gw + (1-alpha)*obj_ot), {"coot" : beta*obj_coot, "ot" : (1-alpha)*obj_ot, "gw" : alpha*obj_gw}
    _X1 = np.hstack([X1, np.zeros((X1.shape[0], 1), )])
    _X2 = np.hstack([X2, np.zeros((X2.shape[0], 1), )])
    # initialize couplings 
    pi_s = np.outer(w1, w2) if pi_s0 is None else pi_s0
    pi_f = np.outer(v1, v2) if pi_f0 is None else pi_f0
    Cs = np.empty_like(pi_s)
    Cf = np.empty_like(pi_f)
    obj = 1e6
    is_converged = False
    # if using sinkhorn or IPOT, retain dual potentials. these are not used for emd
    def init(p, u):
        # helper function for initializing dual potentials
        if u is None:
            return np.ones_like(p)
        else:
            return u
    u_s, v_s = init(w1, u_s0), init(w2, v_s0) if solver != "emd" else (None, None)
    u_f, v_f = init(v1, u_f0), init(v2, v_f0) if solver != "emd" else (None, None)
    for it in range(iter):
        if solver == "exact":
            # block update on samples 
            Cs_coot = eta(_X1, _X2, pi_f.sum(-1), pi_f.sum(0)) - _X1 @ pi_f @ _X2.T
            pi_s = _solve_fgw_balanced(beta*Cs_coot, C1, C2, w1, w2, eps_s, alpha = alpha / (1+alpha))
            # block update on features
            Cf_coot = eta(_X1.T, _X2.T, pi_s.sum(-1), pi_s.sum(0)) - (_X1.T) @ pi_s @ _X2
            Cf_ot = C_pd
            Cf = (1-alpha)*Cf_ot + beta*Cf_coot
            pi_f, _, _ = coot._solve_balanced(v1, v2, Cf, u_f, v_f, eps_f, solver = "emd")
        elif (solver == "sinkhorn") or (solver == "ipot"):
            Ms = alpha*(eta(C1, C2, w1, w2) - C1 @ pi_s @ C2.T) + beta*(eta(_X1, _X2, v1, v2) - _X1 @ pi_f @ _X2.T)
            Mf = (1-alpha)*C_pd + beta*(eta(_X1.T, _X2.T, w1, w2) - _X1.T @ pi_s @ _X2)
            pi_s, u_s, v_s = coot._solve_balanced(w1, w2, Ms, u_s, v_s, eps = eps_s, solver = solver, pi = pi_s, **solver_args)
            pi_f, u_f, v_f = coot._solve_balanced(v1, v2, Mf, u_f, v_f, eps = eps_f, solver = solver, pi = pi_f, **solver_args)
        obj_new, obj_new_terms = _obj(pi_s, pi_f)
        if (abs(obj_new - obj)/obj < rel_tol*obj) or (abs(obj_new - obj) < abs_tol):
            is_converged = True
            print(f"is_converged, obj = {obj}, obj_new = {obj_new}")
            break
        obj = obj_new 
        if print_iter is not None and it % print_iter == 0:
            print(f"Iteration {it},\t obj = {obj},\t obj_terms = {obj_new_terms}")
    return pi_s, pi_f, _obj(pi_s, pi_f)
