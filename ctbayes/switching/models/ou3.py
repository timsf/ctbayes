import numpy as np
from toolz import curry

from ctbayes.switching import static


@curry
def eval_vol(thi, yt):
    return thi[yt, 0]

@curry
def eval_drift(thi, yt, xt):
    return -np.square(thi[yt, 0] / thi[yt, 1]) / 2 * xt

@curry
def sx_eval_drift(thi, yt, xt):
    return -np.square(thi[yt, 0] / thi[yt, 1]) / 2 * np.square(xt) / 2

@curry
def dx_eval_drift(thi, yt, xt):
    return -np.square(thi[yt, 0] / thi[yt, 1]) / 2

@curry
def reduce(vt):
    return vt

@curry
def dv_reduce(vt):
    return vt ** 0

@curry
def dereduce(xt):
    return xt

@curry
def eval_bounds_disc(thi, init_y, le_x, ue_x):
    if le_x < 0 < ue_x:
        ll_phi = eval_disc(thi, init_y, 0)
    else:
        ll_phi = eval_disc(thi, init_y, min(abs(le_x), abs(ue_x)))
    ul_phi = eval_disc(thi, init_y, max(abs(le_x), abs(ue_x)))
    return eval_inf_disc(thi, init_y), ll_phi, ul_phi

def eval_inf_disc(thi, init_y):
    return -np.max(np.square(thi[init_y, 0] / thi[init_y, 1]) / 2) / 2

@curry
def eval_bounds_grad(thi_nil, thi_prime, fin_t, init_v, fin_v, lb_z, ub_z):
    ub_abs_z = max(abs(lb_z), abs(ub_z))
    ub_abs_v = max(abs(init_v), abs(fin_v))
    ub_rho = max(thi_nil[0], thi_prime[0])
    lb_sig = min(thi_nil[1], thi_prime[1])
    ul_drho_phi = (2 * fin_t * ub_rho ** 3 * ub_abs_z ** 2 + ub_rho * ub_abs_v ** 2 + 3 * np.sqrt(fin_t) * ub_rho ** 2 * ub_abs_z * ub_abs_v) / (4 * lb_sig ** 4) + ub_rho / (2 * lb_sig ** 2)
    ul_dsig_phi = (fin_t * ub_rho ** 4 * ub_abs_z ** 2 + (ub_rho * ub_abs_v) ** 2 + 2 * np.sqrt(fin_t) * ub_rho ** 3 * ub_abs_z * ub_abs_v) / (2 * lb_sig ** 5) + ub_rho ** 2 / (2 * lb_sig ** 3)
    return np.array([ul_drho_phi, ul_dsig_phi])

def eval_log_prior(thi):
    if np.any(thi < 0):
        return -np.inf
    rho, sig = thi[:, 0], thi[:, 1]
    return -np.sum(np.square(rho / sig) / 2 + 3 * np.log(sig) + np.square(1 / rho) + 2 * np.log(rho))


init_thi = np.array([1.0, 1.0])
bounds_thi = (np.array([0.0, 0.0]), np.array([np.inf, np.inf]))
bounds_x = (-np.inf, np.inf)

eval_disc, eval_biased_log_lik, sample_aug, gen_normops, \
    maximize_posterior, sample_posterior = static.assemble_model(
    init_thi, bounds_x, bounds_thi, reduce, dereduce, dv_reduce, eval_drift, sx_eval_drift, dx_eval_drift, eval_vol,
    eval_bounds_disc, eval_bounds_grad, eval_log_prior)
