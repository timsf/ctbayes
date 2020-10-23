import numpy as np
from toolz import curry

from ctbayes.ito import static


@curry
def eval_drift(thi, xt):
    return -thi[0] * xt

@curry
def sx_eval_drift(thi, xt):
    return -thi[0] * np.square(xt) / 2

@curry
def dx_eval_drift(thi, xt):
    return -xt ** 0 * thi[0]

@curry
def reduce(thi, vt):
    return vt / thi[1]

@curry
def dv_reduce(thi, vt):
    return vt ** 0 / thi[1]

@curry
def dereduce(thi, xt):
    return xt * thi[1]

@curry
def eval_bounds_disc(thi, le_x, ue_x):
    if le_x < 0 < ue_x:
        ll_phi = eval_disc(thi, 0)
    else:
        ll_phi = eval_disc(thi, min(abs(le_x), abs(ue_x)))
    ul_phi = eval_disc(thi, max(abs(le_x), abs(ue_x)))
    return eval_inf_disc(thi), ll_phi, ul_phi

def eval_inf_disc(thi):
    return -thi[0] / 2

@curry
def eval_grad(thi, fin_t, init_v, fin_v, t, zs):
    vt = (1 - t / fin_t) * init_v + t / fin_t * fin_v
    bet, sig = thi
    dbet_phi = bet * (fin_t * np.square(zs) + np.square(vt / sig) + 2 * np.sqrt(fin_t) * zs * vt / sig) - 1
    dsig_phi = -bet ** 2 * (np.sqrt(fin_t) * zs * vt / sig ** 2 + np.square(vt) / sig ** 3)
    return np.array([dbet_phi, dsig_phi])

@curry
def eval_bounds_grad(thi_nil, thi_prime, fin_t, init_v, fin_v, lb_z, ub_z):
    ub_abs_z = max(abs(lb_z), abs(ub_z))
    ub_abs_v = max(abs(init_v), abs(fin_v))
    ub_bet = max(thi_nil[0], thi_nil[1])
    lb_sig = min(thi_nil[1], thi_prime[1])
    ul_dbet_phi = ub_bet * (fin_t * ub_abs_z ** 2 + (ub_abs_v / lb_sig) ** 2 + 2 * np.sqrt(fin_t) * ub_abs_z * ub_abs_v / lb_sig) + 1 / 2
    ul_dsig_phi = ub_bet ** 2 * (np.sqrt(fin_t) * ub_abs_z * ub_abs_v / lb_sig ** 2 + ub_abs_v ** 2 / lb_sig ** 3)
    return np.array([ul_dbet_phi, ul_dsig_phi])

def eval_log_prior(thi, bet0=(1, 1), sig0=(1, 1)):
    if np.any(thi < 0):
        return -np.inf
    log_p_bet = (bet0[0] - 1) * np.log(thi[0]) - bet0[1] * thi[0]
    log_p_sig = -(2 * sig0[0] + 1) * np.log(thi[1]) - sig0[1] / np.square(thi[1])
    return log_p_bet + log_p_sig


init_thi = np.array([1, 1])
bounds_thi = (np.array([0.0, 0.0]), np.array([np.inf, np.inf]))
bounds_x = (-np.inf, np.inf)

eval_disc, eval_biased_log_lik, sample_aug, gen_normops, \
    sample_sde, maximize_posterior, sample_posterior = static.assemble_model(
    init_thi, bounds_x, bounds_thi, reduce, dereduce, dv_reduce, eval_drift, sx_eval_drift, dx_eval_drift,
    eval_bounds_disc, eval_bounds_grad, eval_log_prior)
