import numpy as np
from toolz import curry

from ctbayes.ito import static


@curry
def eval_drift(thi, xt):
    return (2 * thi[0] * thi[1] / thi[2] ** 2 - 1 / 2) / xt - thi[0] / 2 * xt

@curry
def sx_eval_drift(thi, xt):
    return (2 * thi[0] * thi[1] / thi[2] ** 2 - 1 / 2) * np.log(xt) - thi[0] / 4 * np.square(xt)

@curry
def dx_eval_drift(thi, xt):
    return (1 / 2 - 2 * thi[0] * thi[1] / thi[2] ** 2) / np.square(xt) - thi[0] / 2

@curry
def reduce(thi, vt):
    return 2 * np.sqrt(vt) / thi[2]

@curry
def dv_reduce(thi, vt):
    return 1 / (np.sqrt(vt) * thi[2])

@curry
def dereduce(thi, vt):
    return np.square(vt * thi[2] / 2)

@curry
def eval_bounds_disc(thi, le_x, ue_x):
    edges_phi = eval_disc(thi, le_x), eval_disc(thi, ue_x)
    if 4 * thi[0] * thi[1] < 3 * thi[2] ** 2:
        return np.nan, min(edges_phi), max(edges_phi)
    argmin_phi = f_argmin_disc(thi)
    min_phi = eval_disc(thi, argmin_phi)
    if le_x < bounds_x[0] or bounds_x[1] < ue_x:
        le_phi, ue_phi = min_phi, np.inf
    elif argmin_phi <= le_x:
        le_phi, ue_phi = edges_phi
    elif ue_x <= argmin_phi:
        le_phi, ue_phi = edges_phi[::-1]
    else:
        le_phi, ue_phi = min_phi, max(edges_phi)
    return min_phi, le_phi, ue_phi

def f_argmin_disc(thi):
    return ((4 * thi[1] / thi[2] ** 2 - 2 / thi[0]) ** 2 - 1 / thi[0] ** 2) ** (1 / 4)

def eval_inf_disc(thi):
    return eval_disc(thi)(None, f_argmin_disc(thi))

@curry
def eval_grad(thi, fin_t, init_v, fin_v, t, zs):
    return np.zeros(3)

@curry
def eval_bounds_grad(thi_nil, thi_prime, fin_t, init_v, fin_v, lb_z, ub_z):
    return np.zeros(3)

def eval_log_prior(thi):
    if np.any(thi < 0):
        return -np.inf
    bet, mu, sig = thi[0], thi[1], thi[2]
    return -(bet + mu + 1 / sig ** 2 + 3 * np.log(sig))


init_thi = np.array([1, 1, 1])
bounds_thi = (np.array([0, 0, 0]), np.array([10, 10, 10]))
bounds_x = (0, np.inf)

eval_disc, eval_biased_log_lik, sample_aug, gen_normops, \
    sample_sde, maximize_posterior, sample_posterior = static.assemble_model(
    init_thi, bounds_x, bounds_thi, reduce, dereduce, dv_reduce, eval_drift, sx_eval_drift, dx_eval_drift,
    eval_bounds_disc, eval_bounds_grad, eval_log_prior)
