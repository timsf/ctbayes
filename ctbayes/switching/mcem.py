from joblib import Parallel, delayed
from typing import Callable, Iterator, List, NamedTuple, Optional, Tuple

import numpy as np
from numdifftools import Gradient, Hessian
from scipy.optimize import minimize

from ctbayes.ea3lib import seed as sde_seed
from ctbayes.mjplib import inference as mjp_inf, skeleton as mjp_skel
from ctbayes.switching import mcmc, types
from ctbayes.misc.adapt import MyopicMjpSampler


class Model(NamedTuple):
    t: np.ndarray
    vt: np.ndarray
    bounds_thi: Tuple[Optional[np.ndarray], Optional[np.ndarray]]
    hyper_lam: Tuple[np.ndarray, np.ndarray]
    eval_log_prior: Callable[[np.ndarray], float]
    sample_aug: Callable[[np.ndarray, mjp_skel.Skeleton, np.ndarray, np.ndarray, np.random.Generator], Tuple[sde_seed.Partition, types.Anchorage]]
    gen_normops: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], List[Tuple[Callable, Callable]]]
    eval_biased_loglik: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool], Tuple[float, float]]
    eval_disc: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    eval_bounds_disc: Callable[[np.ndarray, int, float, float], Tuple[float, float, float]]


class Controls(NamedTuple):
    n_thin_mcmc: int = 2
    n_disc_supports: int = 2
    learning_rate: float = 1.5
    ftol: float = 1e-2
    bound_buffer: float = 1e-2
    n_cores: int = 1


def maximize_posterior(init_thi: np.ndarray, mod: Model, ctrl: Controls, ome: np.random.Generator
                       ) -> Iterator[Tuple[float, np.ndarray, np.ndarray, List[sde_seed.Partition], List[types.Anchorage]]]:

    lam = mod.hyper_lam[0] / mod.hyper_lam[1]
    lam[np.isnan(lam)] = -np.nansum(lam, 0)
    thi = np.repeat(init_thi[np.newaxis], lam.shape[0], 0)
    y = mjp_skel.sample_forwards(mod.t[-1], None, lam, ome)
    zz, hh = zip(mod.sample_aug(thi, y, mod.t, mod.vt, ome))
    regime_sampler = MyopicMjpSampler(y)
    n_particles = 1
    obj_nil = -np.inf
    while True:
        obj, thi, lam, zz, hh = iterate_em(thi, lam, zz[-1], hh[-1], mod, ctrl, n_particles, regime_sampler, ome)
        n_particles = int((n_particles ** (1 / ctrl.learning_rate) + 1) ** ctrl.learning_rate)
        yield obj, thi, lam, zz, hh
        if np.abs(obj - obj_nil) < ctrl.ftol:
            break
        obj_nil = obj


def iterate_em(thi: np.ndarray, lam: np.ndarray, z: sde_seed.Partition, h: types.Anchorage,
               mod: Model, ctrl: Controls, n_particles: int, regime_sampler: MyopicMjpSampler, ome: np.random.Generator
               ) -> (float, np.ndarray, np.ndarray, List[sde_seed.Partition], List[types.Anchorage]):

    with Parallel(ctrl.n_cores, 'loky') as pool:
        zz, hh = update_particles(thi, lam, z, h, n_particles, mod, ctrl, regime_sampler, ome, pool)
        thi, obj_thi = update_param(thi, zz, hh, mod, ctrl, ome, pool)
        lam, obj_lam = update_generator(hh, mod.hyper_lam)
    return obj_thi + obj_lam, thi, lam, zz, hh


def update_param(thi_nil: np.ndarray, zz: List[sde_seed.Partition], hh: List[types.Anchorage],
                 mod: Model, ctrl: Controls, ome: np.random.Generator, pool: Parallel) -> (np.ndarray, float):

    def f_wrap_obj(thi_: np.ndarray, i: int) -> float:
        return -np.mean(f_obj(np.vstack([thi_nil[:i], thi_, thi_nil[i+1:]]), i))

    opt_bounds = buffer_bounds(mod.bounds_thi[0], mod.bounds_thi[1], len(thi_nil), ctrl.bound_buffer)
    f_obj = update_objective(thi_nil, zz, hh, mod, ctrl, ome, pool)
    
    opt = pool(
        delayed(minimize)(f_wrap_obj, thi_nil[i], (i,), bounds=list(zip(*opt_bounds)), tol=ctrl.ftol)
        for i in range(thi_nil.shape[0]))

    thi_prime = np.array([opt_.x for opt_ in opt])
    obj = -sum([opt_.fun for opt_ in opt])
    return thi_prime, obj


def update_particles(thi: np.ndarray, lam: np.ndarray, init_z: sde_seed.Partition, init_h: types.Anchorage,
                     n_particles: int, mod: Model, ctrl: Controls, regime_sampler: MyopicMjpSampler, 
                     ome: np.random.Generator, pool: Parallel) -> (List[sde_seed.Partition], List[types.Anchorage]):

    z, h, zz, hh = init_z, init_h, [], []
    for i in range(n_particles * ctrl.n_thin_mcmc):
        z, h = mcmc.update_hidden(thi, lam, z, h, mcmc.Model(*mod, None), mcmc.Controls(), regime_sampler, ome, pool)
        if not i % ctrl.n_thin_mcmc:
            zz.append(z), hh.append(h)
    return zz, hh


def update_objective(thi: np.ndarray, zz: List[sde_seed.Partition], hh: List[types.Anchorage],
                     mod: Model, ctrl: Controls, ome: np.random.Generator, pool: Parallel
                     ) -> Callable[[np.ndarray, int], np.ndarray]:

    def f_obj(thi_prime: np.ndarray, restrict: int) -> np.ndarray:
        log_aug_trans = [est_log_aug_lik(thi_prime, z, h, mod, t, ome, restrict)[0]
                         for z, h, t in zip(new_zz, hh, supp_t)]
        return np.array(log_aug_trans) + mod.eval_log_prior(thi_prime)

    supp_t = [[np.sort(ome.uniform(high=dt, size=ctrl.n_disc_supports)) for dt in np.diff(h.t)] for h in hh]
    new_ome = [np.random.default_rng(seed_) for seed_ in ome.bit_generator._seed_seq.spawn(len(supp_t))]
    _, new_zz = zip(*pool(
        delayed(est_log_aug_lik)(thi, z, h, mod, t, ome_) for z, h, t, ome_ in zip(zz, hh, supp_t, new_ome)))

    return f_obj


def update_generator(hh: List[types.Anchorage], hyper_lam: (np.ndarray, np.ndarray)) -> (np.ndarray, float):

    yy = [types.prune_anchorage(h) for h in hh]
    alp, bet = zip(*[mjp_inf.update(y, *hyper_lam) for y in yy])
    lam_prime = np.mean([(a - 1) for a in alp], 0) / np.mean([b for b in bet], 0)
    lam_prime[np.isnan(lam_prime)] = -np.nansum(lam_prime, 1)
    log_post = np.mean([mjp_inf.eval_loglik(y, lam_prime) for y in yy]) + mjp_inf.eval_logprior(lam_prime, *hyper_lam)
    return lam_prime, log_post


def buffer_bounds(lb: np.ndarray, ub: np.ndarray, n_dim: int, buffer: float = 1e-2) -> (np.ndarray, np.ndarray):

    lb_ = np.repeat(-np.inf, n_dim) if lb is None else lb
    ub_ = np.repeat(np.inf, n_dim) if ub is None else ub
    d_thi = np.array([min(abs(lb_ - ub_) / 2, buffer) for lb_, ub_ in zip(lb_, ub_)])
    return lb_ + d_thi, ub_ - d_thi


def est_aug_lik(thi: np.ndarray, z: sde_seed.Partition, h: types.Anchorage, mod: Model, ome: np.random.Generator,
                restrict: int = None) -> (float, sde_seed.Partition):

    est_st_phi, st_lb_phi, new_z = zip(*[sde_seed.est_poisson_coin(
                                             z_, normop, inv_normop,
                                             mod.eval_disc(thi, y0), mod.eval_bounds_disc(thi, y0), ome)
                                         if (restrict is None or y0 == restrict) else (0, 0, z_)
                                         for z_, y0, (normop, inv_normop) in zip(z, h.yt, mod.gen_normops(thi, *h))])
    return sum(mod.eval_biased_loglik(thi, *h, restrict)) + sum(est_st_phi) + sum(st_lb_phi), new_z


def est_log_aug_lik(thi: np.ndarray, z: sde_seed.Partition, h: types.Anchorage, mod: Model, supp_t: List[np.ndarray],
                    ome: np.random.Generator, restrict: int = None) -> (float, sde_seed.Partition):

    new_z, supp_x = zip(*[sde_seed.interpolate_seed(z_, t_, normop, inv_normop, ome)
                          if (restrict is None or y0 == restrict) else (z_, np.array([]))
                          for t_, y0, z_, (normop, inv_normop) in zip(supp_t, h.yt, z, mod.gen_normops(thi, *h))])
    st_disc = [-dt * np.mean(mod.eval_disc(thi, y0, x_)) for dt, y0, x_ in zip(np.diff(h.t), h.yt, supp_x)
               if restrict is None or y0 == restrict]
    log_p, _ = mod.eval_biased_loglik(thi, *h, restrict)
    return log_p + sum(st_disc), new_z


def est_se(thi: np.ndarray, zz: List[sde_seed.Partition], hh: List[types.Anchorage],
           mod: Model, ctrl: Controls, ome: np.random.Generator, pool: Parallel) -> List[Optional[np.ndarray]]:

    def f_wrap_obj(thi_: np.ndarray, i: int) -> np.ndarray:
        thi_ = np.max([np.min([thi_, opt_bounds[1]], 0), opt_bounds[0]], 0)
        return f_obj(np.vstack([thi[:i], thi_, thi[i+1:]]), i)

    opt_bounds = buffer_bounds(mod.bounds_thi[0], mod.bounds_thi[1], len(thi), ctrl.bound_buffer)
    f_obj = update_objective(thi, zz, hh, mod, ctrl, ome, pool)
    se = []
    for i in range(thi.shape[0]):
        d_thi = np.min([np.abs(thi[i] - opt_bounds[0]), np.abs(thi[i] - opt_bounds[1])])
        if d_thi == 0:
            se.append(None)
        else:
            info_ = Hessian(lambda thi_: -np.mean(f_wrap_obj(thi_, i)))(thi[i]) - np.cov(Gradient(f_wrap_obj)(thi[i], i).T)
            se_ = np.linalg.inv(info_)
            se.append(se_ if np.all(np.diag(se_) > 0) else None)
    return se
