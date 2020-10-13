from typing import Callable, Iterator, List, NamedTuple, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from numdifftools import Gradient, Hessian
from scipy.optimize import minimize

from ctbayes.ea3lib import seed
from ctbayes.ito import mcmc


class Model(NamedTuple):
    t: np.ndarray
    vt: np.ndarray
    bounds_thi: Tuple[Optional[np.ndarray], Optional[np.ndarray]]
    eval_log_prior: Callable[[np.ndarray], float]
    sample_aug: Callable[[np.ndarray, np.ndarray, np.ndarray, np.random.Generator], seed.Partition]
    gen_normops: Callable[[np.ndarray, np.ndarray, np.ndarray], List[Tuple[Callable, Callable]]]
    eval_biased_loglik: Callable[[np.ndarray, np.ndarray, np.ndarray], float]
    eval_disc: Callable[[np.ndarray, np.ndarray], np.ndarray]
    eval_bounds_disc: Callable[[np.ndarray, float, float], Tuple[float, float, float]]


class Controls(NamedTuple):
    n_thin_mcmc: int = 2
    n_disc_supports: int = 2
    learning_rate: float = 1.5
    ftol: float = 1e-2
    bound_buffer: float = 1e-2
    n_cores: int = 1


def maximize_posterior(init_thi: np.ndarray, mod: Model, ctrl: Controls, ome: np.random.Generator
                       ) -> Iterator[Tuple[float, np.ndarray, List[seed.Partition]]]:

    zz = [mod.sample_aug(init_thi, mod.t, mod.vt, ome)]
    n_particles = 1
    obj_nil = -np.inf
    while True:
        obj, thi, zz = iterate_em(init_thi, zz[-1], n_particles, mod, ctrl, ome)
        n_particles = int((n_particles ** (1 / ctrl.learning_rate) + 1) ** ctrl.learning_rate)
        yield obj, thi, zz
        if np.abs(obj - obj_nil) < ctrl.ftol:
            break
        obj_nil = obj


def iterate_em(thi: np.ndarray, z: seed.Partition, n_particles: int, mod: Model, ctrl: Controls,
               ome: np.random.Generator) -> (float, np.ndarray, List[seed.Partition]):

    with Parallel(ctrl.n_cores, 'loky') as pool:
        zz = update_particles(thi, z, n_particles, mod, ctrl, ome, pool)
        thi, obj = update_param(thi, zz, mod, ctrl, ome, pool)
    return obj, thi, zz


def update_param(thi_nil: np.ndarray, zz: List[seed.Partition], mod: Model, ctrl: Controls, ome: np.random.Generator,
                 pool: Parallel) -> (np.ndarray, float):

    f_obj = update_objective(thi_nil, zz, mod, ctrl, ome, pool)
    opt_bounds = buffer_bounds(mod.bounds_thi[0], mod.bounds_thi[1], len(thi_nil), ctrl.bound_buffer)
    opt = minimize(lambda thi_: -np.mean(f_obj(thi_)), thi_nil, bounds=list(zip(*opt_bounds)), tol=ctrl.ftol)
    return opt.x, -opt.fun


def update_particles(thi: np.ndarray, init_z: seed.Partition, n_particles: int, mod: Model, ctrl: Controls,
                     ome: np.random.Generator, pool: Parallel) -> List[seed.Partition]:

    z, zz = init_z, []
    for i in range(n_particles * ctrl.n_thin_mcmc):
        z = mcmc.update_sde(thi, z, mcmc.Model(*mod, None), mcmc.Controls(ctrl.n_cores), ome, pool)
        if not i % ctrl.n_thin_mcmc:
            zz.append(z)
    return zz


def update_objective(thi: np.ndarray, zz: List[seed.Partition], mod: Model, ctrl: Controls, ome: np.random.Generator,
                     pool: Parallel) -> Callable[[np.ndarray], np.ndarray]:
                     
    def f_obj(thi_prime: np.ndarray) -> np.ndarray:
        log_aug_trans = [est_log_aug_lik(thi_prime, zz_, mod, supp_t, ome)[0] for zz_ in new_zz]
        return np.array(log_aug_trans) + mod.eval_log_prior(thi_prime)

    supp_t = [np.sort(ome.uniform(high=dt, size=ctrl.n_disc_supports)) for dt in np.diff(mod.t)]
    new_ome = [np.random.default_rng(seed_) for seed_ in ome.bit_generator._seed_seq.spawn(len(zz))]
    _, new_zz = zip(*pool(delayed(est_log_aug_lik)(thi, zz_, mod, supp_t, ome_) for zz_, ome_ in zip(zz, new_ome)))

    return f_obj


def buffer_bounds(lb: np.ndarray, ub: np.ndarray, n_dim: int, buffer: float = 1e-2) -> (np.ndarray, np.ndarray):

    lb_ = np.repeat(-np.inf, n_dim) if lb is None else lb
    ub_ = np.repeat(np.inf, n_dim) if ub is None else ub
    d_thi = np.array([min(abs(lb_ - ub_) / 2, buffer) for lb_, ub_ in zip(lb_, ub_)])
    return lb_ + d_thi, ub_ - d_thi


def est_aug_lik(thi: np.ndarray, z: seed.Partition, mod: Model, ome: np.random.Generator) -> (float, seed.Partition):

    est_st_phi, st_lb_phi, new_z = zip(*[seed.est_poisson_coin(
                                             z_, normop, inv_normop, mod.eval_disc(thi), mod.eval_bounds_disc(thi), ome)
                                         for z_, (normop, inv_normop) in zip(z, mod.gen_normops(thi, mod.t, mod.vt))])
    return mod.eval_biased_loglik(thi, mod.t, mod.vt) + sum(est_st_phi) + sum(st_lb_phi), new_z


def est_log_aug_lik(thi: np.ndarray, z: seed.Partition, mod: Model, supp_t: List[np.ndarray], ome: np.random.Generator
                    ) -> (float, seed.Partition):

    new_z, supp_x = zip(*[seed.interpolate_seed(z_, t_, normop, inv_normop, ome)
                          for t_, z_, (normop, inv_normop) in zip(supp_t, z, mod.gen_normops(thi, mod.t, mod.vt))])
    st_disc = [-dt * np.mean(mod.eval_disc(thi, x_)) for dt, x_ in zip(np.diff(mod.t), supp_x)]
    return mod.eval_biased_loglik(thi, mod.t, mod.vt) + sum(st_disc), new_z


def est_se(thi: np.ndarray, zz: List[seed.Partition], mod: Model, ctrl: Controls, ome: np.random.Generator,
           pool: Parallel) -> Optional[np.ndarray]:

    def f_wrap_obj(thi_: np.ndarray) -> np.ndarray:
        thi_ = np.max([np.min([thi_, opt_bounds[1]], 0), opt_bounds[0]], 0)
        return f_obj(thi_)

    f_obj = update_objective(thi, zz, mod, ctrl, ome, pool)
    opt_bounds = buffer_bounds(mod.bounds_thi[0], mod.bounds_thi[1], len(thi), ctrl.bound_buffer)
    d_thi = np.min([np.abs(thi - opt_bounds[0]), np.abs(thi - opt_bounds[1])])
    if d_thi == 0:
        return None
    info = Hessian(lambda thi_: -np.mean(f_wrap_obj(thi_)))(thi) - np.cov(Gradient(f_wrap_obj)(thi).T)
    se = np.linalg.inv(info)
    return se if np.all(np.diag(se) > 0) else None


# def est_aug_lik_bin(thi: np.ndarray, z: seed.Partition, mod: Model) -> (float, seed.Partition):
#
#     est_st_phi, st_lb_phi, new_z = zip(*[seed.flip_poisson_coin(
#                                              z_, normop, inv_normop, mod.eval_disc(thi), mod.eval_bounds_disc(thi))
#                                          for z_, (normop, inv_normop) in zip(z, mod.gen_normops(thi, mod.t, mod.vt))])
#     return mod.eval_biased_loglik(thi, mod.t, mod.vt) + (0 if all(est_st_phi) else -np.inf) + sum(st_lb_phi), new_z
