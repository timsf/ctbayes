from typing import Callable, Generator, Iterator, List, Optional, Tuple, NamedTuple

import numpy as np
from joblib import Parallel, delayed

from ctbayes.ea3lib import seed
from ctbayes.mjplib.skeleton import sample_batch_ppp
from ctbayes.misc.bfactories import sample_twocoin, sample_twocoin_joint
from ctbayes.misc.rw import MyopicRwSampler


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
    eval_bounds_grad: Callable[[np.ndarray, np.ndarray, float, float, float, float, float], np.ndarray]


class Controls(NamedTuple):
    pr_portkey: float = 0.1
    n_aux_renewals: int = 1
    n_cores: int = 1
    ea_batch_size: int = 10


def sample_posterior(init_thi: np.ndarray, mod: Model, ctrl: Controls, ome: np.random.Generator
                     ) -> Iterator[Tuple[np.ndarray, seed.Partition]]:

    thi = init_thi
    z = mod.sample_aug(thi, mod.t, mod.vt, ome)
    param_samplers = [MyopicRwSampler(init_thi_, -np.log(len(mod.t)), bounds_thi_)
                      for init_thi_, bounds_thi_ in zip(init_thi, np.array(mod.bounds_thi).T)]
    while True:
        thi, z = update_joint(thi, z, mod, ctrl, param_samplers, ome)
        yield thi, z


def update_joint(thi: np.ndarray, z: seed.Partition, mod: Model, ctrl: Controls,
                 param_samplers: List[MyopicRwSampler], ome: np.random.Generator) -> (np.ndarray, seed.Partition):

    with Parallel(ctrl.n_cores, 'loky') as pool:
        thi, z = update_params(thi, z, mod, ctrl, param_samplers, ome)
        for _ in range(ctrl.n_aux_renewals):
            z = update_sde(thi, z, mod, ctrl, ome, pool)
    return thi, z


def update_params(thi_nil: np.ndarray, z: seed.Partition, mod: Model, ctrl: Controls,
                  param_samplers: List[MyopicRwSampler], ome: np.random.Generator) -> (seed.Partition, np.ndarray):

    def coin() -> Generator[Tuple[bool, seed.Partition], bool, None]:
        new_z = z
        state = (yield None, new_z)
        while True:
            success, new_z = flip_param_coins(thi_nil, thi_prime, new_z, mod, ctrl, state, ome)
            state = (yield success, new_z)

    sector = ome.integers(0, len(thi_nil))
    prop, log_prop_odds = param_samplers[sector].propose(ome)
    thi_prime = np.array([thi_nil[i] if i != sector else prop for i in range(len(thi_nil))])

    if not flip_param_precoin(thi_nil, thi_prime, mod, ome):
        param_samplers[sector].adapt(thi_nil[sector], 0)
        return thi_nil, z

    weight_nil = eval_param_weight(thi_nil, mod) + log_prop_odds
    weight_prime = eval_param_weight(thi_prime, mod)
    accept, _, z_acc = sample_twocoin_joint(weight_prime, weight_nil, coin(), ctrl.pr_portkey, ome)
    thi_acc = thi_prime if accept else thi_nil
    param_samplers[sector].adapt(thi_acc[sector], float(accept))
    return thi_acc, z_acc


def eval_param_weight(thi: np.ndarray, mod: Model) -> float:

    return mod.eval_biased_loglik(thi, mod.t, mod.vt)


def flip_param_precoin(thi_nil: np.ndarray, thi_prime: np.ndarray, mod: Model, ome: np.random.Generator) -> bool:

    log_p_prime = mod.eval_log_prior(thi_prime)
    log_p_nil = mod.eval_log_prior(thi_nil)
    return np.log(ome.uniform()) < log_p_prime - np.logaddexp(log_p_prime, log_p_nil)


def flip_param_coins(thi_nil: np.ndarray, thi_prime: np.ndarray, z: seed.Partition, mod: Model, ctrl: Controls,
                     upside: bool, ome: np.random.Generator) -> (bool, seed.Partition):

    success, new_z = zip(*[
        flip_param_coin(thi_nil, thi_prime, z_, Model(mod.t[i:(i + 2)] - mod.t[i], mod.vt[i:(i + 2)], *mod[2:]), ctrl,
                        upside, ome)
        for i, z_ in enumerate(z)])
    return all(success), list(new_z)


def flip_param_coin(thi_nil: np.ndarray, thi_prime: np.ndarray, z_: seed.SeedSkeleton, mod: Model, ctrl: Controls,
                    upside: bool, ome: np.random.Generator) -> (bool, seed.SeedSkeleton):

    ub_abs_dphi = mod.eval_bounds_grad(thi_nil, thi_prime, mod.t[-1], mod.vt[0], mod.vt[-1], z_.loose_z, z_.tight_z)
    is_updating = thi_nil != thi_prime
    intensity = np.sum(np.abs((thi_nil - thi_prime)[is_updating])) * np.sum(np.abs(ub_abs_dphi[is_updating]))
    t, vt = np.array([0, mod.t[-1]]), np.array([mod.vt[0], mod.vt[-1]])
    return seed.flip_dual_coin(z_, intensity,
                               *mod.gen_normops(thi_nil, t, vt)[0], mod.eval_disc(thi_nil),
                               *mod.gen_normops(thi_prime, t, vt)[0], mod.eval_disc(thi_prime),
                               upside, ome, ctrl.ea_batch_size)


def update_sde(thi: np.ndarray, z_nil: seed.Partition, mod: Model, ctrl: Controls, ome: np.random.Generator, 
               pool: Parallel) -> seed.Partition:

    z_prime = mod.sample_aug(thi, mod.t, mod.vt, ome)
    new_ome = [np.random.default_rng(seed_) for seed_ in ome.bit_generator._seed_seq.spawn(len(z_nil))]
    return list(pool(
        delayed(update_sde_bridge)(i, thi, z_nil_, z_prime_, mod, ctrl, ome_)
        for i, z_nil_, z_prime_, ome_ in zip(range(len(z_nil)), z_nil, z_prime, new_ome)))


def update_sde_bridge(sector: int, thi: np.ndarray, z_nil_: seed.SeedSkeleton, z_prime_: seed.SeedSkeleton, mod: Model,
                      ctrl: Controls, ome: np.random.Generator) -> seed.SeedSkeleton:

    def coin(z_: seed.SeedSkeleton) -> Iterator[Tuple[bool, seed.Partition]]:
        new_z_ = z_
        while True:
            success, new_z = flip_sde_coins(thi, [new_z_], submod, ctrl, ome)
            new_z_, = new_z
            yield success, new_z_

    submod = (Model(mod.t[sector:(sector + 2)] - mod.t[sector], mod.vt[sector:(sector + 2)], *mod[2:]))
    weight_nil = eval_sde_weight(thi, [z_nil_], submod)
    weight_prime = eval_sde_weight(thi, [z_prime_], submod)
    _, _, z_acc_ = sample_twocoin(weight_prime, weight_nil, coin(z_prime_), coin(z_nil_), ctrl.pr_portkey, z_nil_, ome)
    return z_acc_


def eval_sde_weight(thi: np.ndarray, z: seed.Partition, mod: Model) -> float:

    st_lb_phi = [seed.integrate_disc_bound(z_, normop, inv_normop, mod.eval_bounds_disc(thi))
                 for z_, (normop, inv_normop) in zip(z, mod.gen_normops(thi, mod.t, mod.vt))]
    return sum(st_lb_phi)


def flip_sde_coins(thi: np.ndarray, z: seed.Partition, mod: Model, ctrl: Controls, ome: np.random.Generator
                   ) -> (bool, seed.Partition):

    success, _, new_z = zip(*[seed.flip_poisson_coin(z_, normop, inv_normop, mod.eval_disc(thi),
                                                     mod.eval_bounds_disc(thi), ome, ctrl.ea_batch_size)
                              for z_, (normop, inv_normop) in zip(z, mod.gen_normops(thi, mod.t, mod.vt))])
    return all(success), list(new_z)
