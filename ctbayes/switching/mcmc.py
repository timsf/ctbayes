from typing import Callable, Generator, Iterator, List, NamedTuple, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed

from ctbayes.ea3lib import seed as sde_seed
from ctbayes.mjplib import inference as mjp_inf, skeleton as mjp_skel
from ctbayes.switching import types
from ctbayes.misc.adapt import MyopicRwSampler, MyopicMjpSampler
from ctbayes.misc.bfactories import sample_twocoin, sample_twocoin_joint


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
    eval_bounds_grad: Callable[[np.ndarray, np.ndarray, float, float, float, float, float], np.ndarray]


class Controls(NamedTuple):
    opt_acc_prob: float = 0.2
    pr_portkey: float = 0.1
    n_cores: int = 1
    ea_batch_size: int = 10


def sample_posterior(init_thi: np.ndarray, mod: Model, ctrl: Controls, ome: np.random.Generator
                     ) -> Iterator[Tuple[np.ndarray, np.ndarray, sde_seed.Partition, types.Anchorage]]:

    lam = mjp_inf.get_ev(*mod.hyper_lam)
    thi = np.repeat(init_thi[np.newaxis], lam.shape[0], 0)
    y = mjp_skel.sample_forwards(mod.t[-1], None, lam, ome)
    z, h = mod.sample_aug(thi, y, mod.t, mod.vt, ome)
    param_samplers = [[MyopicRwSampler(init_thi_, -np.log(len(mod.t)) * lam.shape[0] / 2, bounds_thi_, ctrl.opt_acc_prob)
                       for init_thi_, bounds_thi_ in zip(init_thi, np.array(mod.bounds_thi).T)]
                      for _ in range(lam.shape[0])]
    regime_sampler = MyopicMjpSampler(y, ctrl.opt_acc_prob)
    while True:
        thi, lam, z, h = update_joint(thi, lam, z, h, mod, ctrl, param_samplers, regime_sampler, ome)
        yield thi, lam, z, h


def update_joint(thi: np.ndarray, lam: np.ndarray, z: sde_seed.Partition, h: types.Anchorage,
                 mod: Model, ctrl: Controls, param_sampler: List[List[MyopicRwSampler]], 
                 regime_sampler: MyopicMjpSampler, ome: np.random.Generator
                 ) -> (np.ndarray, np.ndarray, sde_seed.Partition, types.Anchorage):

    with Parallel(ctrl.n_cores, 'loky') as pool:
        for sector in range(thi.shape[1]):
            thi, z = update_params(thi, z, h, mod, ctrl, param_sampler, sector, ome, pool)
            z, h = update_hidden(thi, lam, z, h, mod, ctrl, regime_sampler, ome, pool)
    lam = update_generator(h, mod.hyper_lam, ome)
    return thi, lam, z, h


def update_params(thi_nil: np.ndarray, z: sde_seed.Partition, h: types.Anchorage, mod: Model, ctrl: Controls, 
                  param_samplers: List[List[MyopicRwSampler]], sector: int, ome: np.random.Generator, pool: Parallel
                  ) -> (np.ndarray, sde_seed.Partition):

    new_ome = [np.random.default_rng(seed_) for seed_ in ome.bit_generator._seed_seq.spawn(thi_nil.shape[0])]
    thi_part, z_part = zip(*pool(
        delayed(update_params_section)(thi_nil, z, h, mod, ctrl, param_samplers, i, sector, ome_)
        for i, ome_ in enumerate(new_ome)))

    thi_prime = np.array([thi_part[i][i] for i in range(thi_nil.shape[0])])
    new_z = [z_part[y0][i] for i, y0 in enumerate(h.yt[:-1])]
    return thi_prime, new_z


def update_params_section(thi_nil: np.ndarray, z: sde_seed.Partition, h: types.Anchorage,
                          mod: Model, ctrl: Controls, param_samplers: List[List[MyopicRwSampler]],
                          state: int, sector: int, ome: np.random.Generator) -> (np.ndarray, sde_seed.Partition):

    def coin() -> Generator[Tuple[bool, sde_seed.Partition], bool, None]:
        new_z = z
        upside = (yield None, new_z)
        while True:
            success, new_z = flip_param_coins(thi_nil, thi_prime, new_z, h, mod, ctrl, upside, ome, state)
            upside = (yield success, new_z)

    prop, log_prop_odds = param_samplers[state][sector].propose(ome)
    thi_prime = thi_nil.copy()
    thi_prime[state, sector] = prop

    if not flip_param_precoin(thi_nil, thi_prime, mod, ome):
        param_samplers[state][sector].adapt(thi_nil[state][sector], 0)
        return thi_nil, z

    weight_nil = eval_param_weight(thi_nil, h, mod, state) + log_prop_odds
    weight_prime = eval_param_weight(thi_prime, h, mod, state)
    accept, _, z_acc = sample_twocoin_joint(weight_prime, weight_nil, coin(), ctrl.pr_portkey, ome)
    thi_acc = thi_prime if accept else thi_nil
    param_samplers[state][sector].adapt(thi_acc[state][sector], float(accept))
    return thi_acc, z_acc


def eval_param_weight(thi: np.ndarray, h: types.Anchorage, mod: Model, state: int = None) -> float:

    return mod.eval_biased_loglik(thi, *h, state)[0]


def flip_param_precoin(thi_nil: np.ndarray, thi_prime: np.ndarray, mod: Model, ome: np.random.Generator) -> bool:

    log_p_prime = mod.eval_log_prior(thi_prime)
    log_p_nil = mod.eval_log_prior(thi_nil)
    return np.log(ome.uniform()) < log_p_prime - np.logaddexp(log_p_prime, log_p_nil)


def flip_param_coins(thi_nil: np.ndarray, thi_prime: np.ndarray, z: sde_seed.Partition, h: types.Anchorage, mod: Model,
                     ctrl: Controls, upside: bool, ome: np.random.Generator, state: int = None
                     ) -> (bool, sde_seed.Partition):

    success, new_z = zip(*[
        flip_param_coin(thi_nil, thi_prime, dt, y0, v0, v1, z_, mod, ctrl, upside, ome)
        if y0 == state else (True, z_)
        for dt, y0, v0, v1, z_ in zip(np.diff(h.t), h.yt, h.vt, h.vt[1:], z)])
    return all(success), list(new_z)


def flip_param_coin(thi_nil: np.ndarray, thi_prime: np.ndarray, fin_t: float, init_y: int, init_v: float, fin_v: float,
                    z_: sde_seed.SeedSkeleton, mod: Model, ctrl: Controls, upside: bool, ome: np.random.Generator
                    ) -> (bool, sde_seed.SeedSkeleton):

    ub_abs_dphi = mod.eval_bounds_grad(thi_nil[init_y], thi_prime[init_y], fin_t, init_v, fin_v, z_.loose_z, z_.tight_z)
    is_updating = (thi_nil != thi_prime)[init_y]
    intensity = np.sum(np.abs((thi_nil - thi_prime)[init_y, is_updating])) * np.sum(np.abs(ub_abs_dphi[is_updating]))
    t, yt, vt = np.array([0, fin_t]), np.array([init_y, init_y]), np.array([init_v, fin_v])
    return sde_seed.flip_dual_coin(z_, intensity,
                                   *mod.gen_normops(thi_nil, t, yt, vt)[0], mod.eval_disc(thi_nil, init_y),
                                   *mod.gen_normops(thi_prime, t, yt, vt)[0], mod.eval_disc(thi_prime, init_y),
                                   upside, ome, ctrl.ea_batch_size)


def update_hidden(thi: np.ndarray, lam: np.ndarray, z_nil: sde_seed.Partition, h_nil: types.Anchorage,
                  mod: Model, ctrl: Controls, regime_sampler: MyopicMjpSampler, ome: np.random.Generator, pool: Parallel
                  ) -> (mjp_skel.Skeleton, sde_seed.Partition, types.Anchorage):

    y_prime, t_cond = regime_sampler.propose(lam, mod.t[1:-1], ome)
    z_prime, h_prime = mod.sample_aug(thi, y_prime, mod.t, mod.vt, ome)
    z_nil_part, h_nil_part = split_hidden(z_nil, h_nil, t_cond)
    z_prime_part, h_prime_part = split_hidden(z_prime, h_prime, t_cond)
    new_ome = [np.random.default_rng(seed_) for seed_ in ome.bit_generator._seed_seq.spawn(len(z_nil_part))]
    
    acc, z_acc_part, h_acc_part = zip(*pool(
        delayed(update_hidden_section)(thi, lam, z_nil_, h_nil_, z_prime_, h_prime_, mod, ctrl, ome_)
        for z_nil_, z_prime_, h_nil_, h_prime_, ome_
        in zip(z_nil_part, z_prime_part, h_nil_part, h_prime_part, new_ome)))
    
    z_acc, h_acc = paste_hidden(z_acc_part, h_acc_part)
    regime_sampler.adapt(np.mean(acc))
    return z_acc, h_acc


def update_hidden_section(thi: np.ndarray, lam: np.ndarray, z_nil: sde_seed.Partition, h_nil: types.Anchorage,
                          z_prime: sde_seed.Partition, h_prime: types.Anchorage, mod: Model, ctrl: Controls, 
                          ome: np.random.Generator) -> (bool, sde_seed.Partition, types.Anchorage):

    def coin(z: sde_seed.Partition, h: types.Anchorage) -> Iterator[Tuple[bool, sde_seed.Partition]]:
        new_z = z
        while True:
            success, new_z = flip_hidden_coins(thi, new_z, h, mod, ctrl, ome)
            yield success, new_z

    weight_nil = eval_hidden_weight(thi, z_nil, h_nil, mod)
    weight_prime = eval_hidden_weight(thi, z_prime, h_prime, mod)
    accept, _, z_acc = sample_twocoin(weight_prime, weight_nil, coin(z_prime, h_prime), coin(z_nil, h_nil),
                                      ctrl.pr_portkey, z_nil, ome=ome)
    if accept:
        return accept, z_acc, h_prime
    return accept, z_acc, h_nil


def eval_hidden_weight(thi: np.ndarray, z: sde_seed.Partition, h: types.Anchorage, mod: Model) -> float:

    st_lb_phi = [sde_seed.integrate_disc_bound(z_, normop, inv_normop, mod.eval_bounds_disc(thi, y0))
                 for z_, y0, (normop, inv_normop) in zip(z, h.yt, mod.gen_normops(thi, *h))]
    return sum(st_lb_phi) + sum(mod.eval_biased_loglik(thi, *h))


def flip_hidden_coins(thi: np.ndarray, z: sde_seed.Partition, h: types.Anchorage, mod: Model, ctrl: Controls,
                      ome: np.random.Generator) -> (bool, sde_seed.Partition):

    success, _, new_z = zip(*[sde_seed.flip_poisson_coin(z_, normop, inv_normop, mod.eval_disc(thi, y0),
                                                         mod.eval_bounds_disc(thi, y0), ome, ctrl.ea_batch_size)
                              for z_, y0, (normop, inv_normop) in zip(z, h.yt, mod.gen_normops(thi, *h))])
    return all(success), list(new_z)


def split_hidden(z: sde_seed.Partition, h: types.Anchorage, new_t: np.ndarray
                 ) -> (List[sde_seed.Partition], List[types.Anchorage]):

    break_r = [0] + list(np.where(np.isin(h.t[1:-1], new_t))[0] + 1) + [len(h.t) - 1]
    z_part = [z[r0:r1] for r0, r1 in zip(break_r, break_r[1:])]
    h_part = [types.Anchorage(h.t[r0:r1+1] - h.t[r0], h.yt[r0:r1+1], h.vt[r0:r1+1])
              for r0, r1 in zip(break_r, break_r[1:])]
    return z_part, h_part


def paste_hidden(z_part: List[sde_seed.Partition], h_part: List[types.Anchorage]
                 ) -> (sde_seed.Partition, types.Anchorage):

    z = sum(z_part, [])
    h = h_part[0]
    for h_ in h_part[1:]:
        h = types.Anchorage(np.append(h.t, h_.t[1:] + h.t[-1]), np.append(h.yt, h_.yt[1:]), np.append(h.vt, h_.vt[1:]))
    return z, h


def update_generator(h: types.Anchorage, lam0: (np.ndarray, np.ndarray), ome: np.random.Generator) -> np.ndarray:

    y = types.prune_anchorage(h)
    return mjp_inf.sample_param(*mjp_inf.update(y, *lam0), ome)
