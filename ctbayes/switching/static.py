from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np
from toolz import curry

from ctbayes.ea3lib import seed as sde_seed
from ctbayes.mjplib import skeleton as mjp_skel
from ctbayes.sdelib.paths import sample_brownbr
from ctbayes.switching import types, mcem, mcmc


def assemble_model(init_thi, bounds_x, bounds_thi,
                   reduce, dereduce, dv_reduce, eval_drift, sx_eval_drift, dx_eval_drift, eval_vol,
                   eval_bounds_disc, eval_bounds_grad, eval_log_prior):

    def maximize_posterior(t: np.ndarray, vt: np.ndarray, n_regimes: int, a0: float = 1, b0: float = 1,
                           ome: np.random.Generator = np.random.default_rng(), **kwargs
                           ) -> Iterator[Tuple[float, np.ndarray, np.ndarray, List[sde_seed.Partition], List[types.Anchorage]]]:

        lam0 = construct_gen_hyperprior(n_regimes, a0, b0)
        mod = mcem.Model(t, vt, bounds_thi, lam0, eval_log_prior, sample_aug, gen_normops,
                         eval_biased_log_lik, eval_disc, eval_bounds_disc)
        ctrl = mcem.Controls(**kwargs)
        return mcem.maximize_posterior(init_thi, mod, ctrl, ome)

    def sample_posterior(t: np.ndarray, vt: np.ndarray, n_regimes: int, a0: float = 1, b0: float = 1,
                         ome: np.random.Generator = np.random.default_rng(), **kwargs
                         ) -> Iterator[Tuple[np.ndarray, np.ndarray, sde_seed.Partition, types.Anchorage]]:

        lam0 = construct_gen_hyperprior(n_regimes, a0, b0)
        mod = mcmc.Model(t, vt, bounds_thi, lam0, eval_log_prior, sample_aug, gen_normops,
                         eval_biased_log_lik, eval_disc, eval_bounds_disc, eval_bounds_grad)
        ctrl = mcmc.Controls(**kwargs)
        return mcmc.sample_posterior(init_thi, mod, ctrl, ome)

    def gen_normops(thi: np.ndarray, t: np.ndarray, yt: np.ndarray, vt: np.ndarray,
                    ) -> List[Tuple[Callable, Callable]]:

        xt = reduce(vt)
        return [(norm_rsde(thi, dt, y0, x0, x1), denorm_rsde(thi, dt, y0, x0, x1))
                for dt, y0, x0, x1 in zip(np.diff(t), yt[:-1], xt, xt[1:])]

    def sample_aug(thi: np.ndarray, y: mjp_skel.Skeleton, t: np.ndarray, vt: np.ndarray, ome: np.random.Generator
                   ) -> (sde_seed.Partition, types.Anchorage):

        xt = reduce(vt)
        y_part = mjp_skel.partition_skeleton(y, t[1:-1])
        xtt = np.hstack([dereduce(sample_anchors(thi, t_, yt_, fin_t_, x0, xt, ome)) for (t_, yt_, fin_t_), x0, xt
                         in zip(y_part, xt, xt[1:])] + [vt[-1]])
        tt = np.append(np.hstack([t0 + y_.t for t0, y_ in zip(t, y_part)]), t[-1])
        ytt = np.append(np.hstack([y_.xt for y_ in y_part]), y.xt[-1])
        vtt = dereduce(xtt)
        z = [sde_seed.sample_raw_seed(bounds_x, normop, inv_normop, ome)
             for (normop, inv_normop) in gen_normops(thi, tt, ytt, vtt)]
        return z, types.Anchorage(tt, ytt, vtt)

    def sample_anchors(thi: np.ndarray, t: np.ndarray, yt: np.ndarray, fin_t: float, init_x: float, fin_x: float,
                       ome: np.random.Generator) -> np.ndarray:

        if len(t) == 1:
            return np.array([init_x])
        sig_t = np.diff(np.append(t, fin_t)) * np.square(eval_vol(thi, yt))
        s = np.cumsum(sig_t)
        xt = sample_brownbr(s[:-1], s[-1], init_x, fin_x, ome)
        return np.hstack([init_x, xt])

    def eval_biased_log_lik(thi: np.ndarray, t: np.ndarray, yt: np.ndarray, vt: np.ndarray, restrict: int = None
                            ) -> (float, float):

        xt = reduce(vt)
        is_obs = np.hstack([True, np.diff(yt) == 0])
        sig_t = np.diff(t) * np.square(eval_vol(thi, yt[:-1]))
        sig_t_obs = np.array([np.sum(sig_t_) for sig_t_ in np.split(sig_t, np.where(is_obs)[0])[1:-1]])
        p_del = (sx_eval_drift(thi, yt[:-1], xt[1:]) - sx_eval_drift(thi, yt[:-1], xt[:-1])) / np.square(eval_vol(thi, yt[:-1]))
        p_vt = np.log(np.abs(dv_reduce(vt[1:]))) - np.log(2 * np.pi * sig_t) / 2 - np.square(np.diff(xt)) / (2 * sig_t)
        p_vt_obs = np.sum(np.log(np.abs(dv_reduce(vt[is_obs][1:]))) - np.log(2 * np.pi * sig_t_obs) / 2 - np.square(np.diff(xt[is_obs])) / (2 * sig_t_obs))
        is_selected = yt[:-1] == restrict if restrict is not None else np.repeat(True, len(t) - 1)
        return np.sum(p_del[is_selected] + p_vt[is_selected]), p_vt_obs - np.sum(p_vt[is_selected])

    @curry
    def eval_disc(thi: np.ndarray, yt: np.ndarray, xt: np.ndarray) -> np.ndarray:

        return (np.square(eval_drift(thi, yt, xt) / eval_vol(thi, yt)) + dx_eval_drift(thi, yt, xt)) / 2

    @curry
    def norm_rsde(thi: np.ndarray, fin_t: float, init_y: int, init_x: float, fin_x: float, t: np.ndarray, xt: np.ndarray
                  ) -> (np.ndarray, np.ndarray):

        s = t / fin_t
        if xt is not None:
            return s, (xt - init_x - s * (fin_x - init_x)) / (np.sqrt(fin_t) * eval_vol(thi, init_y))
        return s, None

    @curry
    def denorm_rsde(thi: np.ndarray, fin_t: float, init_y: int, init_x: float, fin_x: float, s: np.ndarray, zs: np.ndarray
                    ) -> (np.ndarray, np.ndarray):

        t = s * fin_t
        if zs is not None:
            return t, zs * np.sqrt(fin_t) * eval_vol(thi, init_y) + init_x + s * (fin_x - init_x)
        return t, None

    return eval_disc, eval_biased_log_lik, sample_aug, gen_normops, \
        maximize_posterior, sample_posterior


def construct_gen_hyperprior(n_regimes: int, alp: float, bet: float) -> (np.ndarray, np.ndarray):

    ones = np.ones((n_regimes, n_regimes)) + np.diag(np.repeat(np.nan, n_regimes))
    return (alp * ones, bet * ones)


