from itertools import repeat
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np
from toolz import curry

from ctbayes.ea3lib import seed, skeleton
from ctbayes.ito import mcem, mcmc
from ctbayes.misc.ars import sample as sample_logconcave


def assemble_model(init_thi, bounds_x, bounds_thi,
                   reduce, dereduce, dv_reduce, eval_drift, sx_eval_drift, dx_eval_drift,
                   eval_bounds_disc, eval_bounds_grad, eval_log_prior):

    def sample_sde(thi: np.ndarray, t: np.ndarray, fin_t: float, init_v: float, fin_v: Optional[float],
                   ome: np.random.Generator = np.random.default_rng()) -> Iterator[np.ndarray]:

        init_x, = reduce(thi, np.array([init_v]))
        if fin_v is None:
            fin_x_sampler = sample_conv_endpt(thi, fin_t, init_x, ome)
        else:
            fin_x, = reduce(thi, np.array([fin_v]))
            fin_x_sampler = repeat(fin_x)
        x_sampler = skeleton.sample_skeleton(fin_t, init_x, bounds_x, fin_x_sampler,
                                             eval_disc(thi), eval_bounds_disc(thi), ome)
        return (dereduce(thi, skeleton.interpolate_skeleton(x_, t, ome)[1]) for x_ in x_sampler)

    def maximize_posterior(t: np.ndarray, vt: np.ndarray, ome: np.random.Generator = np.random.default_rng(), **kwargs
                           ) -> Iterator[Tuple[float, np.ndarray, List[seed.Partition]]]:

        mod = mcem.Model(t, vt, bounds_thi, eval_log_prior, sample_aug, gen_normops, eval_biased_log_lik,
                         eval_disc, eval_bounds_disc)
        ctrl = mcem.Controls(**kwargs)
        return mcem.maximize_posterior(init_thi, mod, ctrl, ome)

    def sample_posterior(t: np.ndarray, vt: np.ndarray, ome: np.random.Generator = np.random.default_rng(), **kwargs
                         ) -> Iterator[Tuple[np.ndarray, seed.Partition]]:

        mod = mcmc.Model(t, vt, bounds_thi, eval_log_prior, sample_aug, gen_normops, eval_biased_log_lik,
                         eval_disc, eval_bounds_disc, eval_bounds_grad)
        ctrl = mcmc.Controls(**kwargs)
        return mcmc.sample_posterior(init_thi, mod, ctrl, ome)

    def gen_normops(thi: np.ndarray, t: np.ndarray, vt: np.ndarray) -> List[Tuple[Callable, Callable]]:

        xt = reduce(thi, vt)
        return [(norm_rsde(dt, x0, x1), denorm_rsde(dt, x0, x1)) for dt, x0, x1 in zip(np.diff(t), xt, xt[1:])]

    def sample_aug(thi: np.ndarray, t: np.ndarray, vt: np.ndarray, ome: np.random.Generator) -> seed.Partition:

        return [seed.sample_raw_seed(bounds_x, normop, inv_normop, ome)
                for (normop, inv_normop) in gen_normops(thi, t, vt)]

    def eval_biased_log_lik(thi: np.ndarray, t: np.ndarray, vt: np.ndarray) -> float:

        dt = np.diff(t)
        xt = reduce(thi, vt)
        p_v = np.log(np.abs(dv_reduce(thi, vt[1:]))) \
            - np.log(2 * np.pi * dt) / 2 - np.square(np.diff(xt)) / (2 * dt) \
            + sx_eval_drift(thi, xt[1:]) - sx_eval_drift(thi, xt[:-1])
        return sum(p_v)

    @curry
    def eval_disc(thi: np.ndarray, xt: np.ndarray) -> np.ndarray:

        return (np.square(eval_drift(thi, xt)) + dx_eval_drift(thi, xt)) / 2

    @curry
    def norm_rsde(fin_t: float, init_x: float, fin_x: float, t: np.ndarray, xt: np.ndarray) -> (np.ndarray, np.ndarray):

        s = t / fin_t
        if xt is not None:
            return s, (xt - init_x - s * (fin_x - init_x)) / np.sqrt(fin_t)
        return s, None

    @curry
    def denorm_rsde(fin_t: float, init_x: float, fin_x: float, s: np.ndarray, zs: np.ndarray) -> (np.ndarray, np.ndarray):

        t = s * fin_t
        if zs is not None:
            return t, zs * np.sqrt(fin_t) + init_x + s * (fin_x - init_x)
        return t, None

    def sample_conv_endpt(thi: np.ndarray, fin_t: float, init_x: float, ome: np.random.Generator) -> Iterator[float]:

        def f(x):
            return sx_eval_drift(thi, x) - np.square(x - init_x) / (2 * fin_t)
        def df(x):
            return eval_drift(thi, x) - (x - init_x) / fin_t
        return sample_logconcave(f, df, *bounds_x, ome=ome)

    return eval_disc, eval_biased_log_lik, sample_aug, gen_normops, \
        sample_sde, maximize_posterior, sample_posterior
