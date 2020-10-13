from typing import Callable, Iterator, List, NamedTuple, Optional, Tuple

import numpy as np

from ctbayes.mjplib import skeleton
from ctbayes.sdelib import interpolators, layers, transforms
from ctbayes.misc.exceptions import BudgetConstraintError


class Skeleton(NamedTuple):
    t: np.ndarray
    x: np.ndarray
    tight_x: float
    loose_x: float
    hit_x: float
    hit_i: Optional[List[bool]] = None

    def __call__(self, t: np.ndarray, ome: np.random.Generator) -> np.ndarray:
        if np.any(t > self.t[-1]):
            raise ValueError
        return interpolate_skeleton(self, t, ome)[1]


Partition = List[Skeleton]


def sample_skeleton(fin_t: float, init_x: float, bounds_x: (float, float),
                    endpoint_sampler: Iterator[float], eval_disc: Callable[[np.ndarray], np.ndarray],
                    eval_bounds_disc: Callable[[float, float], Tuple[float, float, float]],
                    ome: np.random.Generator, max_props: int = int(1e4)) -> Iterator[Skeleton]:
    """
    :param fin_t:
    :param init_x:
    :param bounds_x:
    :param endpoint_sampler:
    :param eval_disc:
    :param eval_bounds_disc:
    :param ome:
    :param max_props:
    :return:
    :raise: BudgetConstraintError
    """

    while True:

        for _ in range(max_props):
            proposal = sample_raw_skeleton(fin_t, init_x, next(endpoint_sampler), bounds_x, ome)
            success, proposal = flip_poisson_coin(proposal, eval_disc, eval_bounds_disc, ome)
            if success:
                break
        else:
            raise BudgetConstraintError('None of the proposals were accepted.')

        yield proposal


def sample_raw_skeleton(fin_t: float, init_x: float, fin_x: float, bounds_x: (float, float),
                        ome: np.random.Generator) -> Skeleton:
    """
    :param fin_t:
    :param init_x:
    :param fin_x:
    :param bounds_x:
    :param ome:
    :return:
    """

    lo_anchor, lo_sector, hi_sector = layers.sample_anchor(fin_t, init_x, fin_x, *bounds_x, ome=ome)
    tight_t, tight_x, hit_x, loose_x, hit_i = layers.sample_edges(fin_t, init_x, fin_x, lo_sector, hi_sector, ome=ome)
    skel = Skeleton(np.array([0, tight_t, fin_t]), np.array([init_x, tight_x, fin_x]), tight_x, loose_x, hit_x, hit_i)
    if ome.uniform() < .5:
        return skel
    return flip_skeleton(skel)


def flip_poisson_coin(skel: Skeleton, eval_disc: Callable[[np.ndarray], np.ndarray],
                      eval_bounds_disc: Callable[[float, float], Tuple[float, float, float]],
                      ome: np.random.Generator, batch_size: int = 2) -> (bool, Skeleton):
    """
    :param skel:
    :param eval_disc:
    :param eval_bounds_disc:
    :param ome:
    :param batch_size:
    :return:
    """

    le_x, ue_x = min(skel.loose_x, skel.tight_x), max(skel.loose_x, skel.tight_x)
    cent_phi, le_phi, ue_phi = eval_bounds_disc(le_x, ue_x)

    new_skel = skel
    for ppp in skeleton.sample_batch_ppp(1, (ue_phi - cent_phi, skel.t[-1]), batch_size, ome):
        crit_phi, new_t = ppp.T[:, np.argsort(ppp.T[1])]
        new_skel, new_x = interpolate_skeleton(new_skel, new_t, ome)
        if np.any(eval_disc(new_x) - cent_phi > crit_phi):
            return False, new_skel
    return True, new_skel


def est_poisson_coin(skel: Skeleton,
                     eval_disc: Callable[[np.ndarray], np.ndarray],
                     eval_bounds_disc: Callable[[float, float], Tuple[float, float, float]],
                     ome: np.random.Generator) -> (float, Skeleton):
    """
    :param skel:
    :param eval_disc:
    :param eval_bounds_disc:
    :param ome:
    :return:
    """

    le_x, ue_x = min(skel.loose_x, skel.tight_x), max(skel.loose_x, skel.tight_x)
    cent_phi, le_phi, ue_phi = eval_bounds_disc(le_x, ue_x)
    intensity = ue_phi - le_phi

    new_t, = skeleton.sample_ppp(intensity, (skel.t[-1],), ome).T
    if len(new_t) > 0:
        new_skel, new_x = interpolate_skeleton(skel, new_t, ome)
        new_phi = eval_disc(new_x)
        return np.sum(np.log(ue_phi - new_phi)) - len(new_t) * np.log(intensity) + (intensity - ue_phi + cent_phi) * skel.t[-1], new_skel
    return (intensity - ue_phi + cent_phi) * skel.t[-1], skel


def interpolate_skeleton(skel: Skeleton, new_t: np.ndarray, ome: np.random.Generator) -> (Skeleton, np.ndarray):
    """
    :param skel:
    :param new_t:
    :param ome:
    :return:
    """

    if len(new_t) == 0:
        return skel, np.array([])

    if skel.tight_x < skel.loose_x:
        skel_std, new_t_std = skel, new_t
    else:
        skel_std = flip_skeleton(skel)
        new_t_std, _ = transforms.reverse(new_t, np.empty(0), skel.t[-1])

    t_std, x_std, i_std, hit_i_std = interpolators.fill_wiener_outer(
        skel_std.t, skel_std.x, new_t_std, skel_std.tight_x, skel_std.loose_x, skel_std.hit_x, skel_std.hit_i, ome)
    new_skel_std = Skeleton(t_std, x_std, skel_std.tight_x, skel_std.loose_x, skel_std.hit_x, hit_i_std)

    if skel.tight_x < skel.loose_x:
        new_skel, new_x = new_skel_std, x_std[i_std]
    else:
        new_skel = flip_skeleton(new_skel_std)
        new_x = new_skel.x[len(t_std) - 1 - i_std[::-1]]

    return new_skel, new_x


def flip_skeleton(skel: Skeleton) -> Skeleton:
    """
    :param skel:
    :return:
    """

    reflect_x = (skel.x[0] + skel.x[-1]) / 2
    t, x = transforms.reflect(*transforms.reverse(skel.t, skel.x, skel.t[-1]), reflect_x)
    tight_x = transforms.reflect(None, np.array([skel.tight_x]), reflect_x)[1].item()
    loose_x = transforms.reflect(None, np.array([skel.loose_x]), reflect_x)[1].item()
    hit_x = transforms.reflect(None, np.array([skel.hit_x]), reflect_x)[1].item()
    if skel.hit_i is None:
        hit_i = skel.hit_i
    else:
        hit_i = skel.hit_i[::-1]

    return Skeleton(t, x, tight_x, loose_x, hit_x, hit_i)
