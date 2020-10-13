from typing import Callable, Iterator, List, NamedTuple, Optional, Tuple, Union

import numpy as np

from ctbayes.ea3lib import skeleton
from ctbayes.mjplib.skeleton import sample_ppp, sample_batch_ppp
from ctbayes.misc.exceptions import BudgetConstraintError


class SeedSkeleton(NamedTuple):
    s: np.ndarray
    z: np.ndarray
    tight_z: float
    loose_z: float
    hit_z: float
    hit_i: Optional[List[bool]] = None

    def __call__(self, t: np.ndarray,
                 norm_rsde: Callable[[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, np.ndarray]],
                 denorm_rsde: Callable[[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, np.ndarray]],
                 ome: np.random.Generator) -> np.ndarray:
        if np.any(t > denorm_rsde(np.ones(1), None)[0]):
            raise ValueError
        return interpolate_seed(self, t, norm_rsde, denorm_rsde, ome)[1]


Partition = List[SeedSkeleton]


def sample_bridge(bounds_x: (float, float),
                  norm_rsde: Callable[[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, np.ndarray]],
                  denorm_rsde: Callable[[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, np.ndarray]],
                  eval_disc: Callable[[np.ndarray], np.ndarray],
                  eval_bounds_disc: Callable[[float, float], Tuple[float, float, float]],
                  ome: np.random.Generator,
                  max_props: int = int(1e4)) -> Iterator[SeedSkeleton]:
    """
    :param bounds_x:
    :param norm_rsde:
    :param denorm_rsde:
    :param eval_disc:
    :param eval_bounds_disc:
    :param ome:
    :param max_props:
    :return:
    :raise: BudgetConstraintError
    """

    assert bounds_x[0] < bounds_x[1]
    assert 0 <= max_props

    while True:

        for _ in range(max_props):
            proposal = sample_raw_seed(bounds_x, norm_rsde, denorm_rsde, ome)
            success, log_weight, proposal = flip_poisson_coin(proposal, norm_rsde, denorm_rsde, eval_disc, eval_bounds_disc, ome)
            if success and np.log(ome.uniform()) < log_weight:
                break
        else:
            raise BudgetConstraintError('None of the proposals were accepted.')

        yield proposal


def sample_raw_seed(bounds_x: (float, float),
                    norm_rsde: Callable[[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, np.ndarray]],
                    denorm_rsde: Callable[[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, np.ndarray]],
                    ome: np.random.Generator) -> SeedSkeleton:
    """
    :param bounds_x:
    :param norm_rsde:
    :param denorm_rsde:
    :param ome:
    :return:
    """

    t, _ = denorm_rsde(np.array([0, 1]), None)
    lb_z = np.max(norm_rsde(t, np.repeat(bounds_x[0], 2))[1])
    ub_z = np.min(norm_rsde(t, np.repeat(bounds_x[1], 2))[1])
    return SeedSkeleton(*skeleton.sample_raw_skeleton(1, 0, 0, (lb_z, ub_z), ome))


def flip_poisson_coin(seed: SeedSkeleton,
                      norm_rsde: Callable[[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, np.ndarray]],
                      denorm_rsde: Callable[[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, np.ndarray]],
                      eval_disc: Callable[[np.ndarray], np.ndarray],
                      eval_bounds_disc: Callable[[float, float], Tuple[float, float, float]],
                      ome: np.random.Generator,
                      batch_size: int = 2) -> (bool, float, SeedSkeleton):
    """
    :param seed:
    :param norm_rsde:
    :param denorm_rsde,
    :param eval_disc:
    :param eval_bounds_disc:
    :param ome:
    :param batch_size:
    :return:
    """

    t, le_phi, ue_phi = bound_disc_path(seed, norm_rsde, denorm_rsde, eval_bounds_disc, 1)
    st_lb_disc = -np.diff(t) @ le_phi
    intensity = np.max(ue_phi - le_phi)
    if np.isinf(intensity):
        return False, st_lb_disc, seed

    fin_t, = denorm_rsde(np.ones(1), None)[0]
    new_seed = seed
    for ppp in sample_batch_ppp(1, (intensity, fin_t), batch_size, ome):
        crit_phi, new_t = ppp.T[:, np.argsort(ppp.T[1])]
        new_seed, new_x = interpolate_seed(new_seed, new_t, norm_rsde, denorm_rsde, ome)
        new_phi = eval_disc(new_x)
        if np.any(new_phi - le_phi[np.searchsorted(t, new_t, 'right') - 1] > crit_phi):
            return False, st_lb_disc, new_seed
    return True, st_lb_disc, new_seed


def flip_dual_coin(seed: SeedSkeleton,
                   intensity: float,
                   norm_rsde_nil: Callable[[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, np.ndarray]],
                   denorm_rsde_nil: Callable[[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, np.ndarray]],
                   eval_disc_nil: Callable[[np.ndarray], np.ndarray],
                   norm_rsde_prime: Callable[[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, np.ndarray]],
                   denorm_rsde_prime: Callable[[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, np.ndarray]],
                   eval_disc_prime: Callable[[np.ndarray], np.ndarray],
                   upside: bool,
                   ome: np.random.Generator,
                   batch_size: int = 2) -> (bool, float, SeedSkeleton):
    """
    :param seed:
    :param intensity:
    :param norm_rsde_nil,
    :param denorm_rsde_nil:
    :param eval_disc_nil:
    :param norm_rsde_prime:
    :param denorm_rsde_prime:
    :param eval_disc_prime:
    :param upside:
    :param ome:
    :param batch_size:
    :return:
    """

    fin_t, = denorm_rsde_nil(np.ones(1), None)[0]
    new_seed = seed
    for ppp in sample_batch_ppp(1, (intensity, fin_t), batch_size, ome):
        crit_gap, new_t = ppp.T[:, np.argsort(ppp.T[1])]
        new_seed, new_x_nil = interpolate_seed(new_seed, new_t, norm_rsde_nil, denorm_rsde_nil, ome)
        new_seed, new_x_prime = interpolate_seed(new_seed, new_t, norm_rsde_prime, denorm_rsde_prime, ome)
        new_phi_nil = eval_disc_nil(new_x_nil)
        new_phi_prime = eval_disc_prime(new_x_prime)
        if upside:
            gap = np.where(new_phi_prime > new_phi_nil, new_phi_prime - new_phi_nil, np.zeros(len(new_t)))
        else:
            gap = np.where(new_phi_prime < new_phi_nil, new_phi_nil - new_phi_prime, np.zeros(len(new_t)))
        assert not np.any(gap > intensity)
        if np.any(gap > crit_gap):
            return False, new_seed
    return True, new_seed


def est_poisson_coin(seed: SeedSkeleton,
                     norm_rsde: Callable[[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, np.ndarray]],
                     denorm_rsde: Callable[[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, np.ndarray]],
                     eval_disc: Callable[[np.ndarray], np.ndarray],
                     eval_bounds_disc: Callable[[float, float], Tuple[float, float, float]],
                     ome: np.random.Generator) -> (float, float, SeedSkeleton):
    """
    :param seed:
    :param norm_rsde:
    :param denorm_rsde:
    :param eval_disc:
    :param eval_bounds_disc:
    :param ome:
    :return:
    """

    t, le_phi, ue_phi = bound_disc_path(seed, norm_rsde, denorm_rsde, eval_bounds_disc, 1)
    st_lb_disc = -np.diff(t) @ le_phi
    intensity = np.max(ue_phi - le_phi)
    if np.isinf(intensity):
        return -np.inf, st_lb_disc, seed

    fin_t, = denorm_rsde(np.ones(1), None)[0]
    new_t, = sample_ppp(intensity, (fin_t,), ome).T
    if len(new_t) > 0:
        new_seed, new_x = interpolate_seed(seed, new_t, norm_rsde, denorm_rsde, ome)
        new_phi = eval_disc(new_x)
        return np.sum(np.log(ue_phi - new_phi)) - len(new_phi) * np.log(intensity), st_lb_disc, new_seed
    return 0, st_lb_disc, seed


def integrate_disc_bound(seed: SeedSkeleton,
                         norm_rsde: Callable[[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, np.ndarray]],
                         denorm_rsde: Callable[[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, np.ndarray]],
                         eval_bounds_disc: Callable[[float, float], Tuple[float, float, float]]) -> float:
    """
    :param seed:
    :param norm_rsde:
    :param denorm_rsde:
    :param eval_bounds_disc:
    :return:
    """

    t, le_disc, _ = bound_disc_path(seed, norm_rsde, denorm_rsde, eval_bounds_disc)
    return -np.diff(t) @ le_disc


def interpolate_seed(seed: SeedSkeleton, new_t: np.ndarray,
                     norm_rsde: Callable[[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, np.ndarray]],
                     denorm_rsde: Callable[[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, np.ndarray]],
                     ome: np.random.Generator) -> (SeedSkeleton, np.ndarray):
    """
    :param seed:
    :param new_t:
    :param norm_rsde:
    :param denorm_rsde:
    :param ome:
    :return:
    """

    skel = skeleton.Skeleton(*seed)
    new_s, _ = norm_rsde(new_t, None)
    new_skel, new_z = skeleton.interpolate_skeleton(skel, new_s, ome)
    _, new_x = denorm_rsde(new_s, new_z)

    return SeedSkeleton(*new_skel), new_x


def bound_disc_path(seed: SeedSkeleton,
                    norm_rsde: Callable[[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, np.ndarray]],
                    denorm_rsde: Callable[[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, np.ndarray]],
                    eval_bounds_disc: Callable[[float, float], Tuple[float, float, float]],
                    t: Union[np.ndarray, int] = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param seed:
    :param norm_rsde:
    :param denorm_rsde:
    :param eval_bounds_disc:
    :param t:
    :return:
    """

    if seed.tight_z < seed.loose_z:
        le_z, ue_z = seed.tight_z, seed.loose_z
    else:
        ue_z, le_z = seed.tight_z, seed.loose_z
    
    if isinstance(t, int):
        s = np.linspace(0, 1, t + 1)
    else:
        s, _ = norm_rsde(t, None)

    t, le_x = denorm_rsde(s, np.repeat(le_z, len(s)))
    _, ue_x = denorm_rsde(s, np.repeat(ue_z, len(s)))

    le_x = np.min([le_x[1:], le_x[:-1]], 0)
    ue_x = np.max([ue_x[1:], ue_x[:-1]], 0)
    _, le_phi, ue_phi = zip(*[eval_bounds_disc(le_x_, ue_x_) for le_x_, ue_x_ in zip(le_x, ue_x)])

    return t, np.array(le_phi), np.array(ue_phi)
