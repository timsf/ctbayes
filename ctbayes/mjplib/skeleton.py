import itertools as it
from typing import Iterator, Optional, List, NamedTuple, Tuple

import numpy as np
from scipy.linalg import expm
from scipy.special import loggamma


class Skeleton(NamedTuple):
    t: np.ndarray
    xt: np.ndarray
    fin_t: float

    def __call__(self, t: np.ndarray) -> np.ndarray:
        if np.any(t > self.fin_t):
            raise ValueError
        return self.xt[np.searchsorted(self.t, t, side='right') - 1]

    def __eq__(self, skel) -> bool:
        return all([len(self.t) == len(skel.t) and np.all(self.t == skel.t),
                    len(self.xt) == len(skel.xt) and np.all(self.xt == skel.xt),
                    self.fin_t == skel.fin_t])


Partition = List[Skeleton]


def sample_batch_ppp(intensity: float, bounds: Tuple[float, ...] = (1, 1), batch_size: int = 2,
                     ome: np.random.Generator = np.random.default_rng()) -> Iterator[np.ndarray]:
    """Sample from a Poisson point process in batches.

    >>> # generate fixture
    >>> gen = int(1e4)
    >>> bound = np.random.lognormal(size=2)

    >>> # draw iid samples
    >>> y = np.vstack([ppp for ppp in sample_batch_ppp(gen, tuple(bound))])

    >>> # test sampling distribution
    >>> from scipy.stats import kstest, uniform
    >>> alpha = 1e-2
    >>> sample = (y / bound).flatten()
    >>> dist = uniform()
    >>> alpha < kstest(sample, dist.cdf)[1]
    True
    """

    assert 0 < intensity
    assert 0 < min(bounds)
    assert 0 < batch_size

    n_points = ome.poisson(intensity * np.prod(bounds))
    batch_sizes = it.repeat(batch_size, int(n_points // batch_size))
    if n_points % batch_size != 0:
        batch_sizes = it.chain(batch_sizes, [n_points % batch_size])

    ub = 0
    for i in batch_sizes:
        lb, ub = ub, ub + bounds[0] * ome.beta(i, n_points + 1 - i)
        if i == 1:
            u0 = np.array([ub])
        else:
            u0 = np.hstack([np.sort(ome.uniform(lb, ub, i - 1)), ub])
        u = ome.uniform(np.zeros(len(bounds[1:])), bounds[1:], (i, len(bounds[1:])))
        yield np.vstack([u0, u.T]).T


def sample_ppp(intensity: float, bound: Tuple[float, ...] = (1, 1), ome: np.random.Generator = np.random.default_rng()
               ) -> np.ndarray:
    """Sample from a Poisson point process on the plane.

    :param intensity:
    :param bound:
    :param ome:
    :return:

    >>> # generate fixture
    >>> gen = int(1e4)
    >>> bound = np.random.lognormal(size=2)

    >>> # draw iid samples
    >>> y = sample_ppp(gen, tuple(bound))

    >>> # test sampling distribution
    >>> from scipy.stats import kstest, uniform
    >>> alpha = 1e-2
    >>> sample = (y / bound).flatten()
    >>> dist = uniform()
    >>> alpha < kstest(sample, dist.cdf)[1]
    True
    """

    assert 0 < intensity
    assert 0 < min(bound)

    n_points = ome.poisson(intensity * np.prod(bound))
    locations = ome.uniform(np.zeros(len(bound)), bound, (n_points, len(bound)))

    return locations[np.argsort(locations[:, 0])]


def sample_forwards(fin_t: float, init_x: Optional[int], gen: np.ndarray,
                    ome: np.random.Generator = np.random.default_rng()) -> Skeleton:
    """
    :param fin_t:
    :param init_x:
    :param gen:
    :param ome:
    :return:

    >>> # generate fixture
    >>> dim = 3
    >>> fin_t = 1e5
    >>> gen = sample_trial_generator(dim)
    >>> stat = get_stat(gen)

    >>> # test stationary distribution
    >>> skel = sample_forwards(fin_t, None, gen)
    >>> probs = [np.sum(np.ediff1d(skel.t, to_end=(fin_t - skel.t[-1],))[skel.xt == i]) / fin_t for i in range(dim)]
    >>> np.allclose(probs, stat, 1e-2)
    True
    """

    dim = gen.shape[0]
    if init_x is None:
        init_x = ome.choice(dim, 1, p=get_stat(gen)).item()

    t, x = [0], [init_x]
    while t[-1] < fin_t:
        hold = ome.exponential(1 / np.delete(gen[x[-1]], x[-1]))
        move = np.argmin(hold)
        t.append(t[-1] + hold[move])
        x.append(move + int(move >= x[-1]))
    return Skeleton(np.array(t[:-1]), np.array(x[:-1]), fin_t)


def sample_backwards(fin_t: float, fin_x: int, gen: np.ndarray, ome: np.random.Generator = np.random.default_rng()
                     ) -> Skeleton:
    """
    :param fin_t:
    :param fin_x:
    :param gen:
    :param ome:
    :return:

    >>> # generate fixture
    >>> dim = 3
    >>> fin_t = 1e5
    >>> gen = sample_trial_generator(dim)
    >>> stat = get_stat(gen)
    >>> fin_x = np.random.choice(dim, 1, p=get_stat(gen)).item()

    >>> # test stationary distribution
    >>> skel = sample_backwards(fin_t, fin_x, gen)
    >>> probs = [np.sum(np.ediff1d(skel.t, to_end=(fin_t - skel.t[-1],))[skel.xt == i]) / fin_t for i in range(dim)]
    >>> np.allclose(probs, stat, 1e-2)
    True
    """

    ftrans = get_trans(fin_t, gen)
    btrans = ftrans[:, fin_x] / np.sum(ftrans[:, fin_x])
    init_x = ome.choice(np.arange(len(btrans)), 1, p=btrans).item()

    return sample_bridge(fin_t, init_x, fin_x, gen, ome)


def sample_bridge(fin_t: float, init_x: int, fin_x: int, gen: np.ndarray,
                  ome: np.random.Generator = np.random.default_rng()) -> Skeleton:
    """
    :param fin_t:
    :param init_x:
    :param fin_x:
    :param gen:
    :param ome:
    :return:

    >>> # generate fixture
    >>> dim = 3
    >>> fin_t = 1e5
    >>> gen = sample_trial_generator(dim)
    >>> stat = get_stat(gen)
    >>> init_x, fin_x = sample_forwards(fin_t, None, gen)[1][[0, -1]]

    >>> # test stationary distribution
    >>> skel = sample_bridge(fin_t, init_x, fin_x, gen)
    >>> probs = [np.sum(np.ediff1d(skel.t, to_end=(fin_t - skel.t[-1],))[skel.xt == i]) / fin_t for i in range(dim)]
    >>> np.allclose(probs, stat, 1e-2)
    True
    """

    def sample_n_trans() -> (int, List[np.ndarray]):
        u = np.log(ome.uniform()) + np.log(ctrans[init_x, fin_x]) + rate_bound * fin_t
        pow_dtrans = [np.identity(dim)]
        p = [0 if init_x == fin_x else -np.inf]
        n = 0
        while p[-1] < u:
            n += 1
            pow_dtrans.append(pow_dtrans[-1] @ dtrans)
            if pow_dtrans[-1][init_x, fin_x] == 0:
                p.append(p[-1])
            else:
                p.append(np.logaddexp(p[-1], n * np.log(rate_bound * fin_t) - loggamma(n + 1) + np.log(pow_dtrans[-1][init_x, fin_x])))
        return n, pow_dtrans

    dim = gen.shape[0]
    rate_bound = -np.min(np.diag(gen))
    dtrans = np.identity(dim) + gen / rate_bound
    ctrans = get_trans(fin_t, gen)
    n_trans, pow_dtrans = sample_n_trans()

    if n_trans == 0 or (n_trans == 1 and init_x == fin_x):
        return Skeleton(np.zeros(1), np.array([init_x]), fin_t)

    x = [init_x]
    for i in range(1, n_trans):
        q = dtrans[x[i - 1]] * pow_dtrans[n_trans - i][:, fin_x] / pow_dtrans[n_trans - i + 1][x[i - 1], fin_x]
        x.append(ome.choice(np.arange(dim), 1, p=q).item())
    x.append(fin_x)

    t = np.hstack([0, np.sort(ome.uniform(0, fin_t, n_trans))])
    x = np.array(x)
    is_virtual = np.hstack([False, x[1:] == x[:-1]])

    return Skeleton(t[~is_virtual], x[~is_virtual], fin_t)


def sample_partition(skel: Skeleton, intensity: float = 1, ome: np.random.Generator = np.random.default_rng()
                     ) -> (Partition, np.ndarray):
    """
    :param skel:
    :param intensity:
    :param ome:
    :return:

    >>> # generate fixture
    >>> dim = 3
    >>> fin_t = 1e2
    >>> gen = sample_trial_generator(dim)
    >>> stat = get_stat(gen)
    >>> skel = sample_forwards(fin_t, None, gen)

    >>> # test
    >>> skel2 = paste_partition(sample_partition(skel)[0])
    >>> np.all(skel.t == skel2.t), np.all(skel.xt == skel2.xt), skel.fin_t == skel2.fin_t
    (True, True, True)
    """

    n_breaks = ome.poisson(intensity * skel.fin_t)
    new_t = np.linspace(0, skel.fin_t, n_breaks + 2)

    return partition_skeleton(skel, new_t[1:-1]), new_t[1:-1]


def paste_partition(partition: List[Skeleton]) -> Skeleton:
    """
    :param partition:
    :return:

    >>> # generate fixture
    >>> dim = 3
    >>> fin_t = 1e2
    >>> gen = sample_trial_generator(dim)
    >>> stat = get_stat(gen)
    >>> skel = sample_forwards(fin_t, None, gen)

    >>> # test
    >>> skel2 = paste_partition(sample_partition(skel)[0])
    >>> np.all(skel.t == skel2.t), np.all(skel.xt == skel2.xt), skel.fin_t == skel2.fin_t
    (True, True, True)
    """

    partition = [skel for skel in partition if skel.fin_t != 0]
    breaks = np.hstack([0, np.cumsum([skel.fin_t for skel in partition])])
    t, x = [np.hstack(a) for a in zip(*[(init_t + skel.t, skel.xt) for init_t, skel in zip(breaks[:-1], partition)])]
    is_virtual = np.hstack([False, x[1:] == x[:-1]])

    return Skeleton(t[~is_virtual], x[~is_virtual], breaks[-1])


def mutate_partition(partition: List[Skeleton], gen: np.ndarray, ome: np.random.Generator = np.random.default_rng()
                     ) -> Partition:
    """
    :param partition:
    :param gen:
    :param ome:
    :return:

    >>> # generate fixture
    >>> dim = 3
    >>> fin_t = 1e5
    >>> gen = sample_trial_generator(dim)
    >>> stat = get_stat(gen)
    >>> skel = sample_forwards(fin_t, None, gen)

    >>> # test stationary distribution
    >>> skel = paste_partition(mutate_partition(sample_partition(skel)[0], gen))
    >>> probs = [np.sum(np.ediff1d(skel.t, to_end=(fin_t - skel.t[-1],))[skel.xt == i]) / fin_t for i in range(dim)]
    >>> np.allclose(probs, stat, 1e-2)
    True
    """

    if len(partition) > 1:
        init_skel = sample_backwards(partition[0].fin_t, partition[0].xt[-1], gen, ome)
        fin_skel = sample_forwards(partition[-1].fin_t, partition[-1].xt[0], gen, ome)
        new_partition = [init_skel] \
                        + [sample_bridge(skel.fin_t, skel.xt[0], skel.xt[-1], gen, ome) for skel in partition[1:-1]] \
                        + [fin_skel]
    else:
        new_partition = [sample_forwards(partition[0].fin_t, None, gen, ome)]

    return new_partition


def partition_skeleton(skel: Skeleton, new_t: np.ndarray) -> Partition:
    """
    :param skel:
    :param new_t:

    >>> # generate fixture
    >>> dim = 3
    >>> fin_t = 1e2
    >>> gen = sample_trial_generator(dim)
    >>> stat = get_stat(gen)
    >>> new_t = np.sort(np.random.uniform(0, fin_t, 100))
    >>> skel = sample_forwards(fin_t, None, gen)

    >>> # test
    >>> skel2 = paste_partition(partition_skeleton(skel, new_t))
    >>> np.all(skel.t == skel2.t), np.all(skel.xt == skel2.xt), skel.fin_t == skel2.fin_t
    (True, True, True)
    """

    new_t = np.hstack([0, new_t, skel.fin_t])
    slice_right = np.searchsorted(skel.t, new_t, side='right')
    slice_left = np.searchsorted(skel.t, new_t, side='left')

    t = [np.hstack([0, skel.t[i0:i1] - init_t])
         for init_t, i0, i1
         in zip(new_t, slice_right, slice_left[1:])]
    x = [skel.xt[i0 - 1:i0 - 1 + len(t_)] for i0, t_ in zip(slice_right, t)]

    return [Skeleton(t_, x_, fin_t) for t_, x_, fin_t in zip(t, x, np.diff(np.hstack([new_t, skel.fin_t])))]


def repartition_skeleton(partition: Partition, new_t: np.ndarray) -> Partition:
    """
    :param partition:
    :param new_t:
    :return:

    >>> # generate fixture
    >>> dim = 3
    >>> fin_t = 1e2
    >>> gen = sample_trial_generator(dim)
    >>> stat = get_stat(gen)
    >>> new_t = np.sort(np.random.uniform(0, fin_t, 100))
    >>> skel = sample_forwards(fin_t, None, gen)

    >>> # test
    >>> skel2 = paste_partition(repartition_skeleton(partition_skeleton(skel, new_t), new_t))
    >>> np.all(skel.t == skel2.t), np.all(skel.vt == skel2.vt), skel.fin_t == skel2.fin_t
    (True, True, True)
    """

    return partition_skeleton(paste_partition(partition), new_t)


def sample_trial_generator(dim: int, ome: np.random.Generator = np.random.default_rng()) -> np.ndarray:
    """
    :param dim:
    :parma ome:
    :return:
    """

    gen = ome.lognormal(size=(dim, dim))
    np.fill_diagonal(gen, 0)
    np.fill_diagonal(gen, -np.sum(gen, 1))
    return gen


def get_trans(fin_t: float, gen: np.ndarray) -> np.ndarray:
    """
    :param fin_t:
    :param gen:
    :return:
    """

    return expm(fin_t * gen)


def get_stat(gen: np.ndarray) -> np.ndarray:
    """
    :param gen:
    :return:
    """

    return np.linalg.solve(gen.T - 1, -np.ones(gen.shape[0]))


def est_stat(skel: Skeleton, states: np.ndarray) -> np.ndarray:
    """
    :param skel:
    :param states:
    :return:
    """

    return (skel.xt == states[:, np.newaxis]) @ np.ediff1d(np.float64(skel.t), to_end=(skel.fin_t - skel.t[-1],)) / skel.fin_t
