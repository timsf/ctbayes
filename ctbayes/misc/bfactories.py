from typing import Any, Generator, Iterator, Tuple
from itertools import count, repeat

import numpy as np

from ctbayes.misc.exceptions import BudgetConstraintError


def sample_twocoin(w1: float,
                   w2: float,
                   c1: Iterator[Tuple[bool, Any]],
                   c2: Iterator[Tuple[bool, Any]],
                   pr_portkey: float = 0,
                   prop0: Any = None,
                   ome: np.random.Generator = np.random.default_rng(),
                   max_props: int = int(1e5)) -> (bool, bool, Any):
    """Sample from a coin with probability w1 * p1 / (w1 * p1 + w2 * p2) using two coins with probabilities p1 and p2

    :param w1:
    :param w2:
    :param c1: generator yielding heads with probability p1
    :param c2: generator yielding heads with probability p2
    :param pr_portkey: probability of escaping with coin 2
    :param prop0:
    :param ome:
    :param max_props: maximum number of proposals
    :return: coin flip with probability of heads w1 * p1 / (w1 * p1 + w2 * p2)

    >>> # draw iid samples
    >>> nsamples = int(1e4)
    >>> w1, w2 = np.random.uniform(size=2)
    >>> p1, p2 = np.random.uniform(size=2)
    >>> c1 = ((np.random.uniform() < p1, None) for _ in count())
    >>> c2 = ((np.random.uniform() < p2, None) for _ in count())
    >>> sample = [sample_twocoin(np.log(w1), np.log(w2), c1, c2)[0] for _ in range(nsamples)]

    >>> # test
    >>> from scipy.stats import binom_test
    >>> alpha = 1e-2
    >>> alpha < binom_test(np.sum(sample), len(sample), w1 * p1 / (w1 * p1 + w2 * p2))
    True
    """

    v1 = w1 - np.logaddexp(w1, w2)

    for _ in range(max_props):

        # attempt proposal from coin 1
        if np.log(ome.uniform()) < v1:
            success, prop = next(c1)
            if success:
                return True, False, prop

        # attempt proposal from coin 2
        else:
            success, prop = next(c2)
            if success:
                return False, False, prop

        # attempt escape or abort pathological proposal
        if ome.uniform() < pr_portkey:
            return False, True, prop0

    # prevent infinite loop
    else:
        raise BudgetConstraintError


def sample_twocoin_joint(w1: float,
                         w2: float,
                         c: Generator[Tuple[bool, Any], bool, None],
                         pr_portkey: float = 0,
                         ome: np.random.Generator = np.random.default_rng(),
                         max_props: int = int(1e5)) -> (bool, bool, Any):
    """Sample from a coin with probability w1 * p1 / (w1 * p1 + w2 * p2) using two coins with probabilities p1 and p2

    :param w1:
    :param w2:
    :param c: generator yielding heads with probability p1/p2
    :param pr_portkey: probability of escaping with coin 2
    :param max_props: maximum number of proposals
    :return: coin flip with probability of heads w1 * p1 / (w1 * p1 + w2 * p2)

    >>> # draw iid samples
    >>> nsamples = int(1e4)
    >>> w1, w2 = np.random.uniform(size=2)
    >>> p1, p2 = np.random.uniform(size=2)
    >>> c = None
    >>> sample = [sample_twocoin(np.log(w1), np.log(w2), c)[0] for _ in range(nsamples)]

    >>> # test
    >>> from scipy.stats import binom_test
    >>> alpha = 1e-2
    >>> alpha < binom_test(np.sum(sample), len(sample), w1 * p1 / (w1 * p1 + w2 * p2))
    True
    """

    next(c)
    v1 = w1 - np.logaddexp(w1, w2)

    for _ in range(max_props):

        # attempt proposal from coin 1
        if np.log(ome.uniform()) < v1:
            success, prop = c.send(True)
            if success:
                return True, False, prop

        # attempt proposal from coin 2
        else:
            success, prop = c.send(False)
            if success:
                return False, False, prop

        # attempt escape or abort pathological proposal
        if ome.uniform() < pr_portkey:
            return False, True, prop

    # prevent infinite loop
    else:
        raise BudgetConstraintError
