from typing import Iterator

import numpy as np

from ctbayes.mjplib import skeleton as mjp_skel


def seq_update_normal(obs: np.ndarray, n: float, mean: np.ndarray, cov: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    :param obs:
    :param n:
    :param mean:
    :param cov:
    :return:

    >>> y = np.random.standard_normal((10, 2))
    >>> mean, cov = np.mean(y[:2], 0), np.cov(y[:2].T)
    >>> for i in range(2, 10): mean, cov = seq_update_normal(y[i], i, mean, cov)
    >>> np.allclose(mean, np.mean(y, 0))
    True
    >>> np.allclose(cov, np.cov(y.T))
    True
    """

    dev = obs - mean
    mean = mean + dev / n
    cov = cov + (np.outer(dev, dev) - cov) / n

    return mean, cov


def transform_to_reals(x: float, lb: float, ub: float) -> (float, float):

    if np.isinf(lb) and np.isinf(ub) or (not np.isinf(lb) and not np.isinf(ub)):
        return x, 0
    if np.isinf(lb):
        return np.log(ub - x), -np.log(ub - x)
    if np.isinf(ub):
        return np.log(x - lb), -np.log(x - lb)


def transform_to_const(y: float, lb: float, ub: float) -> (float, float):

    if np.isinf(lb) and np.isinf(ub) or (not np.isinf(lb) and not np.isinf(ub)):
        return y, 0
    if np.isinf(lb):
        return ub - np.exp(y), -y
    if np.isinf(ub):
        return lb + np.exp(y), -y


class MyopicMvRwSampler(object):

    def __init__(self, init_state: np.ndarray, bounds: (np.ndarray, np.ndarray), opt_prob: float = .234,
                 adapt_decay: float = .66, air: int = 1):

        self.state = init_state
        self.bounds = bounds
        self.opt_prob = opt_prob
        self.adapt_decay = adapt_decay
        self.air = air
        self.log_prop_scale = [-1.0]
        self.emp_prob = [opt_prob]
        self.adapt_periods = [0, 1]
        self.adapt_mean = np.zeros_like(init_state)
        self.prop_cov = self.adapt_cov = np.identity(len(init_state))

    def propose(self, ome: np.random.Generator) -> (np.ndarray, float, float):

        state_real, log_p_backw = zip(*[transform_to_reals(x_, lb_, ub_) for x_, lb_, ub_ in zip(self.state, *self.bounds)])
        prop_real = np.random.multivariate_normal(state_real, self.prop_cov * np.exp(self.log_prop_scale[-1]))
        prop, log_p_forw = zip(*[transform_to_const(y_, lb_, ub_) for y_, lb_, ub_ in zip(prop_real, *self.bounds)])
        return np.array(prop), sum(log_p_forw), sum(log_p_backw)

    def adapt(self, sample: float, prob: float):

        self.state = sample
        sample_real, _ = zip(*[transform_to_reals(x_, lb_, ub_) for x_, lb_, ub_ in zip(sample, *self.bounds)])
        self.adapt_mean, self.adapt_cov = seq_update_normal(sample_real, 1 + len(self.emp_prob), self.adapt_mean, self.adapt_cov)
        self.emp_prob.append(prob)
        if len(self.emp_prob) == self.adapt_periods[-1] + 1:
            learning_rate = 1 / (len(self.adapt_periods) ** self.adapt_decay)
            mean_prob = np.mean(self.emp_prob[self.adapt_periods[-2]:])
            self.log_prop_scale.append(self.log_prop_scale[-1] + learning_rate * (mean_prob - self.opt_prob))
            self.adapt_periods.append(self.adapt_periods[-1] + len(self.adapt_periods) ** self.air)
            self.prop_cov = self.adapt_cov #self.prop_cov + learning_rate * (self.adapt_cov - self.prop_cov)


class MyopicRwSampler(object):

    def __init__(self, init_state: float, bounds: (float, float) = (np.inf, np.inf), opt_prob: float = .234,
                 adapt_decay: float = .66, air: int = 1):

        self.state = init_state
        self.bounds = bounds
        self.opt_prob = opt_prob
        self.adapt_decay = adapt_decay
        self.air = air
        self.log_prop_scale = [-1.0]
        self.emp_prob = [opt_prob]
        self.adapt_periods = [0, 1]

    def propose(self, ome: np.random.Generator) -> (float, float, float):

        state_real, log_p_backw = transform_to_reals(self.state, *self.bounds)
        prop_real = np.random.normal(state_real, np.exp(self.log_prop_scale[-1]))
        prop, log_p_forw = transform_to_const(prop_real, *self.bounds)
        return prop, sum(log_p_forw), sum(log_p_backw)

    def adapt(self, sample: float, prob: float):

        self.state = sample
        self.emp_prob.append(prob)
        if len(self.emp_prob) == self.adapt_periods[-1] + 1:
            learning_rate = 1 / (len(self.adapt_periods) ** self.adapt_decay)
            mean_prob = np.mean(self.emp_prob[self.adapt_periods[-2]:])
            self.log_prop_scale.append(self.log_prop_scale[-1] + learning_rate * (mean_prob - self.opt_prob))
            self.adapt_periods.append(self.adapt_periods[-1] + len(self.adapt_periods) ** self.air)


class MyopicMjpSampler(object):

    def __init__(self, init_state: mjp_skel.Skeleton, opt_prob: float = .234, adapt_decay: float = .66, air: int = 1):

        self.state = init_state
        self.opt_prob = opt_prob
        self.adapt_decay = adapt_decay
        self.air = air
        self.log_prop_scale = [0.0]
        self.emp_prob = [opt_prob]
        self.adapt_periods = [0, 1]

    def propose(self, lam: np.ndarray, t: np.ndarray, ome: np.random.Generator) -> (mjp_skel.Skeleton, np.ndarray):

        p_cond = 1 / (1 + np.exp(self.log_prop_scale[-1]))
        t_cond = t[1:-1][ome.uniform(size=len(t) - 2) < p_cond]
        prop = mjp_skel.paste_partition(mjp_skel.mutate_partition(mjp_skel.partition_skeleton(self.state, t_cond), lam, ome))
        return (prop, t_cond)

    def adapt(self, sample: mjp_skel.Skeleton, prob: float):

        self.state = sample
        self.emp_prob.append(prob)
        if len(self.emp_prob) == self.adapt_periods[-1] + 1:
            learning_rate = 1 / (len(self.adapt_periods) ** self.adapt_decay)
            self.log_prop_scale.append(self.log_prop_scale[-1] + learning_rate * (prob - self.opt_prob))
            self.adapt_periods.append(self.adapt_periods[-1] + len(self.adapt_periods) ** self.air)
