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


class MyopicRwSampler(object):

    def __init__(self,
                 init_state: float,
                 init_rate: float,
                 bounds: (float, float) = (np.inf, np.inf),
                 opt_prob: float = .234,
                 adapt_decay: float = .75,
                 air: int = 1):

        self.prop_mean = self.running_mean = self.state = init_state
        self.bounds = bounds
        self.opt_prob = opt_prob
        self.adapt_decay = adapt_decay
        self.air = air
        self.log_prop_scale = [init_rate]
        self.emp_prob = [1]
        self.adapt_periods = [0, 1]

    def propose(self, ome: np.random.Generator) -> (float, float):

        # if self.bounds[0] == 0:
        #     sample = np.random.lognormal(np.log(self.state), np.exp(self.log_prop_scale[-1]))
        #     return sample, np.log(self.state) - np.log(sample)
        return ome.normal(self.state, np.exp(self.log_prop_scale[-1])), 0

    def adapt(self, sample: float, prob: float):

        self.state = sample
        self.emp_prob.append(prob)
        if len(self.emp_prob) == self.adapt_periods[-1] + 1:
            learning_rate = 1 / (len(self.adapt_periods) ** self.adapt_decay)
            mean_prob = np.mean(self.emp_prob[self.adapt_periods[-2]:])
            self.log_prop_scale.append(self.log_prop_scale[-1] + learning_rate * (mean_prob - self.opt_prob))
            self.adapt_periods.append(self.adapt_periods[-1] + len(self.adapt_periods) ** self.air)


class MyopicMjpSampler(object):

    def __init__(self, init_state: mjp_skel.Skeleton, opt_prob: float = .234, adapt_decay: float = .75, air: int = 1):

        self.state = init_state
        self.opt_prob = opt_prob
        self.adapt_decay = adapt_decay
        self.air = air
        self.prop_log_scale = [0]
        self.emp_prob = [1]
        self.adapt_periods = [0, 1]

    def propose(self, lam: np.ndarray, t: np.ndarray, ome: np.random.Generator) -> (mjp_skel.Skeleton, np.ndarray):

        p_cond = 1 / (1 + np.exp(self.prop_log_scale[-1]))
        t_cond = t[np.random.uniform(size=len(t)) < p_cond]
        prop = mjp_skel.paste_partition(mjp_skel.mutate_partition(mjp_skel.partition_skeleton(self.state, t_cond), lam, ome))
        return (prop, t_cond)

    def adapt(self, prob: float):

        self.emp_prob.append(prob)
        if len(self.emp_prob) == self.adapt_periods[-1] + 1:
            learning_rate = 1 / (len(self.adapt_periods) ** self.adapt_decay)
            self.prop_log_scale.append(self.prop_log_scale[-1] + learning_rate * (prob - self.opt_prob))
            self.adapt_periods.append(self.adapt_periods[-1] + len(self.adapt_periods) ** self.air)
