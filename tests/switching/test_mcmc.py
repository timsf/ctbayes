import unittest

import numpy as np

from ctbayes.mjplib import skeleton as mjp_skel, inference as mjp_inf
from ctbayes.switching import mcmc
from ctbayes.switching.models import ou

from tests.switching import fix_ou


def sample_posterior(t, vt, init_thi, bounds_thi, hyper_lam, f_loglik, eval_log_prior, ome):

    lam = mjp_inf.get_ev(*hyper_lam)
    param_sampler = [[mcmc.MyopicRwSampler(init_thi_, -1, bounds_thi_)
                      for init_thi_, bounds_thi_ in zip(init_thi, np.array(bounds_thi).T)]
                     for _ in range(lam.shape[0])]
    thi = np.repeat(init_thi[np.newaxis], lam.shape[0], 0)
    y = mjp_skel.sample_forwards(t[-1], None, lam, ome)
    while True:
        thi = sample_params(t, vt, thi, y, f_loglik, eval_log_prior, param_sampler, ome)
        lam = mjp_inf.sample_param(*mjp_inf.update(y, *hyper_lam), ome)
        y = sample_mjp_path(t, vt, thi, lam, y, f_loglik, ome)
        yield thi, lam, y


def sample_mjp_path(t, vt, thi, lam, y_nil, f_loglik, ome):

    n_cond = ome.choice(list(range(len(t) - 1)), 1)
    t_cond = np.sort(ome.choice(t, n_cond, replace=False))
    y_nil_part = mjp_skel.partition_skeleton(y_nil, t_cond)
    y_prime_part = mjp_skel.mutate_partition(mjp_skel.partition_skeleton(y_nil, t_cond), lam, ome)
    break_i = [0] + list(np.where(np.isin(t, t_cond))[0] + 1) + [len(t) - 1]
    y_acc_part = [sample_mjp_section(t[i:j+1] - t[i], vt[i:j+1], thi, lam, y_nil_part[i], y_prime_part[i], f_loglik, ome)
                  for i, j in zip(break_i, break_i[1:])]
    return mjp_skel.paste_partition(y_acc_part)


def sample_mjp_section(t, vt, thi, lam, y_nil, y_prime, f_loglik, ome):

    log_post_odds = f_loglik(t, vt, y_prime, *thi.T) - f_loglik(t, vt, y_nil, *thi.T)
    log_acc_rate = min(0, log_post_odds)
    if np.log(ome.uniform()) < log_acc_rate:
        return y_prime
    else:
        return y_nil


def sample_params(t, vt, thi_nil, y, f_loglik, f_logprior, param_sampler, ome):

    state = ome.integers(0, thi_nil.shape[0])
    sector = ome.integers(0, thi_nil.shape[1])
    prop, log_prop_odds = param_sampler[state][sector].propose(ome)
    thi_prime = thi_nil.copy()
    thi_prime[state, sector] = prop
    log_post_odds = f_loglik(t, vt, y, *thi_prime.T) \
                    - f_loglik(t, vt, y, *thi_nil.T) \
                    + f_logprior(thi_prime) \
                    - f_logprior(thi_nil) \
                    - log_prop_odds
    log_acc_rate = min(0, log_post_odds)
    if np.log(ome.uniform()) < log_acc_rate:
        param_sampler[state][sector].adapt(thi_prime[state, sector], np.exp(log_acc_rate))
        return thi_prime
    else:
        param_sampler[state][sector].adapt(thi_nil[state, sector], np.exp(log_acc_rate))
        return thi_nil


class OuTest(unittest.TestCase):

    def setUp(self, seed=None):

        self.ome = np.random.default_rng(seed)

        self.t, self.vt, self.y, self.thi, self.lam, self.lam0 = fix_ou.generate_fixture(self.ome)
        self.xt = ou.reduce(self.vt)
        self.mod = mcmc.Model(self.t, self.vt, ou.bounds_thi, self.lam0, ou.eval_log_prior, ou.sample_aug, ou.gen_normops,
                              ou.eval_biased_log_lik, ou.eval_disc, ou.eval_bounds_disc, ou.eval_bounds_grad)

    def tearDown(self):

        print('Test conducted with the following seed: {0}'.format(self.ome._bit_generator._seed_seq.entropy))

    def test_integration(self, nsamples=int(1e4), nwarmup=int(1e3), alpha=1e-2):

        test_sampler = ou.sample_posterior(self.t, self.vt, 2, ome=self.ome)
        test_samples = [next(test_sampler) for _ in range(nsamples + nwarmup)][nwarmup::]

        null_sampler = sample_posterior(self.t, self.vt, ou.init_thi, self.mod.bounds_thi, self.mod.hyper_lam,
                                        fix_ou.eval_log_lik, ou.eval_log_prior, self.ome)
        null_samples = [next(null_sampler) for _ in range(nsamples + nwarmup)][nwarmup::]
