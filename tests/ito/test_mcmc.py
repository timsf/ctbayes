import unittest

import numpy as np
from joblib import Parallel
from scipy.stats import norm, epps_singleton_2samp, kstest

from ctbayes.ea3lib import seed
from ctbayes.ito import mcmc
from ctbayes.ito.models import ou
from tests.ito import fix_ou


def sample_posterior(t, vt, init_thi, bounds_thi, eval_log_lik, eval_log_prior, ome):

    param_sampler = [mcmc.MyopicRwSampler(init_thi_, -1, bounds_thi_)
                     for init_thi_, bounds_thi_ in zip(init_thi, np.array(bounds_thi).T)]
    thi = init_thi
    while True:
        thi = sample_parameter(t, vt, thi, eval_log_lik, eval_log_prior, param_sampler, ome)
        yield thi


def sample_parameter(t, vt, thi_nil, eval_loglik, eval_logprior, param_sampler, ome):

    sector = np.random.randint(0, len(thi_nil))
    prop, log_prop_odds = param_sampler[sector].propose(ome)
    thi_prime = np.array([thi_nil[i] if i != sector else prop for i in range(len(thi_nil))])
    log_post_odds = eval_loglik(t, vt, *thi_prime) - eval_loglik(t, vt, *thi_nil) \
                    + eval_logprior(thi_prime) - eval_logprior(thi_nil) \
                    - log_prop_odds
    log_acceptance_rate = min(0, log_post_odds)
    if np.log(ome.uniform()) < log_acceptance_rate:
        param_sampler[sector].adapt(thi_prime[sector], np.exp(log_acceptance_rate))
        return thi_prime
    else:
        param_sampler[sector].adapt(thi_nil[sector], np.exp(log_acceptance_rate))
        return thi_nil


class OuTest(unittest.TestCase):

    def setUp(self, seed=None):

        self.ome = np.random.default_rng(seed)

        self.t, self.vt, self.thi = fix_ou.generate_fixture(self.ome)
        self.xt = ou.reduce(self.thi, self.vt)
        self.mod = mcmc.Model(self.t, self.vt, ou.bounds_thi, ou.eval_log_prior, ou.sample_aug, ou.gen_normops,
                              ou.eval_biased_log_lik, ou.eval_disc, ou.eval_bounds_disc, ou.eval_bounds_grad)
        self.ctrl = mcmc.Controls()

    def tearDown(self):

        print('Test conducted with the following seed: {0}'.format(self.ome._bit_generator._seed_seq.entropy))

    def test_bridge_step(self, nsamples=int(1e4), ntimes=int(1e1), alpha=1e-2):

        z = [next(seed.sample_bridge(ou.bounds_x, *ou.gen_normops(self.thi, self.t, self.vt)[0],
                                     ou.eval_disc(self.thi), ou.eval_bounds_disc(self.thi), self.ome))]
        
        with Parallel(1, 'loky') as pool:
            for _ in range(nsamples):
                z.append(mcmc.update_sde(self.thi, [z[-1]], self.mod, self.ctrl, self.ome, pool)[0])

        test_t = np.sort(self.ome.uniform(high=self.t[-1], size=ntimes))
        test_xt = np.array([seed.interpolate_seed(z_, test_t, *ou.gen_normops(self.thi, self.t, self.vt)[0], self.ome)[1]
                            for z_ in z[::10]])
        test_vt = ou.dereduce(self.thi, test_xt)

        true_mean, true_cov = fix_ou.eval_moments(np.append(test_t, self.t[-1]), *self.vt, *self.thi)
        sample = np.linalg.solve(np.linalg.cholesky(true_cov), (test_vt - true_mean).T).flatten()
        self.assertLess(alpha, kstest(sample, norm().cdf)[1])

    def test_integration(self, nsamples=int(1e4), nwarmup=int(1e3), alpha=1e-2):

        test_sampler = ou.sample_posterior(self.t, self.vt, self.ome)
        test_samples = [next(test_sampler) for _ in range(nsamples + nwarmup)][nwarmup::]

        null_sampler = sample_posterior(self.t, self.vt, ou.init_thi, ou.bounds_thi, fix_ou.eval_log_lik,
                                        ou.eval_log_prior, self.ome)
        null_samples = [next(null_sampler) for _ in range(nsamples + nwarmup)][nwarmup::]

        thi_null = np.array(null_samples[::10])
        thi_test = np.array([sample[0] for sample in test_samples][::10])

        for thi_null_, thi_test_ in zip(thi_null.T, thi_test.T):
            self.assertLess(alpha, epps_singleton_2samp(thi_null_, thi_test_)[1])
