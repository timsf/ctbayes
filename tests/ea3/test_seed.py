import unittest

import numpy as np
from scipy.stats import kstest, norm
from toolz import curry

from ctbayes.ea3lib import seed
from ctbayes.sdelib.moments import moments_brownbr


@curry
def norm_rsde(fin_t, init_x, fin_x, t, xt):

        new_s = t / fin_t
        if xt is not None:
            return new_s, (xt - init_x - new_s * (fin_x - init_x)) / np.sqrt(fin_t)
        return new_s, None


@curry
def denorm_rsde(fin_t, init_x, fin_x, s, zs):

        new_t = s * fin_t
        if zs is not None:
            return new_t, zs * np.sqrt(fin_t) + init_x + s * (fin_x - init_x)
        return new_t, None


class InterpolationTest(unittest.TestCase):

    def setUp(self, seed=None):

        self.ome = np.random.default_rng(seed)

        self.fin_t = self.ome.uniform()
        self.init_x = self.ome.normal()
        self.fin_x = self.ome.normal(self.init_x, np.sqrt(self.fin_t))

        self.t = np.array([0, self.fin_t])
        self.xt = np.array([self.init_x, self.fin_x])
        self.norm_rsde = norm_rsde(self.fin_t, *self.xt)
        self.denorm_rsde = denorm_rsde(self.fin_t, *self.xt)

    def tearDown(self):

        print('Test conducted with the following seed: {0}'.format(self.ome._bit_generator._seed_seq.entropy))

    def test_interpolation(self, nsamples=int(1e3), ntimes=int(1e1), alpha=1e-2):

        seeds = [seed.sample_raw_seed((-np.inf, np.inf), self.norm_rsde, self.denorm_rsde, self.ome)
                 for _ in range(nsamples)]
        test_t = np.sort(self.ome.uniform(high=self.fin_t, size=ntimes))
        test_xt = np.array([seed.interpolate_seed(z, test_t, self.norm_rsde, self.denorm_rsde, self.ome)[1]
                            for z in seeds])

        true_mean, true_cov = moments_brownbr(test_t, self.fin_t, *self.xt)
        sample = np.linalg.solve(np.linalg.cholesky(true_cov), (test_xt - true_mean).T).flatten()

        self.assertLess(alpha, kstest(sample, norm().cdf)[1])
