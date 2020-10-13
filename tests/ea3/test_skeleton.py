import unittest

import numpy as np
from scipy.stats import kstest, norm

from ctbayes.ea3lib import skeleton
from ctbayes.sdelib.moments import moments_brownbr


class InterpolationTest(unittest.TestCase):

    def setUp(self, seed=None):

        self.ome = np.random.default_rng(seed)

        self.fin_t = self.ome.uniform()
        self.init_x = self.ome.normal()
        self.fin_x = self.ome.normal(self.init_x, np.sqrt(self.fin_t))

        self.t = np.array([0, self.fin_t])
        self.xt = np.array([self.init_x, self.fin_x])

    def tearDown(self):

        print('Test conducted with the following seed: {0}'.format(self.ome._bit_generator._seed_seq.entropy))

    def test_interpolation(self, nsamples=int(1e3), ntimes=int(1e1), alpha=1e-2):

        skels = [skeleton.sample_raw_skeleton(self.fin_t, self.init_x, self.fin_x, (-np.inf, np.inf), self.ome)
                 for _ in range(nsamples)]
        test_t = np.sort(self.ome.uniform(high=self.fin_t, size=ntimes))
        test_xt = np.array([skeleton.interpolate_skeleton(x, test_t, self.ome)[1] for x in skels])

        true_mean, true_cov = moments_brownbr(test_t, self.fin_t, *self.xt)
        sample = np.linalg.solve(np.linalg.cholesky(true_cov), (test_xt - true_mean).T).flatten()

        self.assertLess(alpha, kstest(sample, norm().cdf)[1])
