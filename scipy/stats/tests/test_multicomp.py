import numpy as np
import pytest
from numpy.testing import assert_allclose

from scipy import stats
from scipy.stats._multicomp import pvalue_dunnett, DunnettResult


class TestDunnett:

    @pytest.mark.parametrize(
        'n_groups, df, statistic, pvalue, alternative',
        [
            # From Dunnett1995
            # Tables 1a and 1b pages 1117-1118
            (1, 10, 1.81, 0.05, "one-sided"),  # different than two-sided
            (3, 10, 2.34, 0.05, "one-sided"),
            (2, 30, 1.99, 0.05, "one-sided"),
            (5, 30, 2.33, 0.05, "one-sided"),
            (4, 12, 3.32, 0.01, "one-sided"),
            (7, 12, 3.56, 0.01, "one-sided"),
            (2, 60, 2.64, 0.01, "one-sided"),
            (4, 60, 2.87, 0.01, "one-sided"),
            (4, 60, [2.87, 2.21], [0.01, 0.05], "one-sided"),
            # Tables 2a and 2b pages 1119-1120
            (1, 10, 2.23, 0.05, "two-sided"),  # two-sided
            (3, 10, 2.81, 0.05, "two-sided"),
            (2, 30, 2.32, 0.05, "two-sided"),
            (3, 20, 2.57, 0.05, "two-sided"),
            (4, 12, 3.76, 0.01, "two-sided"),
            (7, 12, 4.08, 0.01, "two-sided"),
            (2, 60, 2.90, 0.01, "two-sided"),
            (4, 60, 3.14, 0.01, "two-sided"),
            (4, 60, [3.14, 2.55], [0.01, 0.05], "two-sided"),
            # From Kwong2000
            (9, 30, 2.856, 0.05, "two-sided"),
            (17, 20, 3.162, 0.05, "two-sided"),
        ],
    )
    def test_critical_values(
        self, n_groups, df, statistic, pvalue, alternative
    ):
        rho = np.full((n_groups, n_groups), 0.5)
        np.fill_diagonal(rho, 1)

        statistic = np.array(statistic)
        res = pvalue_dunnett(
            rho=rho, df=df, statistic=statistic,
            alternative=alternative
        )
        assert_allclose(res, pvalue, atol=5e-3)

    def test_imbalanced(self):
        observations = [
            [
                24.0, 27.0, 33.0, 32.0, 28.0, 19.0, 37.0, 31.0, 36.0, 36.0,
                34.0, 38.0, 32.0, 38.0, 32.0
            ],
            [26.0, 24.0, 26.0, 25.0, 29.0, 29.5, 16.5, 36.0, 44.0],
            [25.0, 27.0, 19.0],
            [25.0, 20.0],
            [28.0]
        ]
        control = [
            18.0, 15.0, 18.0, 16.0, 17.0, 15.0, 14.0, 14.0, 14.0, 15.0, 15.0,
            14.0, 15.0, 14.0, 22.0, 18.0, 21.0, 21.0, 10.0, 10.0, 11.0, 9.0,
            25.0, 26.0, 17.5, 16.0, 15.5, 14.5, 22.0, 22.0, 24.0, 22.5, 29.0,
            24.5, 20.0, 18.0, 18.5, 17.5, 26.5, 13.0, 16.5, 13.0, 13.0, 13.0,
            28.0, 27.0, 34.0, 31.0, 29.0, 27.0, 24.0, 23.0, 38.0, 36.0, 25.0,
            38.0, 26.0, 22.0, 36.0, 27.0, 27.0, 32.0, 28.0, 31.0
        ]
        ref = np.array([4.727e-06, 0.022346, 0.97912, 0.99953, 0.86579])

        res = stats.dunnett(*observations, control=control)

        assert isinstance(res, DunnettResult)
        # last value is problematic
        assert_allclose(res.pvalue[:-1], ref[:-1], atol=0.015)