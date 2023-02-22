import numpy as np
import pytest
from numpy.testing import assert_allclose

from scipy.stats._multicomp import pvalue_dunnett


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
            (4, 60, [2.87, 2.87], [0.01, 0.01], "one-sided"),
            # Tables 2a and 2b pages 1119-1120
            (1, 10, 2.23, 0.05, "two-sided"),  # two-sided
            (3, 10, 2.81, 0.05, "two-sided"),
            (2, 30, 2.32, 0.05, "two-sided"),
            (5, 30, 2.72, 0.05, "two-sided"),
            (4, 12, 3.76, 0.01, "two-sided"),
            (7, 12, 4.08, 0.01, "two-sided"),
            (2, 60, 2.90, 0.01, "two-sided"),
            (4, 60, 3.14, 0.01, "two-sided"),
            (4, 60, [3.14, 2.90], [0.01, 0.01], "one-sided"),
            # From Kwong2000
            (9, 30, 2.856, 0.05, "two-sided"),
            (17, 20, 3.162, 0.05, "two-sided"),
        ],
    )
    def test_critical_values(
        self, n_groups, df, statistic, pvalue, alternative
    ):
        statistic = np.asarray(statistic)
        res = pvalue_dunnett(
            n_groups=n_groups, df=df, statistic=statistic,
            alternative=alternative
        )
        assert_allclose(res, pvalue, atol=1e-2)
