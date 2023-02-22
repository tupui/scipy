from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Callable, Literal, Protocol, TYPE_CHECKING
)

import numpy as np

from scipy import stats


if TYPE_CHECKING:
    import numpy.typing as npt
    from scipy._lib._util import DecimalNumber, IntNumber, SeedType


__all__ = [
    'dunnett'
]


@dataclass
class DunnettResult:
    statistic: np.ndarray
    pvalue: np.ndarray


def dunnett(observations, control, *, alternative="two-sided"):
    """Dunnett's test.

    Parameters
    ----------
    observations : array_like, n-D
        The sample measurements for each experiment group.
        With `observations` of shape ``(k, n)``, with ``k`` the number of
        groups and ``n`` the number of sample.
    control : array_like, 1D
        The sample measurements for the control group.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

        * 'two-sided': the means of the distributions underlying the samples
          are unequal.
        * 'less': the mean of the distribution underlying the first sample
          is less than the mean of the distribution underlying the second
          sample.
        * 'greater': the mean of the distribution underlying the first
          sample is greater than the mean of the distribution underlying
          the second sample.

    Returns
    -------
    res : DunnettResult
        An object containing attributes:

        statistic : scalar or ndarray
            The z-score for the test.  For 1-D inputs a scalar is
            returned.
        pvalue : scalar ndarray
            The p-value for the test.

    See Also
    --------
    tukey_hsd : performs pairwise comparison of means.

    Notes
    -----
    Dunnett's test [1]_ compares the mean of multiple experiment groups against
    a control group. `tukey_hsd` instead, performs pairwise comparison of
    means. It means Dunnett's test performs less tests making it more powerful.
    It should be preferred when there is control group.

    The use of this test relies on several assumptions.

    1. The observations are independent within and among groups.
    2. The observations within each group are normally distributed.
    3. The distributions from which the samples are drawn have the same finite
       variance.

    References
    ----------
    .. [1] Charles W. Dunnett. "A Multiple Comparison Procedure for Comparing
       Several Treatments with a Control."
       Journal of the American Statistical Association, 50:272, 1096-1121,
       :doi:`10.1080/01621459.1955.10501294`, 1955.
    .. [2] K.S. Kwong, W. Liu. "Calculation of critical values for Dunnett
       and Tamhane’s step-up multiple test procedure."
       Statistics and Probability Letters, 49, 411-416,
       :doi:`10.1016/S0167-7152(00)00076-6`, 2000.

    Examples
    --------
    ...
    """
    n_obs = np.prod(observations.shape)
    n_control = control.shape[0]

    # should we support different sizes? more complex logic like tukey_hsd
    # then we need to adjust sigma as not 0.5 for non diag elements
    # also observations would need to be passed separately
    if observations.shape[1] != n_control:
        msg = (
            "Dunnett's test assume the same number "
            "of samples in the control group and each observation group"
        )
        raise ValueError(msg)

    n_groups = observations.shape[0]
    # From Dunnett1955 p. 1100 d.f. = (sum N)-(p+1)
    df = n_obs + n_control - n_groups - 1

    ttest = stats.ttest_ind(
        observations, control, axis=-1, alternative=alternative
    )
    statistic = ttest.statistic

    pvalue = pvalue_dunnett(
        n_groups=n_groups, df=df, statistic=statistic, alternative=alternative
    )

    return DunnettResult(statistic=statistic, pvalue=pvalue)


def pvalue_dunnett(n_groups, df, statistic, alternative):
    """pvalue from Dunnett critical value.

    Critical values come from the multivariate student-t distribution.
    """
    rho = np.full((n_groups, n_groups), 0.5)
    # from "Calculation of critical values for Dunnett and Tamhane’s step-up
    # multiple test procedure" p. 412
    np.fill_diagonal(rho, 1)
    statistic_ = statistic.reshape(-1, 1)
    pvalue = 1 - stats.multivariate_t(shape=rho, df=df).cdf(statistic_)

    if alternative == "two-sided":
        pvalue *= 2

    return pvalue
