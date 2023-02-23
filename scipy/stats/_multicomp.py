from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Callable, Literal, Protocol, TYPE_CHECKING
)

import numpy as np
from numpy.testing import suppress_warnings

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


def dunnett(*observations, control, alternative="two-sided"):
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
    rho, df = rho_df(observations=observations, control=control)

    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        statistic = np.array([
            stats.ttest_ind(
                obs_, control, alternative=alternative
            ).statistic
            for obs_ in observations
        ])

    pvalue = pvalue_dunnett(
        rho=rho, df=df,
        statistic=statistic, alternative=alternative
    )

    return DunnettResult(statistic=statistic, pvalue=pvalue)


def rho_df(observations, control):
    n_n_obs = np.array([len(obs_) for obs_ in observations])

    # From Dunnett1955 p. 1100 d.f. = (sum N)-(p+1)
    n_obs = n_n_obs.sum()
    n_control = len(control)
    n = n_obs + n_control
    n_groups = len(observations)
    df = n - n_groups - 1

    rho = n_control/n_n_obs + 1
    rho = 1/np.sqrt(rho[:, None] * rho[None, :])
    np.fill_diagonal(rho, 1)

    return rho, df


def pvalue_dunnett(rho, df, statistic, alternative):
    """pvalue from Dunnett critical value.

    Critical values come from the multivariate student-t distribution.
    """
    statistic = np.asarray(statistic).reshape(-1, 1)

    mvt = stats.multivariate_t(shape=rho, df=df)
    if alternative == "two-sided":
        pvalue = 1 - mvt.cdf(statistic, lower_limit=-statistic)
    else:
        pvalue = 1 - mvt.cdf(statistic)

    return pvalue
