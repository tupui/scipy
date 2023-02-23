from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Literal, TYPE_CHECKING
)

import numpy as np
from numpy.testing import suppress_warnings

from scipy import stats


if TYPE_CHECKING:
    import numpy.typing as npt


__all__ = [
    'dunnett'
]


@dataclass
class DunnettResult:
    statistic: np.ndarray
    pvalue: np.ndarray


def dunnett(
    *observations: npt.ArrayLike, control: npt.ArrayLike,
    alternative: Literal['two-sided', 'less', 'greater'] = "two-sided"
) -> DunnettResult:
    """Dunnett's test.

    Parameters
    ----------
    observations1, observations2, ... : 1D array_like
        The sample measurements for each experiment group.
    control : 1D array_like
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
    rho, df = rho_df_dunnett(observations=observations, control=control)

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


def rho_df_dunnett(
    observations: npt.ArrayLike, control: npt.ArrayLike
) -> tuple[np.ndarray, int]:
    """Specific parameters for Dunnett's test.

    Covariance matrix depends on the number of observations in each group:

    - All groups are equals (including the control), ``rho_ij=0.5`` except for
      the diagonal which is 1.
    - All groups but the control are equal, balanced design.
    - Groups are not equal, unbalanced design.

    Degree of freedom is the number of observations minus the number of groups
    including the control.
    """
    n_n_obs = np.array([len(obs_) for obs_ in observations])

    # From Dunnett1955 p. 1100 d.f. = (sum N)-(p+1)
    n_obs = n_n_obs.sum()
    n_control = len(control)
    n = n_obs + n_control
    n_groups = len(observations)
    df = n - n_groups - 1

    # rho_ij = 1/sqrt((N0/Ni+1)(N0/Nj+1))
    rho = n_control/n_n_obs + 1
    rho = 1/np.sqrt(rho[:, None] * rho[None, :])
    np.fill_diagonal(rho, 1)

    return rho, df


def pvalue_dunnett(
    rho: np.ndarray, df: int, statistic: np.ndarray,
    alternative: Literal['two-sided', 'less', 'greater']
) -> np.ndarray:
    """pvalue from Dunnett critical value.

    Critical values come from the multivariate student-t distribution.
    """
    statistic = statistic.reshape(-1, 1)

    mvt = stats.multivariate_t(shape=rho, df=df)
    if alternative == "two-sided":
        pvalue = 1 - mvt.cdf(statistic, lower_limit=-statistic)
    else:
        pvalue = 1 - mvt.cdf(statistic)

    return pvalue
