from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Literal, TYPE_CHECKING
)

import numpy as np
from numpy.testing import suppress_warnings

from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.stats._qmc import check_random_state

if TYPE_CHECKING:
    import numpy.typing as npt
    from scipy._lib._util import DecimalNumber, SeedType


__all__ = [
    'dunnett'
]


@dataclass
class DunnettResult:
    statistic: np.ndarray
    pvalue: np.ndarray
    _rho: np.ndarray
    _df: int
    _std: float

    def allowance(self, confidence_level: DecimalNumber = 0.95) -> float:
        """Allowance.

        It is the quantity to add/substract from the observed difference
        between the means of observed groups and the mean of the control
        group. The result gives confidence limits.
        """
        dist = stats.multivariate_t(shape=self._rho, df=self._df)
        alpha = 1 - confidence_level

        def pvalue_from_stat(statistic):
            # two-sided to have +- bounds
            sf = 1 - dist.cdf(statistic, lower_limit=-statistic)
            return abs(sf - alpha)

        # scipy.stats.sampling methods are not working for this distribution
        res = minimize_scalar(pvalue_from_stat, method='brent', tol=1e-4)
        critical_value = res.x

        allowance = critical_value*self._std*np.sqrt(2/len(self.pvalue))
        return allowance


def dunnett(
    *observations: npt.ArrayLike, control: npt.ArrayLike,
    alternative: Literal['two-sided', 'less', 'greater'] = "two-sided",
    random_state: SeedType = None
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
    random_state : {None, int, `numpy.random.Generator`}, optional
        If `random_state` is an int or None, a new `numpy.random.Generator` is
        created using ``np.random.default_rng(random_state)``.
        If `random_state` is already a ``Generator`` instance, then the
        provided instance is used.

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
    In [1]_, the influence of drugs on blood count measurements on three groups
    of animal is investigated.

    The following table summarizes the results of the experiment in which
    two groups received different drug, and one group acted as a control.
    Blood counts (in millions of cells per cubic millimeter) were recorded::

         Control      Drug A      Drug B
           7.40        9.76        12.80
           8.50        8.80         9.68
           7.20        7.68        12.16
           8.24        9.36         9.20
           9.84                    10.55
           8.32

    >>> import numpy as np
    >>> control = np.array([7.40, 8.50, 7.20, 8.24, 9.84, 8.32])
    >>> drug_a = np.array([9.76, 8.80, 7.68, 9.36])
    >>> drug_b = np.array([12.80, 9.68, 12.16, 9.20, 10.55])

    The `dunnett` statistic is sensitive to the difference in means between
    the samples.

    We would like to see if the means between any of the groups are
    significantly different. First, visually examine a box and whisker plot.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1, 1)
    >>> ax.boxplot([control, drug_a, drug_b])
    >>> ax.set_xticklabels(["Control", "Drug A", "Drug B"])  # doctest: +SKIP
    >>> ax.set_ylabel("mean")  # doctest: +SKIP
    >>> plt.show()

    From the box and whisker plot, we can see overlap in the interquartile
    ranges between the control group and the group from drug A.
    We can apply the `dunnett`
    test to determine if the difference between means is significant. We
    set a significance level of .05 to reject the null hypothesis.

    >>> from scipy.stats import dunnett
    >>> res = dunnett(drug_a, drug_b, control=control)
    >>> res.pvalue
    array([0.47773146, 0.00889328])  # random

    The null hypothesis is that each group has the same mean. The p-value for
    comparisons between ``control`` and ``drug_b`` do not exceed .05,
    so we reject the null hypothesis that they
    have the same means. The p-value of the comparison between ``control``
    and ``drug_a`` exceeds .05, so we accept the null hypothesis that there
    is not a significant difference between their means.

    """
    rng = check_random_state(random_state)
    control = np.asarray(control)

    rho, df, n_group = iv_dunnett(
        observations=observations, control=control
    )

    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        statistic = np.array([
            stats.ttest_ind(
                obs_, control, alternative=alternative, random_state=rng
            ).statistic
            for obs_ in observations
        ])

    pvalue = pvalue_dunnett(
        rho=rho, df=df,
        statistic=statistic, alternative=alternative,
        rng=rng
    )

    control_mean = np.mean(control)
    observations_mean = np.array([np.mean(obs_) for obs_ in observations])
    std = np.sqrt((
        np.sum(control**2)
        + np.sum([obs_**2 for group in observations for obs_ in group])
        - n_group*(control_mean**2 + np.sum(observations_mean**2))
    ) / df)

    return DunnettResult(
        statistic=statistic, pvalue=pvalue,
        _rho=rho, _df=df, _std=std,
    )


def iv_dunnett(
    observations: npt.ArrayLike, control: npt.ArrayLike
) -> tuple[np.ndarray, int, int]:
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

    return rho, df, n_groups


def pvalue_dunnett(
    rho: np.ndarray, df: int, statistic: np.ndarray,
    alternative: Literal['two-sided', 'less', 'greater'],
    rng: SeedType = None
) -> np.ndarray:
    """pvalue from Dunnett critical value.

    Critical values come from the multivariate student-t distribution.
    """
    statistic = statistic.reshape(-1, 1)

    mvt = stats.multivariate_t(shape=rho, df=df, seed=rng)
    if alternative == "two-sided":
        pvalue = 1 - mvt.cdf(statistic, lower_limit=-statistic)
    else:
        pvalue = 1 - mvt.cdf(statistic)

    return pvalue
