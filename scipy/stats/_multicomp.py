from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.testing import suppress_warnings

from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.stats._common import ConfidenceInterval
from scipy.stats._qmc import check_random_state

if TYPE_CHECKING:
    import numpy.typing as npt
    from scipy._lib._util import DecimalNumber, SeedType
    from typing import Literal


__all__ = [
    'dunnett'
]


@dataclass
class DunnettResult:
    statistic: np.ndarray
    pvalue: np.ndarray
    _alternative: Literal['two-sided', 'less', 'greater']
    _rho: np.ndarray
    _df: int
    _std: float
    _observations_mean: np.ndarray
    _control_mean: np.ndarray
    _ci: ConfidenceInterval | None = None
    _ci_cl: DecimalNumber | None = None

    def __str__(self):
        # Note: `__str__` prints the confidence intervals from the most
        # recent call to `confidence_interval`. If it has not been called,
        # it will be called with the default CL of .95.
        if self._ci is None:
            self.confidence_interval(confidence_level=.95)
        s = (
            "Dunnett's test"
            f" ({self._ci_cl*100:.1f}% Confidence Interval)\n"
            "Comparison               Statistic  p-value  Lower CI  Upper CI\n"
        )
        for i in range(self.pvalue.shape[0]):
            s += (f" (Sample {i} - Control) {self.statistic[i]:>10.3f}"
                  f"{self.pvalue[i]:>10.3f}"
                  f"{self._ci.low[i]:>10.3f}"
                  f"{self._ci.high[i]:>10.3f}\n")

        if self._alternative == 'less':
            s += (
                "\nOne-sided alternative (less): sample i's mean exceed "
                "the control's mean by an amount at least Lower CI"
            )
        elif self._alternative == 'greater':
            s += (
                "\nOne-sided alternative (greater): sample i's mean exceed "
                "the control's mean by an amount at most Upper CI"
            )
        else:
            s += (
                "\nTwo-sided alternative: sample i's mean exceed the "
                "control's mean by an amount between Lower CI and Upper CI"
            )
        return s

    def _allowance(self, confidence_level: DecimalNumber = 0.95) -> float:
        """Allowance.

        It is the quantity to add/substract from the observed difference
        between the means of observed groups and the mean of the control
        group. The result gives confidence limits.

        Parameters
        ----------
        confidence_level : float, optional
            Confidence level for the computed confidence interval
            of the estimated proportion. Default is .95.

        Returns
        -------
        allowance : float
            Allowance around the mean.
        """
        alpha = 1 - confidence_level

        def pvalue_from_stat(statistic):
            # two-sided to have +- bounds
            statistic = np.array(statistic)
            sf = pvalue_dunnett(
                rho=self._rho, df=self._df,
                statistic=statistic, alternative=self._alternative
            )
            return abs(sf - alpha)

        res = minimize_scalar(pvalue_from_stat, method='brent', tol=1e-4)
        critical_value = res.x

        allowance = critical_value*self._std*np.sqrt(2/len(self.pvalue))
        return allowance

    def confidence_interval(
        self, confidence_level: DecimalNumber = 0.95
    ) -> ConfidenceInterval:
        """Compute the confidence interval for the specified confidence level.

        The confidence interval corresponds to the difference in means of the
        groups with the control +- the allowance.

        Parameters
        ----------
        confidence_level : float, optional
            Confidence level for the computed confidence interval
            of the estimated proportion. Default is .95.

        Returns
        -------
        ci : ``ConfidenceInterval`` object
            The object has attributes ``low`` and ``high`` that hold the
            lower and upper bounds of the confidence intervals for each
            comparison. The high and low values are accessible for each
            comparison at index ``(i,)`` for each group ``i``.

        """
        # check to see if the supplied confidence level matches that of the
        # previously computed CI.
        if (self._ci is not None and self._ci_cl is not None and
                confidence_level == self._ci_cl):
            return self._ci

        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1.")

        allowance = self._allowance(confidence_level=confidence_level)
        diff_means = self._observations_mean - self._control_mean

        low = diff_means-allowance
        high = diff_means+allowance

        if self._alternative == 'less':
            high = [np.nan] * len(diff_means)
        elif self._alternative == 'greater':
            low = [np.nan] * len(diff_means)

        self._ci_cl = confidence_level
        self._ci = ConfidenceInterval(
            low=low,
            high=high
        )
        return self._ci


def dunnett(
    *samples: npt.ArrayLike, control: npt.ArrayLike,
    alternative: Literal['two-sided', 'less', 'greater'] = "two-sided",
    random_state: SeedType = None
) -> DunnettResult:
    """Dunnett's test.

    Parameters
    ----------
    sample1, sample2, ... : 1D array_like
        The sample measurements for each experiment group.
    control : 1D array_like
        The sample measurements for the control group.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

        * 'two-sided': the means of the distributions underlying the samples
          and control are unequal.
        * 'less': the means of the distribution underlying the samples
          is less than the mean of the distribution underlying the control.
        * 'greater': the means of the distribution underlying the
          samples is greater than the mean of the distribution underlying
          the control.
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
            The t-statistic for the test.  For 1-D inputs a scalar is
            returned.
        pvalue : scalar ndarray
            The p-value for the test.

    See Also
    --------
    tukey_hsd : performs pairwise comparison of means.

    Notes
    -----
    Dunnett's test [1]_ compares the means of multiple experiment groups
    against a control group.
    `tukey_hsd` instead, performs pairwise comparison of means.
    It means Dunnett's test performs fewer tests, hence there is less p-value
    adjustment which makes the test more powerful.
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
    samples, control, rng = iv_dunnett(
        samples=samples, control=control, random_state=random_state
    )

    rho, df, n_group = params_dunnett(
        samples=samples, control=control
    )

    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        statistic = np.array([
            stats.ttest_ind(
                obs_, control, alternative=alternative, random_state=rng
            ).statistic
            for obs_ in samples
        ])

    pvalue = pvalue_dunnett(
        rho=rho, df=df,
        statistic=abs(statistic), alternative=alternative,
        rng=rng
    )

    control_mean = np.mean(control)
    observations_mean = np.array([np.mean(obs_) for obs_ in samples])
    std = np.sqrt((
        np.sum(control**2)
        + np.sum([obs_**2 for group in samples for obs_ in group])
        - n_group*(control_mean**2 + np.sum(observations_mean**2))
    ) / df)

    return DunnettResult(
        statistic=statistic, pvalue=pvalue,
        _alternative=alternative,
        _rho=rho, _df=df, _std=std,
        _observations_mean=observations_mean,
        _control_mean=control_mean
    )


def iv_dunnett(
    samples: npt.ArrayLike, control: npt.ArrayLike, random_state: SeedType
) -> tuple[npt.ArrayLike, np.ndarray, SeedType]:
    """Input validation for Dunnett's test."""
    rng = check_random_state(random_state)

    ndim_msg = "Control and samples groups must be 1D arrays"
    n_obs_msg = "Control and samples groups must have at least 1 observation"

    control = np.asarray(control)
    samples = [np.asarray(sample) for sample in samples]

    # samples checks
    for sample in (samples + [control]):
        sample = np.asarray(sample)
        if sample.ndim > 1:
            raise ValueError(ndim_msg)

        if len(sample) < 1:
            raise ValueError(n_obs_msg)

    return samples, control, rng


def params_dunnett(
    samples: npt.ArrayLike, control: npt.ArrayLike
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
    n_n_obs = np.array([len(obs_) for obs_ in samples])

    # From Dunnett1955 p. 1100 d.f. = (sum N)-(p+1)
    n_obs = n_n_obs.sum()
    n_control = len(control)
    n = n_obs + n_control
    n_groups = len(samples)
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
