# stats.py implemented by bin-wu at 18:38 in 2019.04.21
import numpy as np
import pandas as pd
from scipy.stats import t
from pprint import pprint


def t_test(v1: np.ndarray,
           v2: np.ndarray = None,
           alternative: str = None,
           paired: bool = False,
           alpha: float = 0.05) -> dict:
    """
    The classic t-tests

    Parameters
    ---------
    v1: a (non-empty) numeric vector of data values
    v2: an optional (non-empty) numeric vector of data values; when empty, performing the one-sample t-test
                 When you want to get the confidence level and p-value of one_sample; leave v2 to be None by default
                 When you compare two groups, set v2 to be the second group
    alternative: specifying the alternative hypothesis, "greater" (one-tailed) or None (default two-tailed)
                 When you want to do test mean of v1 is greater than that of v2; set alternative as "greater"
                 When you want to do test mean of v1 is not equal to v2; leave alternative as None by default
    paired: a logical indicating whether you want a paired t-test
                 when your data is paired with same size, set paired equal to true
                 when your data is not paired with possible different size, set paired equal to false
    alpha: confidence level of the interval

    Returns
    -------
    A dictionary including information of confidence interval and test information

    Examples
    --------
    v1 = np.array([9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19])
    v2 = np.array([7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9])
    # Two sample unpaired t-test
    print(t_test(v1, v2, paired=False)) # two-tailed
    print(t_test(v1, v2, paired=False, alternative="greater")) # one-tailed
    # Two sample paired t-test
    print(t_test(v1, v2, paired=True))
    print(t_test(v1, v2, paired=True, alternative="greater"))
    # One sample t-test
    print(t_test(v1))
    print(t_test(v1, alternative="greater"))

    Checked by R
    ------------
    ++++++++++++++++++++++++++++++++++++++++++++++
    Welch Two Sample t-test (unpaired, two-tailed)
    ++++++++++++++++++++++++++++++++++++++++++++++

    In python
    ---------
    v1 = np.array([9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19])
    v2 = np.array([7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9])
    print(t_test(v1, v2, paired=False))
    # {'alternative_hypothesis': 'true difference in means is not equal to 0',
    #  'confidence_interval': [0.5922803841924627, 3.4166085046964207],
    #  'df': 15.992693827602638,
    #  'mean_v1': 10.297777777777776,
    #  'mean_v2': 8.293333333333335,
    #  'p-value': 0.008322739957316028,
    #  'significance_level': 0.05,
    #  't': 3.009133495521211,
    #  'title': 'Welch Two sample t-test (unpaired, two tailed)'}

    In R
    ----
    v1=c(9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19)
    v2=c(7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9)
    t.test(v1, v2)
    # 	Welch Two Sample t-test
    # data:  v1 and v2
    # t = 3.0091, df = 15.993, p-value = 0.008323
    # alternative hypothesis: true difference in means is not equal to 0
    # 95 percent confidence interval:
    #  0.5922804 3.4166085
    # sample estimates:
    # mean of x mean of y
    # 10.297778  8.29333

    ++++++++++++++++++++++++++++++++++++++++++++++
    Welch Two Sample t-test (unpaired, one-tailed)
    ++++++++++++++++++++++++++++++++++++++++++++++

    In Python
    ---------
    v1 = np.array([9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19])
    v2 = np.array([7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9])
    print(t_test(v1, v2, paired=False, alternative="greater"))
    # {'alternative_hypothesis': 'true difference in means is greater than 0',
    #  'confidence_interval': [0.8414435682377779, inf],
    #  'df': 15.992693827602638,
    #  'mean_v1': 10.297777777777776,
    #  'mean_v2': 8.293333333333335,
    #  'p-value': 0.004161369978658014,
    #  'significance_level': 0.05,
    #  't': 3.009133495521211,
    #  'title': 'Welch Two sample t-test (unpaired, one tailed)'}

    In R
    ----
    v1=c(9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19)
    v2=c(7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9)
    t.test(v1, v2, alternative = "greater")
    # 	Welch Two Sample t-test
    # data:  v1 and v2
    # t = 3.0091, df = 15.993, p-value = 0.004161
    # alternative hypothesis: true difference in means is greater than 0
    # 95 percent confidence interval:
    #  0.8414436       Inf
    # sample estimates:
    # mean of x mean of y
    # 10.297778  8.293333

    ++++++++++++++++++++++++++
    Paired t-test (two-tailed)
    ++++++++++++++++++++++++++
    In Python
    ---------
    v1 = np.array([9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19])
    v2 = np.array([7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9])
    print(t_test(v1, v2, paired=True)) # Two sample paired t-test
    # {'alternative_hypothesis': 'true difference in means is not equal to 0',
    #  'confidence_interval': [0.39454766585932055, 3.614341223029567],
    #  'df': 8,
    #  'mean_diff': 2.004444444444444,
    #  'p-value': 0.020792130628375594,
    #  'significance_level': 0.05,
    #  't': 2.871151268093026,
    #  'title': 'Paired t-test (two tailed)'}

    In R
    ----
    v1=c(9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19)
    v2=c(7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9)
    t.test(v1, v2, paired=TRUE)
    # 	Paired t-test
    # data:  v1 and v2
    # t = 2.8712, df = 8, p-value = 0.02079
    # alternative hypothesis: true difference in means is not equal to 0
    # 95 percent confidence interval:
    #  0.3945477 3.6143412
    # sample estimates:
    # mean of the differences
    #                2.004444


    ++++++++++++++++++++++++++
    Paired t-test (one-tailed)
    ++++++++++++++++++++++++++
    In Python
    ---------
    v1 = np.array([9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19])
    v2 = np.array([7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9])
    print(t_test(v1, v2, paired=True, alternative="greater")) # Two sample paired t-test
    # {'alternative_hypothesis': 'true difference in means is greater than 0',
    #  'confidence_interval': [0.7062332444940196, inf],
    #  'df': 8,
    #  'mean_diff': 2.004444444444444,
    #  'p-value': 0.010396065314187797,
    #  'significance_level': 0.05,
    #  't': 2.871151268093026,
    #  'title': 'Paired t-test (one tailed)'}

    In R
    ----
    v1=c(9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19)
    v2=c(7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9)
    t.test(v1, v2, paired=TRUE, alternative="greater")
    # 	Paired t-test
    # data:  v1 and v2
    # t = 2.8712, df = 8, p-value = 0.0104
    # alternative hypothesis: true difference in means is greater than 0
    # 95 percent confidence interval:
    #  0.7062332       Inf
    # sample estimates:
    # mean of the differences
    #                2.004444

    ++++++++++++++++++++++++++++++
    One sample t-test (two-tailed)
    ++++++++++++++++++++++++++++++
    In Python
    ---------
    v1 = np.array([9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19])
    v2 = np.array([7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9])
    print(t_test(v1)) # One sample t-test
    # {'alternative_hypothesis': 'true mean is not equal to 0',
    #  'confidence_interval': [9.223278703348157, 11.372276852207396],
    #  'df': 8,
    #  'mean': 10.297777777777776,
    #  'p-value': 1.856236320207927e-08,
    #  'significance_level': 0.05,
    #  't': 22.100268583121732,
    #  'title': 'One sample t-test (two tailed)'}

    In R
    ----
    # v=c(9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19)
    # t.test(v)
    # # 	One Sample t-test
    # # data:  v
    # # t = 22.1, df = 8, p-value = 1.856e-08
    # # alternative hypothesis: true mean is not equal to 0
    # # 95 percent confidence interval:
    # #   9.223279 11.372277
    # # sample estimates:
    # # mean of x
    # #  10.29778

    ++++++++++++++++++++++++++++++
    One sample t-test (one-tailed)
    ++++++++++++++++++++++++++++++
    In Python
    ---------
    v1 = np.array([9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19])
    v2 = np.array([7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9])
    print(t_test(v1, alternative="greater"))
    # {'alternative_hypothesis': 'true mean is greater than 0',
    #  'confidence_interval': [9.431308106357697, inf],
    #  'df': 8,
    #  'mean': 10.297777777777776,
    #  'p-value': 9.281181601039634e-09,
    #  'significance_level': 0.05,
    #  't': 22.100268583121732,
    #  'title': 'One sample t-test(one tailed)'}

    In R
    ----
    v=c(9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19)
    t.test(v, alternative="greater")
    # 	One Sample t-test
    # data:  v
    # t = 22.1, df = 8, p-value = 9.281e-09
    # alternative hypothesis: true mean is greater than 0
    # 95 percent confidence interval:
    #  9.431308      Inf
    # sample estimates:
    # mean of x
    #  10.29778
 """
    if v2 is None:
        return one_sample_t_test(v1, alpha, alternative)  # One sample t-test
    elif paired:
        # Two sample paired t-test
        return two_sample_t_test_paired(v1, v2, alpha, alternative)
    else:
        # Two sample unpaired t-test
        return two_sample_t_test_unpaired(v1, v2, alpha, alternative)


def bootstrap_t_test(v1: np.ndarray,
                     v2: np.ndarray = None,
                     alternative: str = None,
                     paired: bool = False,
                     alpha: float = 0.05,
                     bootnum: int = 599,
                     csv_file: str = None,
                     seed: int = 2019) -> dict:
    """
    The bootstrap t-tests
        Bootstrap unpaired t-test as described in Efron and Tibshirani (1993), (Algorithm 16.2, p224) (checked by R result)
        Bootstrap paired t-test as described in Efron and Tibshirani (1993), (Section 16.4, p225) (no R implementation)
        One sample t-test shares the same implementation of Bootstrap paired t-test (no R implementation)

    Parameters
    ---------
    v1: a (non-empty) numeric vector of data values
    v2: an optional (non-empty) numeric vector of data values; when empty, performing the one-sample t-test
                 When you want to get the confidence level and p-value of one_sample; leave v2 to be None by default
                 When you compare two groups, set v2 to be the second group
    alternative: specifying the alternative hypothesis, "greater" (one-tailed) or "less" or "two.sided" (same as default None)
                 When you want to do test mean of v1 is greater than v2; set alternative as "greater"
                 When you want to do test mean of v1 is less than that of v2; set alternative as "less"
                 When you want to do test mean of v1 is equals to that of v2; set alternative as "two.sided"
    paired: a logical indicating whether you want a paired t-test
                 when your data is paired with same size, set paired equal to true
                 when your data is not paired with possible different size, set paired equal to false
    alpha: confidence level of the interval (not used in bootstrap version, may for future implementing bootstrap-t C.I.)
    bootnum: number of samples for bootstrap simulation
         Rand R. Wilcox suggests the bootstrap number can be 599.
         Larry Wasserman suggests B = 10000 is usually sufficient in practice. 
    csv_file: the name of the file to store the test statistics of every sample
    seed: the numpy seed to ensure we can repeat the experiemnts for bootstrap

    Returns
    -------
    A dictionary including information of test information

    Examples
    --------
    v1 = np.array([9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19])
    v2 = np.array([7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9])
    # Two sample unpaired t-test with bootstrap
    print(bootstrap_t_test(v1, v2, paired=False, bootnum=100000)) # two-tailed
    print(bootstrap_t_test(v1, v2, paired=False, alternative="greater", bootnum=100000)) # one-tailed
    print(bootstrap_t_test(v1, v2, paired=False, alternative="less", bootnum=100000)) # one-tailed
    # Two sample paired t-test with bootstrap
    print(bootstrap_t_test(v1, v2, paired=True, bootnum=100000))
    print(bootstrap_t_test(v1, v2, paired=True, alternative="greater", bootnum=100000))
    print(bootstrap_t_test(v1, v2, paired=True, alternative="less", bootnum=100000))
    # One sample t-test with bootstrap
    print(bootstrap_t_test(v1, bootnum=100000))
    print(bootstrap_t_test(v1, alternative="greater", bootnum=100000))
    print(bootstrap_t_test(v1, alternative="less", bootnum=100000))

    Checked by R (only bootstrap unpaired t-test; paired and one sample implemented according to Efron and Tibshirani (1993))
    ------------------------------------------
    +++++++++++++++++++++++++++++++++
    Bootstrap Welch Two Sample t-test
    +++++++++++++++++++++++++++++++++   
    In Python
    ---------
    v1 = np.array([9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19])
    v2 = np.array([7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9])

    # same as pprint(bootstrap_t_test(v1, v2, paired=False, bootnum=100000)) # two-tailed
    pprint(bootstrap_t_test(v1, v2, paired=False, bootnum=100000, alternative="two.sided"))
    # {'alternative_hypothesis': 'true difference in means is not equal to 0',
    #  'p-value': 0.00926,
    #  'significance_level': 0.05,
    #  't': array([-1.00598128, -0.06186053,  0.41434586, ..., -0.376338  ,
    #        -0.73638333, -1.493951  ]),
    #  't_obs': 3.009133495521211,
    #  'title': 'Bootstrap Welch Two sample t-test (unpaired, two tailed)'}

    In R
    ----
    library("nonpar")
    v1=c(9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19)
    v2=c(7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9)
    set.seed(2019)
    boot.t.test(x = v1, y = v2, reps = 100000)
    # Bootstrap Two Sample t-test
    # t = 3.009, p-value = 0.0091
    # Alternative hypothesis: true difference in means is not equal to 0
    # $mu0
    # 0
    # $statistic
    # 3.00913349552122
    # $alternative
    # 'two.sided'
    # $p.value
    # 0.00914

    In Python
    ---------
    v1 = np.array([9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19])
    v2 = np.array([7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9])
    pprint(bootstrap_t_test(v1, v2, paired=False, alternative="greater", bootnum=100000)) # one-tailed
    # {'alternative_hypothesis': 'true difference in means is greater than 0',
    #  'p-value': 0.00388,
    #  'significance_level': 0.05,
    #  't': array([-1.00598128, -0.06186053,  0.41434586, ..., -0.376338  ,
    #        -0.73638333, -1.493951  ]),
    #  't_obs': 3.009133495521211,
    #  'title': 'Bootstrap Welch Two sample t-test (unpaired, one tailed)'}

    In R
    ----
    library("nonpar")
    v1=c(9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19)
    v2=c(7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9)
    # Bootstrap Two Sample t-test
    # t = 3.009, p-value = 0.004
    # Alternative hypothesis: true difference in means is greater than 0
    # $mu0
    # 0
    # $statistic
    # 3.00913349552122
    # $alternative
    # 'greater'
    # $p.value
    # 0.00396

    In Python
    ---------
    v1 = np.array([9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19])
    v2 = np.array([7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9])
    pprint(bootstrap_t_test(v1, v2, paired=False, alternative="less", bootnum=100000)) # one-tailed
    # {'alternative_hypothesis': 'true difference in means is less than 0',
    #  'p-value': 0.99612,
    #  'significance_level': 0.05,
    #  't': array([-1.00598128, -0.06186053,  0.41434586, ..., -0.376338  ,
    #        -0.73638333, -1.493951  ]),
    #  't_obs': 3.009133495521211,
    #  'title': 'Bootstrap Welch Two sample t-test (unpaired, one tailed)'}

    In R
    ----
    library("nonpar")
    v1=c(9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19)
    v2=c(7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9)
    # Bootstrap Two Sample t-test
    # t = 3.009, p-value = 0.996
    # Alternative hypothesis: true difference in means is less than 0
    # $mu0
    # 0
    # $statistic
    # 3.00913349552122
    # $alternative
    # 'less'
    # $p.value
    # 0.99604

    In Python
    ---------
    ++++++++++++++++++++++++++
    Bootstrap paired t-test
    ++++++++++++++++++++++++++
    v1 = np.array([9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19])
    v2 = np.array([7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9])
    # Two sample paired t-test with bootstrap
    pprint(bootstrap_t_test(v1, v2, paired=True, bootnum=100000))
    pprint(bootstrap_t_test(v1, v2, paired=True, alternative="greater", bootnum=100000))
    pprint(bootstrap_t_test(v1, v2, paired=True, alternative="less", bootnum=100000))
    # {'alternative_hypothesis': 'true difference in means is not equal to 0',
    #  'p-value': 0.02632,
    #  'significance_level': 0.05,
    #  't': array([-0.65836578, -0.11941834, -0.82259935, ..., -1.74996078,
    #        -1.95073269, -1.12268204]),
    #  't_obs': 2.871151268093026,
    #  'title': 'Bootstrap paired t-test (paired, two tailed)'}
    # {'alternative_hypothesis': 'true difference in means is greater than 0',
    #  'p-value': 0.01543,
    #  'significance_level': 0.05,
    #  't': array([-0.65836578, -0.11941834, -0.82259935, ..., -1.74996078,
    #        -1.95073269, -1.12268204]),
    #  't_obs': 2.871151268093026,
    #  'title': 'Bootstrap paired t-test (paired, one tailed)'}
    # {'alternative_hypothesis': 'true difference in means is less than 0',
    #  'p-value': 0.98457,
    #  'significance_level': 0.05,
    #  't': array([-0.65836578, -0.11941834, -0.82259935, ..., -1.74996078,
    #        -1.95073269, -1.12268204]),
    #  't_obs': 2.871151268093026,
    #  'title': 'Bootstrap paired t-test (paired, one tailed)'}

    +++++++++++++++++++++++++++
    Bootstrap one sample t-test
    +++++++++++++++++++++++++++
    v1 = np.array([9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19])
    v2 = np.array([7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9])
    # One sample t-test with bootstrap
    pprint(bootstrap_t_test(v1, bootnum=100000))
    pprint(bootstrap_t_test(v1, alternative="greater", bootnum=100000))
    pprint(bootstrap_t_test(v1, alternative="less", bootnum=100000))
    {'alternative_hypothesis': 'true mean is not equal to 0',
    #  'p-value': 0.0,
    #  'significance_level': 0.05,
    #  't': array([ -1.5873776 ,  -0.99710713,   0.23796843, ...,  -0.74470933,
    #         -0.36206014, -10.37798998]),
    #  't_obs': 22.100268583121732,
    #  'title': 'Bootstrap one sample t-test (two tailed)'}
    # {'alternative_hypothesis': 'true mean is greater than 0',
    #  'p-value': 0.0,
    #  'significance_level': 0.05,
    #  't': array([ -1.5873776 ,  -0.99710713,   0.23796843, ...,  -0.74470933,
    #         -0.36206014, -10.37798998]),
    #  't_obs': 22.100268583121732,
    #  'title': 'Bootstrap one sample t-test (one tailed)'}
    # {'alternative_hypothesis': 'true mean is less than 0',
    #  'p-value': 1.0,
    #  'significance_level': 0.05,
    #  't': array([ -1.5873776 ,  -0.99710713,   0.23796843, ...,  -0.74470933,
    #         -0.36206014, -10.37798998]),
    #  't_obs': 22.100268583121732,
    #  'title': 'Bootstrap one sample t-test (one tailed)'}

"""
    if v2 is None:  # Bootstrap one sample t-test
        return bootstrap_one_sample_t_test(v1, alpha, alternative, bootnum, csv_file, seed)
    elif paired:  # Bootstrap two sample paired t-test
        return bootstrap_two_sample_t_test_paired(v1, v2, alpha, alternative, bootnum, csv_file, seed)
    else:  # Bootstrap two sample unpaired t-test (checked by R)
        return bootstrap_two_sample_t_test_unpaired(v1, v2, alpha, alternative, bootnum, csv_file, seed)


def bootstrap_ci(v, bootnum=599, bootfunc=np.mean, alpha=0.05, csv_file=None, seed=2019):
    """
    Simple bootstrap confidence interval of test statistics of one sample based on percentile.
    # Bootstrap percetile interval as described in Efron and Tibshirani (1993), (Section 13.3, p170)

    Parameters
    ----------
    v: a one dimensional numpy vector as the observered one sample
    bootnum: number of samples for bootstrap simulation
             Rand R. Wilcox suggests the bootstrap number can be 599.
             Larry Wasserman suggests B = 10000 is usually sufficient in practice.
    bootfunc: test statistics on one sample
    csv_file: csv file to store the test statistics of all bootstrap samples
    seed: the numpy seed to ensure the replicates of the experiments

    Example:
    -------
    v1 = np.array([9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19])
    v2 = np.array([7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9])
    diff = v1-v2
    print(bootstrap_ci(diff, bootnum=100000, bootfunc=np.mean, alpha=0.05)['ci'])
    # output:
    # [0.71       3.26780556] # R result: ( 0.701,  3.266 )

    Checked by R
    ------------
    v1 = c(9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19)
    v2 = c(7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9)

    # diff <- x1-x2
    diff <- v1-v2
    library(boot)
    set.seed(2019)
    sample_mean = function(data, indices){ return(mean(data[indices]))}
    results = boot(diff, statistic=sample_mean, R=100000)
    confidence_interval_H = boot.ci(results, index = 1, conf = 0.95, type = 'perc')
    print(confidence_interval_H)
    # BOOTSTRAP CONFIDENCE INTERVAL CALCULATIONS
    # Based on 100000 bootstrap replicates

    # CALL :
    # boot.ci(boot.out = results, conf = 0.95, type = "perc", index = 1)

    # Intervals :
    # Level     Percentile
    # 95%   ( 0.701,  3.266 )
    # Calculations and Intervals on Original Scale
   """
    assert len(v.shape) == 1, "only support the one dimension numpy array"
    with NumpyRNGContext(seed):
        boot_stat = bootstrap(v, bootnum, bootfunc=bootfunc)
    if csv_file:
        np.savetxt(csv_file, boot_stat, fmt="%.4f")
    return {'ci': np.quantile(boot_stat, [alpha / 2, 1 - alpha / 2]), 'stat': boot_stat}


class NumpyRNGContext:
    """
    A context manager (for use with the ``with`` statement) that will seed the
    numpy random number generator (RNG) to a specific value, and then restore
    the RNG state back to whatever it was before.

    This is primarily intended for use in the astropy testing suit, but it
    may be useful in ensuring reproducibility of Monte Carlo simulations in a
    science context.

    # From http://docs.astropy.org/en/stable/_modules/astropy/utils/misc.html#NumpyRNGContext


    Parameters
    ----------
    seed : int
        The value to use to seed the numpy RNG

    Examples
    --------
    A typical use case might be::

        with NumpyRNGContext(<some seed value you pick>):
            from numpy import random

            randarr = random.randn(100)
            ... run your test using `randarr` ...

        # Any code using numpy.random at this indent level will act just as it
        # would have if it had been before the with statement - e.g. whatever
        # the default seed is.
    """

    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        from numpy import random

        self.startstate = random.get_state()
        random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        from numpy import random

        random.set_state(self.startstate)


def bootstrap(data, bootnum=100, samples=None, bootfunc=None, *args, **kwargs):
    """
    Performs bootstrap resampling on numpy arrays.

    Bootstrap resampling is used to understand confidence intervals of sample
    estimates. This function returns versions of the dataset resampled with
    replacement ("case bootstrapping"). These can all be run through a function
    or statistic to produce a distribution of values which can then be used to
    find the confidence intervals.

    # Modified from http://docs.astropy.org/en/stable/_modules/astropy/stats/funcs.html#bootstrap

    Parameters
    ----------
    data : numpy.ndarray
        N-D array. The bootstrap resampling will be performed on the first
        index, so the first index should access the relevant information
        to be bootstrapped.
    bootnum : int, optional
        Number of bootstrap resamples
    samples : int, optional
        Number of samples in each resample. The default `None` sets samples to
        the number of datapoints
    bootfunc : function, optional
        Function to reduce the resampled data. Each bootstrap resample will
        be put through this function and the results returned. If `None`, the
        bootstrapped data will be returned
    *args, **kwargs: parameters for the boot function


    Returns
    -------
    boot : numpy.ndarray

        If bootfunc is None, then each row is a bootstrap resample of the data.
        If bootfunc is specified, then the columns will correspond to the
        outputs of bootfunc.

    Examples
    --------
    Obtain a twice resampled array:

    >>> bootarr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
    >>> with NumpyRNGContext(1):
    ...     bootresult = bootstrap(bootarr, 2)
    ...
    >>> bootresult  # doctest: +FLOAT_CMP
    array([[6., 9., 0., 6., 1., 1., 2., 8., 7., 0.],
           [3., 5., 6., 3., 5., 3., 5., 8., 8., 0.]])
    >>> bootresult.shape
    (2, 10)

    Obtain a statistic on the array

    >>> with NumpyRNGContext(1):
    ...     bootresult = bootstrap(bootarr, 2, bootfunc=np.mean)
    ...
    >>> bootresult  # doctest: +FLOAT_CMP
    array([4. , 4.6])

    Obtain a statistic with two outputs on the array

    >>> test_statistic = lambda x: (np.sum(x), np.mean(x))
    >>> with NumpyRNGContext(1):
    ...     bootresult = bootstrap(bootarr, 3, bootfunc=test_statistic)
    >>> bootresult  # doctest: +FLOAT_CMP
    array([[40. ,  4. ],
           [46. ,  4.6],
           [35. ,  3.5]])
    >>> bootresult.shape
    (3, 2)

    Obtain a statistic with two outputs on the array, keeping only the first
    output

    >>> bootfunc = lambda x:test_statistic(x)[0]
    >>> with NumpyRNGContext(1):
    ...     bootresult = bootstrap(bootarr, 3, bootfunc=bootfunc)
    ...
    >>> bootresult  # doctest: +FLOAT_CMP
    array([40., 46., 35.])
    >>> bootresult.shape
    (3,)

    with NumpyRNGContext(2):
        boot_result = bootstrap(np.array([[1,1], [2, 2], [3, 3]]), bootnum=2)
        print(boot_result)
    with NumpyRNGContext(2):
        boot_result = bootstrap(np.array([[1,1], [2, 2], [3, 3]]), bootnum=2, bootfunc=np.mean)
        print(boot_result)
    with NumpyRNGContext(2):
        boot_result = bootstrap(np.array([[1,1], [2, 2], [3, 3]]), bootnum=2, bootfunc=np.mean, axis=0)
        print(boot_result)

    # [[[1. 1.]
    #   [2. 2.]
    #   [1. 1.]]

    #  [[3. 3.]
    #   [3. 3.]
    #   [1. 1.]]]
    # [1.33333333 2.33333333]
    # [[1.33333333 1.33333333]
    #  [2.33333333 2.33333333]]
    """
    if samples is None:
        samples = data.shape[0]

    # make sure the input is sane
    if samples < 1 or bootnum < 1:
        raise ValueError("neither 'samples' nor 'bootnum' can be less than 1.")

    if bootfunc is None:
        resultdims = (bootnum,) + (samples,) + data.shape[1:]
    else:
        # test number of outputs from bootfunc, avoid single outputs which are
        # array-like
        try:
            resultdims = (bootnum, len(bootfunc(data, *args, **kwargs)))
        except TypeError:
            resultdims = (bootnum,)

    # create empty boot array
    boot = np.empty(resultdims)

    for i in range(bootnum):
        bootarr = np.random.randint(low=0, high=data.shape[0], size=samples)
        if bootfunc is None:
            boot[i] = data[bootarr]
        else:
            boot[i] = bootfunc(data[bootarr], *args, **kwargs)

    return boot


def df_bootstrap(data, bootnum=100, samples=None, bootfunc=None, *args, **kwargs):
    """bootfunc input an array with shape (N, D), output an array with shape (D, )

       eg.
       df=pd.DataFrame(np.array([[1,1], [2, 2], [3, 3]]), columns=["first", "second"])
        print(df)
        with NumpyRNGContext(2):
            print(df_bootstrap(df, bootnum=2, bootfunc=np.mean, axis=0))
        #    first  second
        # 0      1       1
        # 1      2       2
        # 2      3       3
        # [[[1. 1.]
        #   [2. 2.]
        #   [1. 1.]]

        # [[3. 3.]
        #  [3. 3.]
        #  [1. 1.]]]
        #
        #       first    second
        # 0  1.333333  1.333333
        # 1  2.333333  2.333333
       """
    data_values = bootstrap(data.values, bootnum,
                            samples, bootfunc, *args, **kwargs)
    return pd.DataFrame(data_values, columns=data.columns)


def two_sample_t_test_unpaired(v1, v2, alpha=0.05, alternative=None):
    """
    v1 = np.array([9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19])
    v2 = np.array([7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9])
    print(two_sample_t_test_unpaired(v1, v2))
    print(two_sample_t_test_unpaired(v1, v2, alternative="greater"))

    # In R
    # v1=c(9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19)
    # v2=c(7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9)
    # ## A two sample Welch t-test
    # t.test(v1, v2)
    # t.test(v1, v2, alternative = "greater")
    """
    assert len(v1.shape) == 1 and len(
        v2.shape) == 1, "only support the one dimension numpy array"
    assert alternative == None or alternative == "greater", "alternative have to be empty or greater"

    result = {}
    if not alternative:
        result['title'] = "Welch Two sample t-test (unpaired, two tailed)"
    else:
        result['title'] = "Welch Two sample t-test (unpaired, one tailed)"

    if not alternative:
        result['alternative_hypothesis'] = "true difference in means is not equal to 0"
    else:
        result['alternative_hypothesis'] = "true difference in means is greater than 0"

    result['significance_level'] = alpha
    n1 = len(v1)
    n2 = len(v2)

    # distance_mean
    data_distance = v1.mean() - v2.mean()
    result['mean_v1'] = v1.mean()
    result['mean_v2'] = v2.mean()

    # distance_variance
    # make std of numpy same as sd in R, normalized by n - 1
    std1 = v1.std(ddof=1)
    std2 = v2.std(ddof=1)
    mean_var1 = pow(std1, 2) / n1
    mean_var2 = pow(std2, 2) / n2
    pooled_variance = mean_var1 + mean_var2

    # t_quantile
    degrees_of_freedom = pow(
        pooled_variance, 2) / ((pow(mean_var1, 2) / (n1 - 1) + pow(mean_var2, 2) / (n2 - 1)))
    result['df'] = degrees_of_freedom

    # confidence interval
    t_quantile = t.ppf(1 - alpha / 2, degrees_of_freedom)
    lower = data_distance - t_quantile * np.sqrt(pooled_variance)
    upper = data_distance + t_quantile * np.sqrt(pooled_variance)
    # confidence interval for alternative "greater"
    if alternative:
        t_quantile = t.ppf(1 - alpha, degrees_of_freedom)
        lower = data_distance - t_quantile * np.sqrt(pooled_variance)
        upper = np.inf
    result['confidence_interval'] = [lower, upper]

    # p-value
    t_stat = data_distance / np.sqrt(pooled_variance)
    result['t'] = t_stat
    p_value = (1 - t.cdf(t_stat, degrees_of_freedom)) * 2
    # p-value for alternative "greater"
    if alternative == "greater":
        p_value /= 2
    result['p-value'] = p_value

    return result


def two_sample_t_test_paired(v1, v2, alpha=0.05, alternative=None):
    """
    v1 = np.array([9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19])
    v2 = np.array([7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9])
    print(two_sample_t_test_paired(v1, v2))
    print(two_sample_t_test_paired(v1, v2, alternative="greater"))

    # In R
    # v1=c(9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19)
    # v2=c(7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9)
    # ## Paired t-test
    # t.test(v1, v2, paired=TRUE)
    # t.test(v1, v2, paired=TRUE, alternative="greater")
    """
    assert len(v1.shape) == 1 and len(
        v2.shape) == 1, "only support the one dimension numpy array"
    assert alternative == None or alternative == "greater", "alternative have to be empty or greater"
    assert len(v1) == len(
        v2), "We are using paired T-test, the number of samples are same"

    result = {}
    if not alternative:
        result['title'] = "Paired t-test (two tailed)"
    else:
        result['title'] = "Paired t-test (one tailed)"

    if not alternative:
        result['alternative_hypothesis'] = "true difference in means is not equal to 0"
    else:
        result['alternative_hypothesis'] = "true difference in means is greater than 0"

    result['significance_level'] = alpha

    n = len(v1)

    # distance_mean
    data_distance = (v1 - v2).mean()
    result['mean_diff'] = data_distance

    # distance_variance
    mean_variance = pow((v1 - v2).std(ddof=1), 2) / n

    # t_quantile to create confidence interval surrounding data
    degrees_of_freedom = n - 1
    result['df'] = degrees_of_freedom

    # confidence interval
    t_quantile = t.ppf(1 - alpha / 2, df=degrees_of_freedom)
    lower = data_distance - t_quantile * np.sqrt(mean_variance)
    upper = data_distance + t_quantile * np.sqrt(mean_variance)
    # confidence interval for alternative "greater"
    if alternative:
        t_quantile = t.ppf(1 - alpha, degrees_of_freedom)
        lower = data_distance - t_quantile * np.sqrt(mean_variance)
        upper = np.inf
    result['confidence_interval'] = [lower, upper]

    # p-value
    t_stat = data_distance / np.sqrt(mean_variance)
    result['t'] = t_stat
    p_value = (1 - t.cdf(t_stat, df=degrees_of_freedom)) * 2
    # p-value for alternative "greater"
    if alternative == "greater":
        p_value /= 2
    result['p-value'] = p_value

    return result


def one_sample_t_test(v, alpha=0.05, alternative=None):
    """
    print(t_test(v1))
    print(t_test(v1, alternative="greater"))

    # In R
    # v=c(9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19)
    # # One sample t-test
    # t.test(v)
    # t.test(v, alternative="greater")
    """
    assert alternative == None or alternative == "greater", "alternative have to be empty or greater"
    result = {}
    if not alternative:
        result['title'] = "One sample t-test (two tailed)"
    else:
        result['title'] = "One sample t-test(one tailed)"

    if not alternative:
        result['alternative_hypothesis'] = "true mean is not equal to 0"
    else:
        result['alternative_hypothesis'] = "true mean is greater than 0"

    result['significance_level'] = alpha

    n = len(v)

    # distance_mean
    data_distance = v.mean() - 0
    result['mean'] = data_distance

    # distance_variance
    mean_variance = pow(v.std(ddof=1), 2) / n

    # t_quantile
    degrees_of_freedom = n - 1
    result['df'] = degrees_of_freedom

    # confidence interval
    t_quantile = t.ppf(1 - alpha / 2, df=degrees_of_freedom)
    lower = data_distance - t_quantile * np.sqrt(mean_variance)
    upper = data_distance + t_quantile * np.sqrt(mean_variance)
    # confidence interval for alternative "greater"
    if alternative:
        t_quantile = t.ppf(1 - alpha, degrees_of_freedom)
        lower = data_distance - t_quantile * np.sqrt(mean_variance)
        upper = np.inf
    result['confidence_interval'] = [lower, upper]

    # p-value
    t_stat = data_distance / np.sqrt(mean_variance)
    result['t'] = t_stat
    p_value = (1 - t.cdf(t_stat, df=degrees_of_freedom)) * 2
    # p-value for alternative "greater"
    if alternative == "greater":
        p_value /= 2
    result['p-value'] = p_value

    return result


def test_t_test():
    v1 = np.array([9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19])
    v2 = np.array([7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9])

    # Two sample unpaired t-test
    pprint(t_test(v1, v2, paired=False))  # two-tailed
    pprint(t_test(v1, v2, paired=False, alternative="greater"))  # one-tailed

    # Two sample paired t-test
    pprint(t_test(v1, v2, paired=True))
    pprint(t_test(v1, v2, paired=True, alternative="greater"))

    # One sample t-test
    pprint(t_test(v1))
    pprint(t_test(v1, alternative="greater"))


def test_bootstrap():
    with NumpyRNGContext(2019):
        boot_result = bootstrap(np.array([[1, 1], [2, 2], [3, 3]]), bootnum=2)
        print(boot_result)
    with NumpyRNGContext(2019):
        boot_result = bootstrap(
            np.array([[1, 1], [2, 2], [3, 3]]), bootnum=2, bootfunc=np.mean)
        # print(boot_result)
    with NumpyRNGContext(2019):
        boot_result = bootstrap(
            np.array([[1, 1], [2, 2], [3, 3]]), bootnum=2, bootfunc=np.mean, axis=0)
        print(boot_result)

    df = pd.DataFrame(np.array([[1, 1], [2, 2], [3, 3]]), columns=[
                      "first", "second"])
    print(df)
    with NumpyRNGContext(2019):
        print(df_bootstrap(df, bootnum=2, bootfunc=np.mean, axis=0))


def test_bootstrap_ci():
    v1 = np.array([9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19])
    v2 = np.array([7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9])

    x1 = np.array([8, 24, 7, 20, 6, 20, 13, 15, 11, 22, 15])
    x2 = np.array([5, 11, 0, 15, 0, 20, 15, 19, 12, 0, 6])

    diff = x1 - x2
    diff_v = v1 - v2
    result = bootstrap_ci(diff, bootnum=100000, bootfunc=np.mean, alpha=0.05)
    print({'py_ci': result['ci'], 'r_ci': "[1.364,  9.818]"})
    result_v = bootstrap_ci(diff_v, bootnum=100000,
                            bootfunc=np.mean, alpha=0.05)
    print({'py_ci': result_v['ci'], 'r_ci': "[0.701,  3.266]"})


def bootstrap_two_sample_t_test_unpaired(v1, v2, alpha=0.05, alternative=None, bootnum=599, csv_file=None, seed=2019):
    """
    # Bootstrap unpaired t-test as described in Efron and Tibshirani (1993), (Algorithm 16.2, p224)

    # In python:
    v1 = np.array([9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19])
    v2 = np.array([7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9])
    # same as print(bootstrap_two_sample_t_test_unpaired(v1, v2, bootnum=100000))
    print(bootstrap_two_sample_t_test_unpaired(v1, v2, bootnum=100000, alternative="two.sided"))
    print(bootstrap_two_sample_t_test_unpaired(v1, v2, bootnum=100000, alternative="greater"))
    print(bootstrap_two_sample_t_test_unpaired(v1, v2, bootnum=100000, alternative="less"))

    # In R:
    library("nonpar")
    v1=c(9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19)
    v2=c(7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9)

    set.seed(2019)
    boot.t.test(x = v1, y = v2, reps = 100000)
    set.seed(2019)
    boot.t.test(x = v1, y = v2, reps = 100000, alternative="greater")
    set.seed(2019)
    boot.t.test(x = v1, y = v2, reps = 100000, alternative="less")
    """
    assert len(v1.shape) == 1 and len(
        v2.shape) == 1, "only support the one dimension numpy array"
    assert alternative == None or alternative == "greater" or alternative == "less" or alternative == "two.sided",\
        "alternative have to be empty(default, same as two.sided) or greater or less or two.sided"

    result = {}
    if not alternative or alternative == "two.sided":
        result['title'] = "Bootstrap Welch two sample t-test (unpaired, two tailed)"
    else:
        result['title'] = "Bootstrap Welch two sample t-test (unpaired, one tailed)"

    if not alternative or alternative == "two.sided":
        result['alternative_hypothesis'] = "true difference in means is not equal to 0"
    elif alternative == "greater":
        result['alternative_hypothesis'] = "true difference in means is greater than 0"
    else:
        result['alternative_hypothesis'] = "true difference in means is less than 0"

    result['significance_level'] = alpha

    def two_sample_common_mean_normalization(v1, v2):
        # Simple normalization suggested by Efron and Tibshirani (1993) (Section 16.2, p223)
        # in order to acheive the assumption of a common mean
        # Note we compute observered statistics before the normalization
        # We apply normalization for get empirical distribution obeys the null hypothesis assumption
        # See Efron and Tibshirani (1993) (Section 16.4, p226)
        common_mean = (v1.sum() + v2.sum()) / (len(v1) + len(v2))
        v1 = v1 - v1.mean() + common_mean
        v2 = v2 - v2.mean() + common_mean
        return v1, v2

    def t_stat_two_samples_with_pooled_variance(mean1, mean2, var1, var2, n1, n2):
        return (mean1 - mean2) / np.sqrt(var1 / n1 + var2 / n2)

    t_stat_observed = t_stat_two_samples_with_pooled_variance(
        v1.mean(), v2.mean(), v1.var(ddof=1), v2.var(ddof=1), len(v1), len(v2))
    result['t_obs'] = t_stat_observed
    v1, v2 = two_sample_common_mean_normalization(v1, v2)

    with NumpyRNGContext(seed):
        boot_mean_var_len1 = bootstrap(v1, bootnum=bootnum, bootfunc=lambda v: np.array([
                                       v.mean(), v.var(ddof=1), len(v)]))
        boot_mean_var_len2 = bootstrap(v2, bootnum=bootnum, bootfunc=lambda v: np.array([
                                       v.mean(), v.var(ddof=1), len(v)]))

    df = pd.DataFrame(np.concatenate([boot_mean_var_len1, boot_mean_var_len2], axis=1),
                      columns=['mean1', 'var1', 'len1', 'mean2', 'var2', 'len2'])
    boot_t_stat_pooled = t_stat_two_samples_with_pooled_variance(
        df['mean1'], df['mean2'], df['var1'], df['var2'], df['len1'], df['len2']).values
    result['t'] = boot_t_stat_pooled

    both = len(boot_t_stat_pooled[np.abs(
        boot_t_stat_pooled) >= np.abs(t_stat_observed)])
    greater = len(boot_t_stat_pooled[boot_t_stat_pooled >= t_stat_observed])
    less = len(boot_t_stat_pooled[boot_t_stat_pooled <= t_stat_observed])
    total = len(boot_t_stat_pooled)
    if not alternative or alternative == "two.sided":
        p_value = both / total
    elif alternative == "greater":
        p_value = greater / total
    else:
        p_value = less / total
    result['p-value'] = p_value
    return result


def test_bootstrap_two_sample_t_test_unpaired():
    v1 = np.array([9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19])
    v2 = np.array([7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9])
    # same as print(bootstrap_two_sample_t_test_unpaired(v1, v2, bootnum=100000))
    pprint(bootstrap_two_sample_t_test_unpaired(
        v1, v2, bootnum=100000, alternative="two.sided"))
    pprint(bootstrap_two_sample_t_test_unpaired(
        v1, v2, bootnum=100000, alternative="greater"))
    pprint(bootstrap_two_sample_t_test_unpaired(
        v1, v2, bootnum=100000, alternative="less"))


def bootstrap_two_sample_t_test_paired(v1, v2, alpha=0.05, alternative=None, bootnum=599, csv_file=None, seed=2019):
    """
    # Bootstrap paired t-test as described in Efron and Tibshirani (1993), (Section 16.4, p225)

    # In python:
    v1 = np.array([9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19])
    v2 = np.array([7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9])
    # same as print(bootstrap_two_sample_t_test_unpaired(v1, v2, bootnum=100000))
    pprint(bootstrap_two_sample_t_test_paired(v1, v2, bootnum=100000, alternative="two.sided"))
    pprint(bootstrap_two_sample_t_test_paired(v1, v2, bootnum=100000, alternative="greater"))
    pprint(bootstrap_two_sample_t_test_paired(v1, v2, bootnum=100000, alternative="less"))
    """
    assert len(v1.shape) == 1 and len(
        v2.shape) == 1, "only support the one dimension numpy array"
    assert alternative == None or alternative == "greater" or alternative == "less" or alternative == "two.sided",\
        "alternative have to be empty(default, same as two.sided) or greater or less or two.sided"
    assert len(v1) == len(
        v2), "We are using paired T-test, the number of samples are same"

    result = {}
    if not alternative or alternative == "two.sided":
        result['title'] = "Bootstrap paired t-test (paired, two tailed)"
    else:
        result['title'] = "Bootstrap paired t-test (paired, one tailed)"

    if not alternative or alternative == "two.sided":
        result['alternative_hypothesis'] = "true difference in means is not equal to 0"
    elif alternative == "greater":
        result['alternative_hypothesis'] = "true difference in means is greater than 0"
    else:
        result['alternative_hypothesis'] = "true difference in means is less than 0"

    result['significance_level'] = alpha

    def one_sample_normalization(v):
        # Note we compute observered statistics before the normalization
        # We apply normalization for get empirical distribution obeys the null hypothesis assumption
        # See Efron and Tibshirani (1993) (Formula 16.16, p226)
        return v - v.mean()

    def t_stat_one_sample(mean, var, n):
        return mean / np.sqrt(var / n)

    def t_stat_one_sample_ndarray(v):
        return t_stat_one_sample(v.mean(), v.var(ddof=1), len(v))

    v = v1 - v2
    t_stat_observed = t_stat_one_sample_ndarray(v)
    result['t_obs'] = t_stat_observed

    v = one_sample_normalization(v)
    with NumpyRNGContext(seed):
        boot_t_stat = bootstrap(
            v, bootnum, bootfunc=lambda vec: t_stat_one_sample_ndarray(vec))
    result['t'] = boot_t_stat

    both = len(boot_t_stat[np.abs(boot_t_stat) >= np.abs(t_stat_observed)])
    greater = len(boot_t_stat[boot_t_stat >= t_stat_observed])
    less = len(boot_t_stat[boot_t_stat <= t_stat_observed])
    total = len(boot_t_stat)
    if not alternative or alternative == "two.sided":
        p_value = both / total
    elif alternative == "greater":
        p_value = greater / total
    else:
        p_value = less / total
    result['p-value'] = p_value
    return result


def test_bootstrap_two_sample_t_test_paired():
    v1 = np.array([9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19])
    v2 = np.array([7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9])
    # same as print(bootstrap_two_sample_t_test_paired(v1, v2, bootnum=100000))
    pprint(bootstrap_two_sample_t_test_paired(
        v1, v2, bootnum=100000, alternative="two.sided"))
    pprint(bootstrap_two_sample_t_test_paired(
        v1, v2, bootnum=100000, alternative="greater"))
    pprint(bootstrap_two_sample_t_test_paired(
        v1, v2, bootnum=100000, alternative="less"))


def bootstrap_one_sample_t_test(v, alpha=0.05, alternative=None, bootnum=599, csv_file=None, seed=2019):
    """
    # One sample t-test shares the same implementation of 
    # Bootstrap paired t-test as described in Efron and Tibshirani (1993), (Section 16.4, p225)

    # In python:
    v1 = np.array([9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19]) 
    v2 = np.array([7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9])
    # same as print(bootstrap_one_sample_t_test(v1, v2, bootnum=100000))
    pprint(bootstrap_one_sample_t_test(v1, bootnum=100000, alternative="two.sided"))
    pprint(bootstrap_one_sample_t_test(v1, bootnum=100000, alternative="greater"))
    pprint(bootstrap_one_sample_t_test(v1, bootnum=100000, alternative="less"))
    """
    assert len(v.shape) == 1, "only support the one dimension numpy array"
    assert alternative == None or alternative == "greater" or alternative == "less" or alternative == "two.sided",\
        "alternative have to be empty(default, same as two.sided) or greater or less or two.sided"

    result = {}
    if not alternative or alternative == "two.sided":
        result['title'] = "Bootstrap one sample t-test (two tailed)"
    else:
        result['title'] = "Bootstrap one sample t-test (one tailed)"

    if not alternative or alternative == "two.sided":
        result['alternative_hypothesis'] = "true mean is not equal to 0"
    elif alternative == "greater":
        result['alternative_hypothesis'] = "true mean is greater than 0"
    else:
        result['alternative_hypothesis'] = "true mean is less than 0"

    result['significance_level'] = alpha

    def one_sample_normalization(v):
        # Note we compute observered statistics before the normalization
        # We apply normalization for get empirical distribution obeys the null hypothesis assumption
        # See Efron and Tibshirani (1993) (Formula 16.16, p226)
        return v - v.mean()

    def t_stat_one_sample(mean, var, n):
        return mean / np.sqrt(var / n)

    def t_stat_one_sample_ndarray(v):
        return t_stat_one_sample(v.mean(), v.var(ddof=1), len(v))

    t_stat_observed = t_stat_one_sample_ndarray(v)
    result['t_obs'] = t_stat_observed

    v = one_sample_normalization(v)
    with NumpyRNGContext(seed):
        boot_t_stat = bootstrap(
            v, bootnum, bootfunc=lambda vec: t_stat_one_sample_ndarray(vec))
    result['t'] = boot_t_stat

    both = len(boot_t_stat[np.abs(boot_t_stat) >= np.abs(t_stat_observed)])
    greater = len(boot_t_stat[boot_t_stat >= t_stat_observed])
    less = len(boot_t_stat[boot_t_stat <= t_stat_observed])
    total = len(boot_t_stat)
    if not alternative or alternative == "two.sided":
        p_value = both / total
    elif alternative == "greater":
        p_value = greater / total
    else:
        p_value = less / total
    result['p-value'] = p_value
    return result


def test_bootstrap_one_sample_t_test():
    # v1 = np.array([9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19])
    # v2 = np.array([7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9])
    v1 = np.array([-9.21, -11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19])
    # same as print(bootstrap_one_sample_t_test(v1, v2, bootnum=100000))
    pprint(bootstrap_one_sample_t_test(
        v1, bootnum=100000, alternative="two.sided"))
    pprint(bootstrap_one_sample_t_test(
        v1, bootnum=100000, alternative="greater"))
    pprint(bootstrap_one_sample_t_test(v1, bootnum=100000, alternative="less"))


def test_bootstrap_t_test():
    v1 = np.array([9.21, 11.51, 12.79, 11.85, 9.97, 8.79, 9.69, 9.68, 9.19])
    v2 = np.array([7.53, 7.48, 8.08, 8.09, 10.15, 8.4, 10.88, 6.13, 7.9])
    # Two sample unpaired t-test with bootstrap
    pprint(bootstrap_t_test(v1, v2, paired=False, bootnum=100000))  # two-tailed
    pprint(bootstrap_t_test(v1, v2, paired=False,
                            alternative="greater", bootnum=100000))  # one-tailed
    pprint(bootstrap_t_test(v1, v2, paired=False,
                            alternative="less", bootnum=100000))  # one-tailed
    # Two sample paired t-test with bootstrap
    pprint(bootstrap_t_test(v1, v2, paired=True, bootnum=100000))
    pprint(bootstrap_t_test(v1, v2, paired=True,
                            alternative="greater", bootnum=100000))
    pprint(bootstrap_t_test(v1, v2, paired=True,
                            alternative="less", bootnum=100000))
    # One sample t-test with bootstrap
    pprint(bootstrap_t_test(v1, bootnum=100000))
    pprint(bootstrap_t_test(v1, alternative="greater", bootnum=100000))
    pprint(bootstrap_t_test(v1, alternative="less", bootnum=100000))

# test_t_test()
# # test_bootstrap()
# test_bootstrap_ci()
# # test_bootstrap_two_sample_t_test_unpaired()
# # test_bootstrap_two_sample_t_test_paired()
# # test_bootstrap_one_sample_t_test()
# test_bootstrap_t_test()
