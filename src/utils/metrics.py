"""Information Retrieval metrics
Useful Resources:
http://www.cs.utexas.edu/~mooney/ir-course/slides/Evaluation.ppt
http://www.nii.ac.jp/TechReports/05-014E.pdf
http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
http://hal.archives-ouvertes.fr/docs/00/72/67/60/PDF/07-busa-fekete.pdf
Learning to Rank for Information Retrieval (Tie-Yan Liu)
"""
import numpy as np
from statistics import mean

def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        reciprocal rank
    """
    rs = np.asarray(rs).nonzero()[0]
    return 1. / (rs[0] + 1) if rs.size else 0.


def r_precision(r):
    """Score is precision after all relevant documents have been retrieved
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> r_precision(r)
    0.33333333333333331
    >>> r = [0, 1, 0]
    >>> r_precision(r)
    0.5
    >>> r = [1, 0, 0]
    >>> r_precision(r)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        R Precision
    """
    r = np.asarray(r) != 0
    z = r.nonzero()[0]
    if not z.size:
        return 0.
    return np.mean(r[:z[-1] + 1])


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


# fairness metric
def absolute_unfairness(df_group_0, df_group_1):
    group_0_iid_set = set(df_group_0['iid'].unique())
    group_1_iid_set = set(df_group_1['iid'].unique())
    iid_list = list(group_0_iid_set & group_1_iid_set)
    diff = []
    for iid in iid_list:
        group_0_iid_res = df_group_0[df_group_0['iid'] == iid]
        group_1_iid_res = df_group_1[df_group_1['iid'] == iid]
        group_0_iid_val = abs(mean(list(group_0_iid_res['score'])) - mean(list(group_0_iid_res['label'])))
        group_1_iid_val = abs(mean(list(group_1_iid_res['score'])) - mean(list(group_1_iid_res['label'])))
        diff.append(abs(group_0_iid_val - group_1_iid_val))
    return mean(diff)

# fairness metric
def value_unfairness(df_group_0, df_group_1):
    group_0_iid_set = set(df_group_0['iid'].unique())
    group_1_iid_set = set(df_group_1['iid'].unique())
    iid_list = list(group_0_iid_set & group_1_iid_set)
    diff = []
    for iid in iid_list:
        group_0_iid_res = df_group_0[df_group_0['iid'] == iid]
        group_1_iid_res = df_group_1[df_group_1['iid'] == iid]
        group_0_iid_val = mean(list(group_0_iid_res['score'])) - mean(list(group_0_iid_res['label']))
        group_1_iid_val = mean(list(group_1_iid_res['score'])) - mean(list(group_1_iid_res['label']))
        diff.append(abs(group_0_iid_val - group_1_iid_val))
    return mean(diff)

def user_oriented_unfairness(df_group_0, df_group_1, metric):

    k = int(metric.split('@')[-1])
    df_group_0 = df_group_0.sort_values(by='score', ascending=False)
    df_group_0 = df_group_0.groupby(USER)
    if metric.startswith('ndcg@'):
        ndcgs = []
        for uid, group in df_group_0:
            ndcgs.append(ndcg_at_k(group['label'].tolist(), k=k, method=1))
        ndcg_group_0 = np.average(ndcgs)

    df_group_1 = df_group_1.sort_values(by='score', ascending=False)
    df_group_1 = df_group_1.groupby(USER)
    if metric.startswith('ndcg@'):
        ndcgs = []
        for uid, group in df_group_1:
            ndcgs.append(ndcg_at_k(group['label'].tolist(), k=k, method=1))
        ndcg_group_1 = np.average(ndcgs)


    return np.abs(ndcg_group_0 - ndcg_group_1)


import numpy as np

def calibrated_groupwise_utility(df_group_0, df_group_1, metric, c=0.1):
    """
    Compute the social welfare function based on the weighted sum of log-transformed ndcg values.

    Args:
        df_group_0 (DataFrame): DataFrame for group 0.
        df_group_1 (DataFrame): DataFrame for group 1.
        metric (str): The ranking metric (e.g., 'ndcg@10').
        c (float): A small constant to adjust magnitude, default is 0.1.

    Returns:
        float: The social welfare value.
    """
    k = int(metric.split('@')[-1])

    # Compute NDCG for group 0
    df_group_0 = df_group_0.sort_values(by='score', ascending=False)
    df_group_0 = df_group_0.groupby(USER)
    if metric.startswith('ndcg@'):
        ndcgs_0 = [ndcg_at_k(group['label'].tolist(), k=k, method=1) for uid, group in df_group_0]
        ndcg_group_0 = np.average(ndcgs_0)
    
    # Compute NDCG for group 1
    df_group_1 = df_group_1.sort_values(by='score', ascending=False)
    df_group_1 = df_group_1.groupby(USER)
    if metric.startswith('ndcg@'):
        ndcgs_1 = [ndcg_at_k(group['label'].tolist(), k=k, method=1) for uid, group in df_group_1]
        ndcg_group_1 = np.average(ndcgs_1)

    # Compute the proportions of each group
    total_users = len(df_group_0) + len(df_group_1)
    proportion_0 = len(df_group_0) / total_users
    proportion_1 = len(df_group_1) / total_users

    # Apply the social welfare function
    welfare = (proportion_0 * np.log(ndcg_group_0 - c)) + (proportion_1 * np.log(ndcg_group_1 - c))

    return welfare