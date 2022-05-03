import fairlearn.metrics as fairlearn_metrics
import numpy as np


def demographic_parity_difference_soft(y, c, y_hat):
    """
    works only for binary y
    This computes $|\sum_x p(y=1|x,c=1) - \sum_x p(y=1|x,c=0)|$

    This considers that the the output is probability.
    Other libraries have parity implementation that use predicted label (1/0) as input

    When c has more than two values i.e c in \{0, d-1\},
    this will evaluate absolute difference for every i,j in \{0,d-1\} and report the max.

    This is a crude generalization, and we also return the average. For only two values they will be
    the same. Max is also used by fairlearn's implementation.
    """
    c = c.reshape(-1)
    assert y_hat.shape[1] == 2
    assert y.shape[0] == c.shape[0]

    d = len(set(c))
    p_yhat_1_given_c = []
    for i in set(c):
        p_yhat_1_given_c.append(np.mean(y_hat[c == i, 1]))

    differences = []
    for j in range(d):
        differences.append([])
        for i in range(d):
            differences[j].append(np.abs(p_yhat_1_given_c[i] - p_yhat_1_given_c[j]))
    return np.max(differences), np.sum(differences) / (d * (d - 1))


def demographic_parity_difference(y, c, y_hat):
    """This will only return max, mean implementation is not there"""
    c = c.reshape(-1)
    assert y_hat.shape[1] == 2
    assert y.shape[0] == c.shape[0]


    y_pred = y_hat[:, 1] > 0.5

    return fairlearn_metrics.demographic_parity_difference(y, y_pred, sensitive_features=c), None



def demographic_parity_ratio_soft(y, c, y_hat):
    """
    works only for binary y
    This computes $|\sum_x p(y=1|x,c=1) / \sum_x p(y=1|x,c=0)|$ or its inverse

    This considers that the the output is probability.
    Other libraries have parity implementation that use predicted label (1/0) as input

    When c has more than two values i.e c in \{0, d-1\},
    this will evaluate the quantity for every i,j in \{0,d-1\} and report the max.

    This is a crude generalization, and we also return the average. For only two values they will be
    the same
    """

    c = c.reshape(-1)
    assert y_hat.shape[1] == 2
    assert y.shape[0] == c.shape[0]


    d = len(set(c))
    p_yhat_1_given_c = []
    for i in set(c):
        p_yhat_1_given_c.append(np.mean(y_hat[c == i, 1]))

    ratios = []
    for j in range(d):
        ratios.append([])
        for i in range(d):
            ratios[j].append(np.max([p_yhat_1_given_c[i] / p_yhat_1_given_c[j],
                                     p_yhat_1_given_c[j] / p_yhat_1_given_c[i]]))
    return np.max(ratios), (np.sum(ratios) - d) / (d * (d - 1))


def demographic_parity_ratio(y, c, y_hat):
    c = c.reshape(-1)
    assert y_hat.shape[1] == 2
    assert y.shape[0] == c.shape[0]


    y_pred = y_hat[:, 1] > 0.5
    return fairlearn_metrics.demographic_parity_ratio(y, y_pred, sensitive_features=c)


__all__ = ["demographic_parity_difference_soft", "demographic_parity_difference",
           "demographic_parity_ratio", "demographic_parity_ratio_soft"]

if __name__ == "__main__":
    y = np.array([0, 1, 0, 1])
    c = np.array([0, 0, 1, 1])
    prob = np.array([[0.6, 0.4], [0.4, 0.6], [0.3, 0.7], [0.3, 0.7]])
    max_delta_dp, mean_delta_dp = demographic_parity_difference_soft(y, c, prob)

    assert abs(max_delta_dp - 0.2) < 1e-4
    assert abs(mean_delta_dp - 0.2) < 1e-4

    max_ratio_dp, mean_ratio_dp = demographic_parity_ratio_soft(y, c, prob)
    assert abs(max_ratio_dp - 1.4) < 1e-4
    assert abs(mean_ratio_dp - 1.4) < 1e-4

    max_delta_dp, mean_delta_dp  = demographic_parity_difference(y, c, prob)

    assert (max_delta_dp- 0.5) < 1e-4
