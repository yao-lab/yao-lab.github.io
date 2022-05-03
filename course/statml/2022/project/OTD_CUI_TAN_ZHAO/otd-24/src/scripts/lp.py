import cvxpy
import numpy
from fairlearn.metrics import demographic_parity_difference

from src.common.data.adult import load_adult
from src.common.data.health import load_health


def get_optimal_front(Y, C):
    len_c = len(numpy.unique(C))
    len_y = len(numpy.unique(Y))
    p_y_c = numpy.zeros((len_y, len_c))

    for c in range(len_c):
        for y in range(len_y):
            p_y_c[y, c] = numpy.logical_and(Y == y, C == c).mean()
    # print(p_y_c)
    #
    # # compute desired rate i.e p(y=1|C=c)
    # desired_rate = p_y_c[1, :].mean()
    # errors = p_y_c[1, :] - desired_rate

    majority_acc = max(numpy.mean(Y == 1), 1 - numpy.mean(Y == 1))
    max_dp = demographic_parity_difference(Y, Y, sensitive_features=C)
    STEPS = 0.01

    solution = []
    for dp in numpy.arange(0, max_dp, STEPS):
        delta = cvxpy.Variable(p_y_c.shape[1])

        # delta is fraction flipped to 0
        objective = cvxpy.Maximize(1-cvxpy.sum(cvxpy.abs(delta)))
        constraints = []
        constraints.extend([-p_y_c[0, :] <= delta, delta <= p_y_c[1, :]])

        p_c = p_y_c.sum(axis=0)
        for i in range(p_y_c.shape[1]):
            for j in range(i+1,p_y_c.shape[1]):
                constraints.extend([
                    -dp <= (p_y_c[1, i] - delta[i]) / p_c[i] - (p_y_c[1, j] - delta[j]) / p_c[j],
                    (p_y_c[1, i] - delta[i]) / p_c[i] - (p_y_c[1, j] - delta[j]) / p_c[j] <= dp,
                ])
        prob = cvxpy.Problem(objective, constraints)
        result = prob.solve()
        # breakpoint()
        solution.append([result, dp])
        print(f"DP: {dp}, sol : {result}")

    return solution


if __name__ == "__main__":
    for data in ["adult", "health"]:
        # compute idea areas
        if data == "adult":
            adult = load_adult(0.2)
            Y = adult["test"][2]
            C = adult["test"][1]
        elif data == "health":
            health = load_health(0.2)
            Y = health["test"][2]
            C = health["test"][1]

        solution = get_optimal_front(Y, C)
