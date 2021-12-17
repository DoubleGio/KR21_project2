from time import perf_counter
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from BayesNet import BayesNet
from BNReasoner import BNReasoner, init_factor


def generate_BN(n_vars: int) -> BayesNet:
    """
    Generate a Bayesian Network with n_vars variables, a random number of edges and random pr-values.
    :param n_vars:  integer specifying the number of variables in the network
    :return:        a BayesNet instance
    """
    var_names = [str(v) for v in np.arange(1, n_vars+1)]
    edges = []
    cpts = {}

    for i, var in enumerate(var_names):
        # Creating the cpt for var -> depends on the parents -> find parents by looking through edges
        parents = [p for p, c in edges if c == var]
        new_cpt = init_factor(list(np.append(parents, var)))
        # Generate uniform random p-values for each True/False pair
        for j in range(0, len(new_cpt), 2):
            x = np.around(np.random.rand(), decimals=2)
            new_cpt.loc[j, 'p'] = x
            new_cpt.loc[j + 1, 'p'] = 1 - x
        cpts[var] = new_cpt

        # Choose the amount of children - between 1 and (n_vars - current_var - previously selected vars).
        # Uses the pareto distribution, resulting in roughly a 20/80 split of low/high amounts.
        min_n_children = 1
        n_children = min(int(np.around(np.random.pareto(a=1) + min_n_children)), n_vars - 1 - i)
        # Choose from var_names[i+1:], meaning ignore the current and previously selected vars, to avoid cycles.
        children = np.random.choice(var_names[i+1:], size=n_children, replace=False)
        for child in children:
            edge = (var, child)
            edges.append(edge)

    bn = BayesNet()
    bn.create_bn(var_names, edges, cpts)
    return bn


def main():
    np.random.seed(1)
    generate_BN(10)


if __name__ == '__main__':
    main()
