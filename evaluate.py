from time import perf_counter
import pandas as pd
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
    var_names = [str(v) for v in np.arange(1, n_vars + 1)]
    edges = np.array([[], []])  # Edges going from first row to second row
    cpts = {}

    for i, var in enumerate(var_names):
        # Creating the cpt for var -> depends on the parents -> find parents by looking through edges
        parents = edges[0, np.where(edges[1] == var)].ravel() if edges.size > 0 else []
        new_cpt = init_factor(list(np.append(parents, var)))
        # Generate uniform random p-values for each True/False pair
        x = np.around(np.random.rand(int(len(new_cpt) / 2)), decimals=2)
        new_cpt.loc[::2, 'p'] = x
        new_cpt.loc[1::2, 'p'] = 1-x
        cpts[var] = new_cpt

        # Choose the amount of children - between 1 and (n_vars - current_var - previously selected vars).
        # Uses the pareto distribution, resulting in roughly a 20/80 split of low/high amounts of children.
        min_n_children = 1
        n_children = min(int(np.around(np.random.pareto(a=1) + min_n_children)), n_vars - 1 - i)
        # Choose from var_names[i+1:], meaning ignore the current and previously selected vars, to avoid cycles.
        children = np.random.choice(var_names[i + 1:], size=n_children, replace=False)
        new_edges = np.vstack((np.repeat(var, repeats=n_children), children))
        edges = np.hstack((edges, new_edges))

    bn = BayesNet()
    bn.create_bn(var_names, list(map(tuple, edges.T)), cpts)
    return bn


def run_queries(bnr: BNReasoner, n_queries: int, order: str) -> float:
    """
    Run a number of randomly generated MPE & MAP queries on a BayesNet, using a specific order.
    :param bnr:         BNReasoner object, containing a BayesNet
    :param n_queries:   Number of queries to be generated and performed
    :param order:       String describing which type of ordering method MPE & MAP should use
    :return:            The average amount of time taken in seconds
    """
    timings = np.zeros(n_queries)
    for q in range(n_queries):
        variables = bnr.bn.get_all_variables()
        e = pd.Series({}, dtype=str)
        for v in np.random.choice(variables, 2, replace=False):
            e = e.append(pd.Series({v: np.random.choice([True, False])}))
        m = np.random.choice(np.setdiff1d(variables, e.keys()), 2)

        t_r = perf_counter()
        bnr.MPE(e, order)
        bnr.MAP(m, e, order)
        timings[q] = perf_counter() - t_r
        # timings[q] = np.random.randint(0, 10)  # test data
    return np.average(timings)


def color_bp(boxplot, fill_colors):
    for patch, color in zip(boxplot['boxes'], fill_colors):
        patch.set_facecolor(color)


def main():
    # np.random.seed(1)
    n_experiments = 3  # Amount of different graph sizes we experiment on
    begin_size = 5
    interval = 5
    graph_sizes = np.arange(begin_size, begin_size + interval * n_experiments, interval)

    n_graphs = 5  # Number of graphs per graph size
    n_queries = 1  # Number of queries per graph

    res_df = pd.DataFrame(columns=graph_sizes)
    results = pd.DataFrame(
        {'timings': {'random': res_df.copy(), 'min_degree': res_df.copy(), 'min_fill': res_df.copy()},
         'widths': {'random': res_df.copy(), 'min_degree': res_df.copy(), 'min_fill': res_df.copy()}})

    for graph_size in graph_sizes:  # For each graph size...
        print(f'Graphs with {graph_size} variables...')
        for n in range(n_graphs):   # create n_graphs variations with the same graph size
            bnr = BNReasoner(generate_BN(graph_size))
            print(f'\r    Evaluating # {n+1}/{n_graphs}...', end='')
            results.loc['random', 'timings'].loc[n, graph_size] = run_queries(bnr, n_queries, 'random')
            results.loc['random', 'widths'].loc[n, graph_size] = bnr.order_width(bnr.random_order())
            
            results.loc['min_degree', 'timings'].loc[n, graph_size] = run_queries(bnr, n_queries, 'min_degree')
            results.loc['min_degree', 'widths'].loc[n, graph_size] = bnr.order_width(bnr.min_degree_order())

            results.loc['min_fill', 'timings'].loc[n, graph_size] = run_queries(bnr, n_queries, 'min_fill')
            results.loc['min_fill', 'widths'].loc[n, graph_size] = bnr.order_width(bnr.min_fill_order())
        print('    Done!')
    # save results
    with open('timings_results.txt', 'a') as file:
        file.write('\nrandom\n')
        file.write(results['timings']['random'].to_csv(index=False))
        file.write('\nmin_degree\n')
        file.write(results['timings']['min_degree'].to_csv(index=False))
        file.write('\nmin_fill\n')
        file.write(results['timings']['min_fill'].to_csv(index=False))
    with open('widths_results.txt', 'a') as file:
        file.write('\nrandom\n')
        file.write(results['widths']['random'].to_csv(index=False))
        file.write('\nmin_degree\n')
        file.write(results['widths']['min_degree'].to_csv(index=False))
        file.write('\nmin_fill\n')
        file.write(results['widths']['min_fill'].to_csv(index=False))

    plot(results, n_experiments, graph_sizes)


def plot(results, n_experiments, graph_sizes):
    colors = ['pink', 'lightblue', 'lightgreen']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    fig.suptitle(f'Results Evaluation BNReasoner')

    bp1a = ax1.boxplot(results.loc['random', 'timings'],
                       positions=np.array(range(n_experiments)) * 3 - 0.6, sym='', patch_artist=True, widths=0.5)
    [patch.set_facecolor(colors[0]) for patch in bp1a['boxes']]
    bp1b = ax1.boxplot(results.loc['min_degree', 'timings'],
                       positions=np.array(range(n_experiments)) * 3, sym='', patch_artist=True, widths=0.5)
    [patch.set_facecolor(colors[1]) for patch in bp1b['boxes']]
    bp1c = ax1.boxplot(results.loc['min_fill', 'timings'],
                       positions=np.array(range(n_experiments)) * 3 + 0.6, sym='', patch_artist=True, widths=0.5)
    [patch.set_facecolor(colors[2]) for patch in bp1c['boxes']]
    ax1.set(title='Time taken', xlabel='# variables', ylabel='Duration (s)', xticks=range(0, n_experiments * 3, 3),
            xticklabels=graph_sizes, yscale='log')
    ax1.legend([bp1a['boxes'][0], bp1b['boxes'][0], bp1c['boxes'][0]], ['Random', 'Min Degree', 'Min Fill'])

    bp2a = ax2.boxplot(results.loc['random', 'widths'],
                       positions=np.array(range(n_experiments)) * 3 - 0.6, sym='', patch_artist=True, widths=0.5)
    [patch.set_facecolor(colors[0]) for patch in bp2a['boxes']]
    bp2b = ax2.boxplot(results.loc['min_degree', 'widths'],
                       positions=np.array(range(n_experiments)) * 3, sym='', patch_artist=True, widths=0.5)
    [patch.set_facecolor(colors[1]) for patch in bp2b['boxes']]
    bp2c = ax2.boxplot(results.loc['min_fill', 'widths'],
                       positions=np.array(range(n_experiments)) * 3 + 0.6, sym='', patch_artist=True, widths=0.5)
    [patch.set_facecolor(colors[2]) for patch in bp2c['boxes']]
    ax2.set(title='Widths', xlabel='# variables', ylabel='Ordering width', xticks=range(0, n_experiments * 3, 3),
            xticklabels=graph_sizes)
    ax2.legend([bp2a['boxes'][0], bp2b['boxes'][0], bp2c['boxes'][0]], ['Random', 'Min Degree', 'Min Fill'])

    plt.savefig('plot.png')
    plt.show()


if __name__ == '__main__':
    main()
