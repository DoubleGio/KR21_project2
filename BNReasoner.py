from typing import Union, List
from itertools import product, combinations
import pandas as pd
import numpy as np
from copy import deepcopy
from BayesNet import BayesNet


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    # TODO: This is where your methods should go

    def sum_out_factors(self, factor: Union[str, pd.DataFrame], subset: Union[str, list]) -> pd.DataFrame:
        if isinstance(factor, str):
            factor = self.bn.get_cpt(factor)
        if isinstance(subset, str):
            subset = [subset]

        variables = [v for v in factor.keys() if v not in subset + ['p']]  # Set subtraction: Factor - Subset - p
        new_factor = self.init_factor(variables, 0)
        subset_factor = self.init_factor(subset, 0)

        for i, y in new_factor.iterrows():
            for _, z in subset_factor.iterrows():
                new_factor.loc[i, 'p'] = new_factor.loc[i, 'p'] + self.bn.get_compatible_instantiations_table(
                    y[:-1].append(z[:-1]), factor)['p'].sum()

        return new_factor

    def multiply_factors(self, factors: List[Union[str, pd.DataFrame]]) -> pd.DataFrame:
        # If there are strings in the input-list of factors, replace them with the corresponding cpt
        for x, y in enumerate(factors):
            if isinstance(y, str):
                factors[x] = self.bn.get_cpt(y)

        variables = list(set().union(*factors))
        variables.remove('p')  # Remove 'p' col to add it again in the next step, ensuring it ends up as the last col
        new_factor = self.init_factor(variables, 1)

        for i, z in new_factor.iterrows():
            for _, f in enumerate(factors):
                new_factor.loc[i, 'p'] = new_factor.loc[i, 'p'] * self.bn.get_compatible_instantiations_table(
                    z[:-1], f)['p'].sum()

        return new_factor

    def compute_marginal(self, query: List[str], evidence: pd.Series = None, order: List[str] = None) -> pd.DataFrame:
        S = deepcopy(self.bn)
        if evidence is not None:
            for v in self.bn.get_all_variables():
                cpt = self.bn.get_cpt(v)
                cpt_e = self.bn.get_compatible_instantiations_table(evidence, cpt)
                S.update_cpt(v, cpt_e)

        order = self.min_degree_order()
        pi = [nv for nv in order if nv not in query]
        for ele in pi:
            func_k = [f for f in S.get_all_cpts().values() if ele in f]
            factor = self.multiply_factors(func_k)
            # sum out enzo

        # if evidence, normalize door te delen door Pr(evidence)

    def min_degree_order(self) -> List[str]:
        G = self.bn.get_interaction_graph()
        order = []
        for i in range(len(self.bn.get_all_variables())):
            min_degree_var = ""
            min_n_neighbors = np.inf
            min_neighbors = []

            for node in G.nodes:
                neighbors = []
                n = 0
                for neighbor in G.neighbors(node):
                    neighbors.append(neighbor)
                    n += 1
                if n < min_n_neighbors:
                    min_degree_var = node
                    min_n_neighbors = n
                    min_neighbors = neighbors

            # n_neighbors = [sum(1 for _ in G.neighbors(node)) for node in G.nodes]
            # min_degree_var = list(G.nodes)[np.argmin(n_neighbors)]
            order.append(min_degree_var)

            if min_n_neighbors > 1:
                for pair in combinations(min_neighbors, 2):
                    G.add_edge(pair[0], pair[1])

            G.remove_node(min_degree_var)

        return order


    @staticmethod
    def init_factor(variables: list, value=0) -> pd.DataFrame:
        truth_table = product([True, False], repeat=len(variables))
        factor = pd.DataFrame(truth_table, columns=variables)
        factor['p'] = value
        return factor


def main():
    # bnr = BNReasoner('testing/lecture_example.BIFXML')
    # a = bnr.sum_out_factors('Wet Grass?', 'Wet Grass?')
    # print(a)
    #
    # bnr2 = BNReasoner('testing/multiply_example.BIFXML')
    # b = bnr2.multiply_factors(['D', 'E'])
    # print(b)
    bnr = BNReasoner('testing/lecture_example.BIFXML')
    bnr.compute_marginal(['Wet Grass?', 'Slippery Road?'], pd.Series({"Winter?": True}))


if __name__ == '__main__':
    main()
