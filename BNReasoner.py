from typing import Union, List
from itertools import product, combinations
import pandas as pd
import numpy as np
from copy import deepcopy
import pandas.errors

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

    def d_separation(self, x: List[str], z: List[str], y: List[str]) -> bool:
        """
        Based on graph pruning.
        Given z, determine whether x is independent of y.
        :param x:   list of variables
        :param z:   list of variables
        :param y:   list of variables
        :return:    boolean, whether x is independent or not
        """
        query_variables = [x, y, z]
        query_variables = [item for sublist in query_variables for item in sublist]
        all_variables = self.bn.get_all_variables()
        non_instantiated_variables = list(set(all_variables) - set(query_variables))

        # iteratively removes leaf nodes that are not in the set of {x,y,z}
        counter = 0
        while counter <= len(non_instantiated_variables):

            if not non_instantiated_variables:
                break

            for var in non_instantiated_variables[:]:
                if not self.bn.get_children(var):
                    non_instantiated_variables.remove(var)
                    self.bn.del_var(var)
                    counter -= 1
                else:
                    counter += 1

        # remove outgoing edges from all nodes in Z
        for var in z:
            for child in self.bn.get_children(var):
                self.bn.del_edge((var, child))

        # X and Y are d-separated by Z in if X and Y are disconnected in the pruned graph w.r.t. Z
        # For each var in X, check if that variable reaches any of the variables in Y
        # -if this is the case, then they are not d-separated --> return False
        # -if this is not the case, then they are d-separated --> return True
        num_of_connections = 0
        for node in x:
            nodes = [node]
            visited = []

            while len(nodes) > 0:
                for var in nodes[:]:
                    visited.append(var)
                    for c in self.bn.get_children(var):
                        if c not in visited:
                            nodes.append(c)
                    for p in self.bn.get_parents(var):
                        if p not in visited:
                            nodes.append(p)
                    nodes.pop(0)

                if any(item in y for item in visited):
                    num_of_connections += 1
                    break

        # print('Num_of_connections: ' + str(num_of_connections))

        if num_of_connections == 0:
            return True
        else:
            return False

    def network_pruning(self, q: List[str], e: pd.Series, network: BayesNet = None) -> BayesNet:
        """
        :param network: network to be pruned
        :param q: set of query vars --> e.g. ['A','B','C']
        :param e: evidence set --> e.g. {'A': True, 'B': False}
        """
        if network is None:
            network = self.bn

        evidence = list(e.keys())
        query_plus_evidence_variables = [q, evidence]
        query_plus_evidence_variables = [item for sublist in query_plus_evidence_variables for item in sublist]
        all_variables = network.get_all_variables()
        non_instantiated_variables = list(set(all_variables) - set(query_plus_evidence_variables))

        # iteratively removes leaf nodes that are not in the set of {q,e}
        counter = 0
        while counter <= len(non_instantiated_variables):

            if not non_instantiated_variables:
                break

            for var in non_instantiated_variables[:]:
                if not network.get_children(var):
                    non_instantiated_variables.remove(var)
                    network.del_var(var)
                    counter -= 1
                else:
                    counter += 1

        # remove outgoing edges from all nodes in Z, and update CPTs
        for var in evidence:
            for child in network.get_children(var):
                network.del_edge((var, child))

                # new = self.bn.get_compatible_instantiations_table(pd.Series(e), self.bn.get_cpt(child))
                new = self.bn.get_compatible_instantiations_table(pd.Series({var: e.get(var)}), network.get_cpt(child))
                new = new.drop([var], axis=1)
                network.update_cpt(child, new.reset_index(drop=True))
        return network

    def sum_out_factors(self, factor: Union[str, pd.DataFrame], subset: Union[str, list]) -> pd.DataFrame:
        """
        Sum out some variable(s) in subset from a factor.
        :param factor:  factor over variables X
        :param subset:  a subset of variables X
        :return:        a factor corresponding to the factor with the subset summed out
        """
        if isinstance(factor, str):
            factor = self.bn.get_cpt(factor)
        if isinstance(subset, str):
            subset = [subset]

        new_factor = factor.drop(subset + ['p'], axis=1).drop_duplicates()
        new_factor['p'] = 0
        subset_factor = init_factor(subset)

        for i, y in new_factor.iterrows():
            for _, z in subset_factor.iterrows():
                new_factor.loc[i, 'p'] = new_factor.loc[i, 'p'] + self.bn.get_compatible_instantiations_table(
                    y[:-1].append(z[:-1]), factor)['p'].sum()
                # sum() instead of float() here, since the compatible table can be empty at times, this works around it

        return new_factor.reset_index(drop=True)

    def maximise_out(self, factor: Union[str, pd.DataFrame], subset: Union[str, list]) -> pd.DataFrame:
        """

        :param factor:
        :param subset:
        :return:
        """
        if isinstance(factor, str):
            factor = self.bn.get_cpt(factor)
        if isinstance(subset, str):
            subset = [subset]

        # Copy the factor, drop the variable(s) to be maximized, the extensions and 'p' and drop all duplicates to get
        # each possible instantiation.
        ext = [c for c in factor.columns if c[:3] == 'ext']
        instantiations = deepcopy(factor).drop(subset + ext + ['p'], axis=1).drop_duplicates()
        res_factor = pd.DataFrame(columns=factor.columns)

        if len(instantiations.columns) == 0:
            try:
                res_factor = res_factor.append(factor.iloc[factor['p'].idxmax()])
            except IndexError:
                print('w')
        else:
            for _, instantiation in instantiations.iterrows():
                cpt = self.bn.get_compatible_instantiations_table(instantiation, factor)
                res_factor = res_factor.append(factor.iloc[cpt['p'].idxmax()])

        # For each maximized-out variable(s), rename them to ext(variable)
        for v in subset:
            x = res_factor.pop(v)
            res_factor[f'ext({v})'] = x

        return res_factor.reset_index(drop=True)

    def multiply_factors(self, factors: List[Union[str, pd.DataFrame]]) -> pd.DataFrame:
        """
        Multiply multiple factors with each other.
        :param factors: a list of factors
        :return:        a factor corresponding to the product of all given factors
        """
        # If there are strings in the input-list of factors, replace them with the corresponding cpt
        for x, y in enumerate(factors):
            if isinstance(y, str):
                factors[x] = self.bn.get_cpt(y)

        new_factor = factors[0].drop('p', axis=1)
        for i, factor in enumerate(factors[1:]):
            try:
                new_factor = new_factor.merge(factor.drop('p', axis=1), how='outer')
            except pandas.errors.MergeError:
                new_factor = new_factor.join(factor.drop('p', axis=1), how='outer')
        new_factor['p'] = 1

        for i, z in new_factor.iterrows():
            for _, f in enumerate(factors):
                new_factor.loc[i, 'p'] = new_factor.loc[i, 'p'] * self.bn.get_compatible_instantiations_table(
                    z[:-1], f)['p'].sum()
                # sum() instead of float() here, since the compatible table can be empty at times, this works around it

        # Reordering new_factor, putting the extensions to the back
        cols = new_factor.columns
        ext = [c for c in cols if c[:3] == 'ext']
        if len(ext) > 0:
            rest = list(np.setdiff1d(cols, ext))
            new_factor = new_factor[rest + ext]

        return new_factor.reset_index(drop=True)

    def random_order(self, network: BayesNet = None) -> List[str]:
        """
        :return: a random ordering of all variables in self.bn
        """
        if network is None:
            return list(np.random.permutation(self.bn.get_all_variables()))
        else:
            return list(np.random.permutation(network.get_all_variables()))

    def min_degree_order(self, network: BayesNet = None) -> List[str]:
        """
        Orders variables in self.bn by choosing nodes wth the smallest degrees.
        :return: an ordering of all variables in self.bn
        """
        if network is None:
            G = self.bn.get_interaction_graph()
            variables = self.bn.get_all_variables()
        else:
            G = network.get_interaction_graph()
            variables = network.get_all_variables()
        order = []
        for i in range(len(variables)):
            degrees = dict(G.degree)  # Returns the degrees (number of connected edges) per node
            min_degree_var = min(degrees, key=degrees.get)
            min_neighbors = [neighbor for neighbor in G.neighbors(min_degree_var)]
            order.append(min_degree_var)

            if len(min_neighbors) > 1:
                for pair in combinations(min_neighbors, 2):
                    G.add_edge(pair[0], pair[1])
            G.remove_node(min_degree_var)

        return order

    def min_fill_order(self, network: BayesNet = None) -> List[str]:
        """
        Orders variables in self.bn by choosing nodes whose elimination adds smallest number of edges.
        :return: an ordering of all variables in self.bn
        """
        if network is None:
            G = self.bn.get_interaction_graph()
            variables = self.bn.get_all_variables()
        else:
            G = network.get_interaction_graph()
            variables = network.get_all_variables()
        order = []
        for i in range(len(variables)):
            min_var = ""
            min_added_edges = np.inf
            min_neighbors = []

            # For each node, count and remember the neighbors; append the node with the smallest count to order
            for node in G.nodes:
                neighbors = []
                p = 0
                for neighbor in G.neighbors(node):
                    neighbors.append(neighbor)

                for node_pair in combinations(neighbors, 2):
                    if not G.has_edge(node_pair[0], node_pair[1]):
                        p += 1
                if p < min_added_edges:
                    min_var = node
                    min_added_edges = p
                    min_neighbors = neighbors
            order.append(min_var)

            if min_added_edges > 1:
                for pair in combinations(min_neighbors, 2):
                    G.add_edge(pair[0], pair[1])
            G.remove_node(min_var)

        return order

    def order_width(self, order: List[str]) -> int:
        """
        Evaluates the quality of an ordering by returning its width (smaller = better).
        :param order:   an ordering of self.bn
        :return:        the the largest width in order
        """
        G = self.bn.get_interaction_graph()
        width = 0
        for var in order:
            var_degree = G.degree[var]
            width = max(width, var_degree)
            if var_degree > 1:
                for pair in combinations(G.neighbors(var), 2):
                    G.add_edge(pair[0], pair[1])
            G.remove_node(var)
        return width

    def compute_marginal(self, query: List[str], evidence: pd.Series = None, order: List[str] = None) -> pd.DataFrame:
        """
        Compute the prior marginal Pr(query) or joint/posterior marginal Pr(query, evidence).
        :param query:       variables in network N
        :param evidence:    optional; instantiation of some variables in network N
        :param order:       optional; ordering of variables in network N
        :return:            a factor describing the marginal
        """
        if order is None:
            order = self.random_order()

        S = self.bn.get_all_cpts()

        if evidence is not None:  # If there's evidence, reduce all CPTs using the evidence
            for var in self.bn.get_all_variables():
                var_cpt = self.bn.get_cpt(var)
                if any(evidence.keys().intersection(var_cpt.columns)):  # If the evidence occurs in the cpt
                    new_cpt = self.bn.get_compatible_instantiations_table(evidence, var_cpt)
                    S[var] = new_cpt

        pi = [nv for nv in order if nv not in query]
        for var_pi in pi:
            # Pop all functions from S, which mention var_pi...
            func_k = [S.pop(key) for key, cpt in deepcopy(S).items() if var_pi in cpt]

            new_factor = self.multiply_factors(func_k)
            new_factor = self.sum_out_factors(new_factor, var_pi)
            # And replace them with the new factor
            S[var_pi] = new_factor

        res_factor = self.multiply_factors(list(S.values())) if len(S) > 1 else S.popitem()[1]

        if evidence is not None:  # Normalizing over pr_evidence
            cpt_e = self.compute_marginal(list(evidence.keys()), order=order)
            pr_evidence = float(self.bn.get_compatible_instantiations_table(evidence, cpt_e)['p'])
            res_factor['p'] = res_factor['p'] / pr_evidence

        return res_factor

    def MPE(self, evidence: pd.Series, order_func: str = None) -> pd.DataFrame:
        """
        Compute the MPE instantiation for some given evidence.
        :param evidence:
        :param order_func:  String describing which order function to use
        :return:            Dataframe describing the MPE instantiation
        """
        N = deepcopy(self.bn)
        # Prune Edges
        for var in evidence.keys():
            for child in N.get_children(var):
                N.del_edge((var, child))

                new = N.get_compatible_instantiations_table(evidence, N.get_cpt(child)).reset_index(drop=True)
                new = new.drop([var], axis=1)
                N.update_cpt(child, new)
            u = N.get_compatible_instantiations_table(evidence, N.get_cpt(var)).reset_index(drop=True)
            N.update_cpt(var, u)

        if order_func is None or order_func == "random":
            order = self.random_order(N)
        elif order_func == "min_degree":
            order = self.min_degree_order(N)
        elif order_func == "min_fill":
            order = self.min_fill_order(N)
        else:
            raise Exception("Wrong order argument")

        S = N.get_all_cpts()
        for var_pi in order:
            # Pop all functions from S, which mention var_pi...
            func_k = [S.pop(key) for key, cpt in deepcopy(S).items() if var_pi in cpt]

            new_factor = self.multiply_factors(func_k) if len(func_k) > 1 else func_k[0]
            new_factor = self.maximise_out(new_factor, var_pi)
            # And replace them with the new factor
            S[var_pi] = new_factor

        res_factor = self.multiply_factors(list(S.values())) if len(S) > 1 else S.popitem()[1]
        return res_factor

    def MAP(self, M: List[str], evidence: pd.Series, order_func: str = None) -> pd.DataFrame:
        """
        Compute the MAP instantiation for some variables and some given evidence.
        :param M:           The MAP variables
        :param evidence:
        :param order_func:  String describing which order function to use
        :return:            Dataframe describing the MAP instantiation
        """
        if len(np.intersect1d(list(evidence.keys()), M)) > 0:
            raise Exception("Evidence cannot intersect with M")

        N = deepcopy(self.bn)
        N = self.network_pruning(M, evidence, N)
        for e in evidence.items():
            N.update_cpt(e[0], self.bn.get_compatible_instantiations_table(evidence,
                                                                           N.get_cpt(e[0])).reset_index(drop=True))

        if order_func is None or order_func == "random":
            order = self.random_order(N)
        elif order_func == "min_degree":
            order = self.min_degree_order(N)
        elif order_func == "min_fill":
            order = self.min_fill_order(N)
        else:
            raise Exception("Wrong order argument")

        S = N.get_all_cpts()
        for var_pi in order:
            # Pop all functions from S, which mention var_pi...
            func_k = [S.pop(key) for key, cpt in deepcopy(S).items() if var_pi in cpt]

            new_factor = self.multiply_factors(func_k) if len(func_k) > 1 else func_k[0]
            if var_pi in M:
                new_factor = self.maximise_out(new_factor, var_pi)
            else:
                new_factor = self.sum_out_factors(new_factor, var_pi)
            # And replace them with the new factor
            # print(new_factor)
            S[var_pi] = new_factor

        res_factor = self.multiply_factors(list(S.values())) if len(S) > 1 else S.popitem()[1]
        return res_factor


def init_factor(variables: List[str], value=0) -> pd.DataFrame:
    """
    Generate a default CPT.
    :param variables:   Column names
    :param value:       Which the default p-value should be
    :return:            A CPT
    """
    truth_table = product([True, False], repeat=len(variables))
    factor = pd.DataFrame(truth_table, columns=variables)
    factor['p'] = value
    return factor


def main():
    # Some examples
    bnr = BNReasoner('testing/lecture_example.BIFXML')
    a = bnr.sum_out_factors('Wet Grass?', 'Wet Grass?')
    print(a)
    #
    # bnr2 = BNReasoner('testing/multiply_example.BIFXML')
    # b = bnr2.multiply_factors(['D', 'E'])
    # print(b)

    # bnr3 = BNReasoner('testing/marginals_example.BIFXML')
    # c = bnr3.compute_marginal(['C'], pd.Series({'A': True}), bnr3.min_degree_order())
    # print(c)
    # cc = bnr3.compute_marginal(['C'], order=bnr3.min_degree_order())
    # print(cc)

    # bnr4 = BNReasoner('testing/lecture_example.BIFXML')
    # d = bnr4.compute_marginal(['Wet Grass?', 'Slippery Road?'], order=bnr4.min_degree_order())
    # print(d)
    # dd = bnr4.compute_marginal(['Wet Grass?', 'Slippery Road?'], pd.Series({'Winter?': True, 'Sprinkler?': False}),
    #                          order=bnr4.min_degree_order())
    # print(dd)
    # bnr5 = BNReasoner('testing/lecture_example2.BIFXML')
    # d = bnr5.MPE(pd.Series({'J': True, 'O': False}))
    # print(d)
    # e = bnr5.MAP(['I', 'J'], pd.Series({'O': True}))
    # print(e)


if __name__ == '__main__':
    main()
