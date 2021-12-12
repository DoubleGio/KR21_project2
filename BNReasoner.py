from typing import Union, List
from itertools import product
import pandas as pd
import numpy as np
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

    def sum_out_factors(self, factor: Union[str, pd.DataFrame], subset: Union[str, list]):
        if isinstance(factor, str):
            factor = self.bn.get_cpt(factor)
        if isinstance(subset, str):
            subset = [subset]

        variables = [v for v in factor.keys() if v not in subset + ['p']]  # Set subtraction
        new_factor = self.init_factor(variables, 0)
        subset_factor = self.init_factor(subset, 0)

        for i, y in new_factor.iterrows():
            for _, z in subset_factor.iterrows():
                new_factor.loc[i, 'p'] = new_factor.loc[i, 'p'] + self.bn.get_compatible_instantiations_table(
                    y[:-1].append(z[:-1]), factor)['p'].sum()
                # new_factor.loc[i, 'p'] = new_factor.loc[i, 'p'] + factor

        return new_factor

    def multiply_factors(self, factors: List[Union[str, pd.DataFrame]]):
        # If there are strings in the input-list of factors, replace them with the corresponding cpt
        for x, y in enumerate(factors):
            if isinstance(y, str):
                factors[x] = self.bn.get_cpt(y)

        variables = list(set().union(*factors))
        variables.remove('p')
        new_factor = self.init_factor(variables, 1)

        for i, z in new_factor.iterrows():
            for j, f in enumerate(factors):
                # xi = self.bn.get_compatible_instantiations_table()
                new_factor.loc[i, 'p'] = new_factor.loc[i, 'p'] * self.bn.get_compatible_instantiations_table(
                    z[:-1], f)['p'].sum()

        return new_factor

    @staticmethod
    def init_factor(variables: list, value=0):
        truth_table = product([True, False], repeat=len(variables))
        factor = pd.DataFrame(truth_table, columns=variables)
        factor['p'] = value
        return factor


def main():
    bnr = BNReasoner('testing/lecture_example.BIFXML')
    # a = bnr.sum_out_factors('Wet Grass?', 'Wet Grass?')
    a = bnr.multiply_factors(['Wet Grass?', 'Rain?'])

    print(a)


if __name__ == '__main__':
    main()
