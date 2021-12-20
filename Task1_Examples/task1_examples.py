from typing import Union
from BayesNet import BayesNet

import pandas as pd

'''
--- Add this in BatenNet.py ---
    def get_parents(self, variable: str) -> List[str]:
        """
        Returns the parents of the variable in the graph.
        :param variable: Variable to get the parents from
        :return: List of parents
        """
        return [c for c in self.structure.predecessors(variable)]    
'''


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

    def dseparation(self, x, z, y):

        """
        Based on graph pruning.
        Inputs should be lists of variables/nodes (see examples below).
        """
        query_variables = [x, y, z]
        query_variables = [item for sublist in query_variables for item in sublist]
        all_variables = self.bn.get_all_variables()
        non_instantiated_variables = list(set(all_variables) - set(query_variables))

        # iteratively removes leaf nodes that are not in the set of {x,y,z}
        counter = 0
        while counter <= len(non_instantiated_variables):

            if non_instantiated_variables == []:
                break

            for var in non_instantiated_variables[:]:
                if self.bn.get_children(var) == []:
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

        #print('Num_of_connections: ' + str(num_of_connections))

        if num_of_connections == 0:
            return True
        else:
            return False

    def network_pruning(self, q, e):

        """
        q: set of query vars --> e.g. ['A','B','C']
        e: evidence set --> e.g. {'A': True, 'B': False}
        """

        evidence = list(e.keys())
        query_plus_evidence_variables = [q, evidence]
        query_plus_evidence_variables = [item for sublist in query_plus_evidence_variables for item in sublist]
        all_variables = self.bn.get_all_variables()
        non_instantiated_variables = list(set(all_variables) - set(query_plus_evidence_variables))

        # iteratively removes leaf nodes that are not in the set of {q,e}
        counter = 0
        while counter <= len(non_instantiated_variables):

            if non_instantiated_variables == []:
                break

            for var in non_instantiated_variables[:]:
                if self.bn.get_children(var) == []:
                    non_instantiated_variables.remove(var)
                    self.bn.del_var(var)
                    counter -= 1
                else:
                    counter += 1

        # remove outgoing edges from all nodes in Z, and update CPTs
        for var in evidence:
            for child in self.bn.get_children(var):
                self.bn.del_edge((var, child))

                # new = self.bn.get_compatible_instantiations_table(pd.Series(e), self.bn.get_cpt(child))
                new = self.bn.get_compatible_instantiations_table(pd.Series({var: e.get(var)}), self.bn.get_cpt(child))
                new = new.drop([var], axis=1)
                self.bn.update_cpt(child, new)


########################################################################################################################

# EXAMPLES:


################
# D-Separation #
################

x = BNReasoner('/Users/Wes/Downloads/KR21_project2-main/testing/lecture_example.BIFXML')  # change to your own path

#print(x.dseparation(['Slippery Road?'], ['Rain?'], ['Winter?', 'Sprinkler?'])) #True
#print(x.dseparation(['Winter?', 'Sprinkler?'], ['Rain?'], ['Slippery Road?']))  # True
#print(x.dseparation(['Winter?', 'Sprinkler?'], ['Rain?'], ['Wet Grass?']))  # False
#print(x.dseparation(['Slippery Road?'], ['Sprinkler?'], ['Wet Grass?']))  # False
#print(x.dseparation(['Slippery Road?'], ['Rain?'], ['Wet Grass?']))  # True
#print(x.dseparation(['Wet Grass?'], ['Rain?'], ['Slippery Road?']))  # True
#print(x.dseparation(['Slippery Road?', 'Wet Grass?'], ['Rain?'], ['Winter?', 'Sprinkler?']))  # False
#print(x.dseparation(['Winter?'], ['Rain?'], ['Wet Grass?', 'Slippery Road?']))  # False

#print(x.bn.draw_structure())
#print(x.bn.get_all_variables())


y = BNReasoner('/Users/Wes/Downloads/KR21_project2-main/testing/lecture_example2.BIFXML')  # change to your own path

#print(y.dseparation(['J'], [], ['I']))  # True
#print(y.dseparation(['J'], ['X'], ['I']))  # False
#print(y.dseparation(['J'], ['X'], ['O']))  # False
#print(y.dseparation(['I','J'], [], ['Y']))  # False
#print(y.dseparation(['J','I'], [], ['Y']))  # False
#print(y.dseparation(['Y'], [], ['J','I']))  # False
#print(y.dseparation(['J'], ['X','O'], ['I']))  # False
#print(y.dseparation(['J'], [], ['I','X','O']))  # False
#print(y.dseparation(['O'], ['Y','X'], ['J']))  # True
#print(y.dseparation(['Y'], ['X','Y'], ['O']))  # True

#print(y.bn.draw_structure())
#print(y.bn.get_all_variables())


###################
# Network Pruning #
###################

z = BNReasoner('/Users/Wes/Downloads/KR21_project2-main/testing/lecture_example.BIFXML')

#z.network_pruning(['Winter?', 'Wet Grass?', 'Slippery Road?'], {'Rain?': False})
#z.network_pruning(['Wet Grass?'], {'Winter?': True, 'Rain?': False})

#print(z.bn.get_all_cpts())
#print(z.bn.draw_structure())
#print(z.bn.get_all_variables())
