from BNReasoner import BNReasoner
import pandas as pd

# EXAMPLES:

# Files:
x = BNReasoner('testing/lecture_example.BIFXML')
y = BNReasoner('testing/lecture_example2.BIFXML')

################
# D-Separation #
################

# Note: Run examples individually/separate as it changes the object internally

#print(x.d_separation(['Slippery Road?'], ['Rain?'], ['Winter?', 'Sprinkler?'])) # True
#print(x.d_separation(['Winter?', 'Sprinkler?'], ['Rain?'], ['Slippery Road?']))  # True
#print(x.d_separation(['Winter?', 'Sprinkler?'], ['Rain?'], ['Wet Grass?']))  # False
#print(x.d_separation(['Slippery Road?'], ['Sprinkler?'], ['Wet Grass?']))  # False
#print(x.d_separation(['Slippery Road?'], ['Rain?'], ['Wet Grass?']))  # True
#print(x.d_separation(['Wet Grass?'], ['Rain?'], ['Slippery Road?']))  # True
#print(x.d_separation(['Slippery Road?', 'Wet Grass?'], ['Rain?'], ['Winter?', 'Sprinkler?']))  # False
#print(x.d_separation(['Winter?'], ['Rain?'], ['Wet Grass?', 'Slippery Road?']))  # False

#x.bn.draw_structure()
#print(x.bn.get_all_variables())

#
#
#

#print(y.d_separation(['J'], [], ['I']))  # True
#print(y.d_separation(['J'], ['X'], ['I']))  # False
#print(y.d_separation(['J'], ['X'], ['O']))  # False
#print(y.d_separation(['I','J'], [], ['Y']))  # False
#print(y.d_separation(['J','I'], [], ['Y']))  # False
#print(y.d_separation(['Y'], [], ['J','I']))  # False
#print(y.d_separation(['J'], ['X','O'], ['I']))  # False
#print(y.d_separation(['J'], [], ['I','X','O']))  # False
#print(y.d_separation(['O'], ['Y','X'], ['J']))  # True
#print(y.d_separation(['Y'], ['X','Y'], ['O']))  # True

#y.bn.draw_structure()
#print(y.bn.get_all_variables())


############
# ordering #
############

#print(x.min_degree_order())  # 'Slippery Road?', 'Winter?', 'Sprinkler?', 'Rain?', 'Wet Grass?'
#print(x.min_fill_order())  # 'Winter?', 'Sprinkler?', 'Wet Grass?', 'Rain?', 'Slippery Road?'

#
#
#

#print(y.min_degree_order())  # 'I', 'J', 'Y', 'X', 'O'
#print(y.min_fill_order())  # 'I', 'J', 'Y', 'X', 'O'


###################
# Network Pruning #
###################

#x.network_pruning(['Wet Grass?'], pd.Series({'Winter?': True, 'Rain?': False}))  # (lecture 4 example)
#x.network_pruning(['Winter?', 'Wet Grass?', 'Slippery Road?'], pd.Series({'Rain?': False}))

#print(x.bn.get_all_variables())
#print(x.bn.get_all_cpts())
#x.bn.draw_structure()

#
#
#

#y.network_pruning(['J', 'I'], pd.Series({'Y': True}))
#y.network_pruning(['J'], pd.Series({'Y': True, 'X': False}))

#print(y.bn.get_all_variables())
#print(y.bn.get_all_cpts())
#y.bn.draw_structure()


###############
# Summing Out #
###############

bnr = BNReasoner('testing/lecture_example.BIFXML')
#print(bnr.bn.get_cpt('Wet Grass?'))
#a = bnr.sum_out_factors('Wet Grass?', 'Wet Grass?')
#print(a)


########################
# Multipliying factors #
########################

bnr2 = BNReasoner('testing/lecture_example.BIFXML')
#print(bnr2.bn.get_cpt('Winter?'))
#print(bnr2.bn.get_cpt('Rain?'))
#b = bnr2.multiply_factors(['Winter?', 'Rain?'])
#print(b)


####################
# Compute Marginal #
####################

bnr3 = BNReasoner('testing/lecture_example.BIFXML')
#c = bnr3.compute_marginal(['Wet Grass?', 'Slippery Road?'], order=bnr3.min_degree_order())
#print(c)

#
#
#

#c2 = bnr3.compute_marginal(['Wet Grass?', 'Slippery Road?'], pd.Series({'Winter?': True, 'Sprinkler?': False}),
#                              order=bnr3.min_fill_order())
#print(c2)


#######
# MPE #
#######

bnr4 = BNReasoner('testing/lecture_example2.BIFXML')
#d = bnr4.MPE(pd.Series({'J': True, 'O': False}))
#print(d)

#
#
#

#pd.set_option("display.max_columns", None) # print full dataframe
bnr4_2 = BNReasoner('testing/lecture_example.BIFXML')
#d2 = bnr4_2.MPE(pd.Series({'Winter?': True, 'Rain?': False}))
#print(d2)


#######
# MAP #
#######

bnr5 = BNReasoner('testing/lecture_example2.BIFXML')
#e = bnr5.MAP(['I', 'J'], pd.Series({'O': True}))
#print(e)

#
#
#

bnr5_2 = BNReasoner('testing/lecture_example.BIFXML')
#e2 = bnr5_2.MAP(['Winter?', 'Rain?'], pd.Series({'Wet Grass?': True}))
#print(e2)
