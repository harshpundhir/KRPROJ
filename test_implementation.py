# import copy
# from typing import Union
from BNReasonerOrig import BNReasoner
from BayesNet import BayesNet
from networkpruning import NetworkPruning
# from Independence import Independence
from map_mep import MapAndMpe
from DSeparation import DSeparation
from ordering import Ordering
from marginalDistribution import MarginalDistributions


# from map_mep import MapAndMpe
# from ordering import Ordering


def test_network_pruning(file, query, evidence):
    networkpruning_test = NetworkPruning(file, query, evidence)
    networkpruning_test.bn.draw_structure()
    networkpruning_test.execute()
    networkpruning_test.bn.draw_structure()


def test_dsep(file, x, y, z):
    dsep_test = DSeparation(file, x, y, z)
    return dsep_test.execute()


print("\nHello there! Hope you have a great day :).")
print("We will walk you through our some test cases on our implementation:")

test_file1 = 'testing/lecture_example.BIFXML'
test_file2 = 'testing/lecture_example2.BIFXML'

print("\n-- D-Separation --\n")
dsep_test1 = DSeparation(test_file1, {"Slippery Road?"}, {"Winter?"}, {"Rain?"})  # x, y ,z
print(" >> Test 1\n  Manual calculation gives: True\n  BNReasoner gives: ", dsep_test1.execute())
dsep_test2 = DSeparation(test_file2, {"X"}, {"O"}, {"J"})  # x, y ,z
print(" >> Test 1\n  Manual calculation gives: True\n  BNReasoner gives: ", dsep_test2.execute())
#
#
# print("\n-- Independence --\n")
# test_dsep(test_file1, {"Slippery Road?"}, {"Winter?"}, {"Rain?"}, 1)  # x, y ,z
# test_dsep(test_file2, {"X"}, {"O"}, {"J"}, 2)  # x, y ,z
#
# test_network_pruning(test_file2, {"O"}, {"X"})  # (file, query, evidence)

print("\n-- Network Pruning --\n ")


# while True:
#     input1 = input("\n FIRST unpruned sample THEN pruned sample will shown. \n Continue? --> Press enter.\n ")
#     if input1 == "":
#         test_network_pruning(test_file1, {"Slippery Road?"}, {"Winter?"})
#         break


def marginalize():
    print(" -- Marginalization --")
    instance = BNReasoner(test_file1)
    print(f"Before Marginalization of Rain from Wet Grass\n ________________________ \n \
    {instance.bn.get_cpt('Wet Grass?')} \n\n After Marginalizing Rain from Wet Grass\n _____________________")
    print(instance.marginalize(instance.bn.get_cpt("Wet Grass?"), "Rain?"))


def maxingout():
    print(" -- Maxing out --")
    instance = BNReasoner(test_file1)
    print(f"Before Maxing out Rain from Wet Grass \n ________________________ \n \
        {instance.bn.get_cpt('Wet Grass?')} \n\n After maxing out Rain from Wet Grass\n _____________________")
    print(instance.max_out(instance.bn.get_cpt("Wet Grass?"), "Rain?"))


def factormultiplication():
    print("-- Factor Multiplication--")
    instance = BNReasoner(test_file1)
    print(f"Before Multiplication \n ________________________ \n {instance.bn.get_cpt('Wet Grass?')} \n \
          {instance.bn.get_cpt('Slippery Road?')} \n After factor multiplication \n ____________________")
    print(instance.multiply_factors([instance.bn.get_cpt('Wet Grass?'), instance.bn.get_cpt('Slippery Road?')]))


def order():
    print(" \n -- ORDERING --")
    orderer = Ordering(test_file1)
    print('Order over "Rain?", "Winter?", "Sprinkler?", "Wet Grass?", "Slippery Road?" \n ______________________')
    X = ["Rain?", "Winter?", "Sprinkler?", "Wet Grass?", "Slippery Road?"]
    result_minf = orderer.min_fill(orderer.bn, X)
    result_mind = orderer.min_degree(orderer.bn, X)
    print("Min fill order: ", result_minf)
    print("Min degree order: ", result_mind)


def marginalDistributions():
    print(" \n-- Marginal Distribution --")
    instance = MarginalDistributions(test_file1, {"Slippery Road?"}, {"Winter?": True}).execute()
    print(instance)


def mapmpe():
    print(" \n-- MAP MPE -- ")
    instance = MapAndMpe(test_file2, {"X", "I"}, {"J": True})

    print("MAP: ", instance.map('min_degree'))
    print("MPE: ", instance.mpe('min_degree'))


marginalize()
maxingout()
factormultiplication()
order()
marginalDistributions()
mapmpe()
