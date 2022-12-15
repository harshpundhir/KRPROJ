# import copy
# from typing import Union
# from BNReasoner import BNReasoner
# from BayesNet import BayesNet
from networkpruning import NetworkPruning
# from Independence import Independence
from DSeparation import DSeparation
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
while True:
    input1 = input("\n FIRST unpruned sample THEN pruned sample will shown. \n Continue? --> Press enter.\n ")
    if input1 == "":
        test_file1
        test_network_pruning(file, query, evidence)
        break



