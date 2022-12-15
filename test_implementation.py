import copy
from typing import Union
from BNReasoner import BNReasoner
from BayesNet import BayesNet
from networkpruning import NetworkPruning
from Independence import Independence
from DSeparation import DSeparation
from map_mep import MapAndMpe
from ordering import Ordering

def test_network_pruning(file, query, evidence):
    print("----- Network Pruning\n >> Test 1\n The network before and after pruning will be drawn now")
    networkpruning_test = NetworkPruning(file, query, evidence)
    networkpruning_test.bn.draw_structure()
    networkpruning_test.execute()
    networkpruning_test.bn.draw_structure()


def test_dsep(file, x, y, z):
    dsep_test = DSeparation(file, x, y, z)
    result = dsep_test.execute()
    print("----- D-Separation\n >> Test 1\n  Manual calculation gives: True\n  BNReasoner gives: ", result)


print("Hello there! Hope you have a great day :).")
print("We will walk you through our some test cases on our implementation")

test_file1 = 'testing/lecture_example.BIFXML'
test_file2 = 'testing/lecture_example2.BIFXML'

test_network_pruning(test_file2, {"O"}, {"X"})  # (file, query, evidence)
test_dsep(test_file1, {"Slippery Road?"}, {"Winter?"}, {"Rain?"})  # x, y ,z
test_dsep(test_file2, {"X"}, {"O"}, {"J"})  # x, y ,z

