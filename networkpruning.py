import copy
from typing import Union
from BNReasoner import BNReasoner
from BayesNet import BayesNet


class NetworkPruning(BNReasoner):
    def __init__(self, net: Union[str, BayesNet], query: set, evidence: set):
        super().__init__(net)
        self.query = query
        self.evidence = evidence

    def execute(self):
        self.edge_pruning(self.evidence)
        self.leaf_node_pruning_loop(self.evidence ^ self.query)
        return

#
# example_query = {"Winter?"}
# example_evidence = {"Rain?", "Sprinkler?"}
#
# bnReasoner = NetworkPruning('testing/lecture_example.BIFXML', example_query, example_evidence)
# print(bnReasoner.bn.get_all_cpts())
# pruned_network = copy.deepcopy(bnReasoner)  # why the fuck here
# pruned_network.execute()
# bnReasoner.bn.draw_structure()
# pruned_network.bn.draw_structure()
# print(bnReasoner.bn.get_all_cpts())

