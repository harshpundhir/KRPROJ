import copy
from typing import Union
from BNReasoner import BNReasoner
from BayesNet import BayesNet


class NetworkPruning(BNReasoner):
    def __init__(self, net: Union[str, BayesNet], query: set, evidence: {}):
        super().__init__(net)
        self.query = query
        self.evidence = evidence

    def execute_pruning(self):
        print("Evidence and Query:", self.evidence, self.query)
        self.edge_pruning(self.evidence)
        self.leaf_node_pruning_loop(self.evidence ^ self.query)
        return


example_query = {"X"}
example_evidence = {"J"}
bnReasoner = NetworkPruning('testing/lecture_example2.BIFXML', example_query, example_evidence)
pruned_network = copy.deepcopy(bnReasoner)  # why the fuck here
pruned_network.execute_pruning()
bnReasoner.bn.draw_structure()
