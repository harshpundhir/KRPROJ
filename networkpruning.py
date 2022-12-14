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
        for x in self.evidence:
            print("Print X: ", x)
            self.edge_pruning(x)
        self.leaf_node_pruning_loop(self.evidence ^ self.query)
        return


example_query = {"Wet Grass?"}
example_evidence = {"Winter?", "Rain?"}
bnReasoner = NetworkPruning('testing/lecture_example.BIFXML', example_query, example_evidence)
pruned_network = copy.deepcopy(bnReasoner)  # why the fuck here
pruned_network.execute_pruning()
bnReasoner.bn.draw_structure()
pruned_network.bn.draw_structure()

