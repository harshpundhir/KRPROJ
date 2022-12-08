from typing import Union
from BNReasoner import BNReasoner
from BayesNet import BayesNet

class NetworkPruning(BNReasoner):
    def __init__(self, net: Union[str, BayesNet], query: set, evidence: {}):
        super().__init__(net)
        self.query = query
        self.evidence = evidence

    def execute_pruning(self):
        print(self.evidence)

query = {"X"}
evidence = {"J"}
bnReasoner = NetworkPruning('testing/lecture_example2.BIFXML', query, evidence)
bnReasoner.execute_pruning()
