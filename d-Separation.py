import copy
from typing import Union
from BNReasoner import BNReasoner
from BayesNet import BayesNet


class DSeparation(BNReasoner):
    def __init__(self, net: Union[str, BayesNet], x: set, y: set, z: set):

        super().__init__(net)
        self.X = x
        self.Y = y
        self.Z = z

    def execute(self):
        self.edge_pruning(self.Z)
        self.leaf_node_pruning_loop(self.X ^ self.Z ^ self.Z)
        return


X = {"Y"}
Y = {"J"}
Z = {"O"}
bnReasoner = DSeparation('testing/lecture_example2.BIFXML', X, Y, Z)
d_sep = copy.deepcopy(bnReasoner)  # why the fuck here
d_sep.execute()
