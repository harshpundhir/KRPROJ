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

    def connected_nodes(self, node_1, node_2):
        connected = False
        if node_2 in self.bn.get_descendants(node_1) or node_1 in self.bn.get_descendants(node_2):
            connected = True
        return connected

    def connected_sets(self, set_1, set_2):
        check = False
        for node_1 in set_1:
            for node_2 in set_2:
                check = self.connected_nodes(node_1, node_2)
        return check

    def execute(self):
        self.edge_pruning(self.Z)
        self.leaf_node_pruning_loop(self.X ^ self.Y ^ self.Z)
        if self.connected_sets(self.X, self.Y):
            return False
        else:
            return True


X = {"Rain?"}
Y = {"Winter?"}
Z = {"Sprinkler?"}
bnReasoner = DSeparation('testing/lecture_example.BIFXML', X, Y, Z)
bnReasoner.bn.draw_structure()
d_sep = copy.deepcopy(bnReasoner)  # why the fuck here
checking = d_sep.execute()
d_sep.bn.draw_structure()
print("D-Seperation is", checking)
