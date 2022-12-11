import copy
from typing import Union
from BNReasoner import BNReasoner
from BayesNet import BayesNet


class NetworkPruning(BNReasoner):
    def __init__(self, net: Union[str, BayesNet], query: set, evidence: {}):
        super().__init__(net)
        self.query = query
        self.evidence = evidence

    def edge_pruning(self):
        for evidence_edge in self.evidence:
            copy_edges = []
            for f, t in self.bn.structure.edges(nbunch=self.evidence):
                copy_edges.append(t)
            for e in copy_edges:
                self.bn.del_edge((evidence_edge, e))
        return

    def get_regular_nodes(self):
        regular_nodes = []
        for node in self.bn.get_all_variables():
            if node not in self.evidence ^ self.query:
                regular_nodes.append(node)
        print("Regular nodes: ", regular_nodes)
        return regular_nodes

    def node_pruning(self):
        regular_nodes = self.get_regular_nodes()
        leaves = self.get_leaf_nodes(regular_nodes)

        while leaves:
            print("<< Loop")
            print("leaves: ", leaves)
            for leave in leaves:
                self.bn.del_var(leave)

            regular_nodes = self.get_regular_nodes()
            leaves = self.get_leaf_nodes(regular_nodes)
        return pruned_network

    def get_leaf_nodes(self, nodes):  # : object) -> object
        leaf_nodes = []
        for node in nodes:
            if len(self.bn.get_children(node)) == 0:
                leaf_nodes.append(node)
        return leaf_nodes

    def execute_pruning(self):
        print("Evidence and Query:", self.evidence, self.query)
        self.edge_pruning()
        self.node_pruning()
        return


example_query = {"X"}
example_evidence = {"J"}
bnReasoner = NetworkPruning('testing/lecture_example2.BIFXML', example_query, example_evidence)
pruned_network = copy.deepcopy(bnReasoner)  # why the fuck here
pruned_network.execute_pruning()
pruned_network.bn.draw_structure()
bnReasoner.bn.draw_structure()
