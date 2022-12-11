import copy
from typing import Union
from BNReasoner import BNReasoner
from BayesNet import BayesNet

class NetworkPruning(BNReasoner):
    def __init__(self, net: Union[str, BayesNet], query: set, evidence: {}):
        super().__init__(net)
        self.query = query
        self.evidence = evidence

    def edge_pruning(self, pruned_network):
        for evidence_edge in self.evidence:
            copy_edges = []
            print(">> edge pruning")
            for f, t in self.bn.structure.edges(nbunch=self.evidence):
                copy_edges.append(t)
            for e in copy_edges:
                pruned_network.del_edge((evidence_edge, e))
        return pruned_network

    def node_pruning(self, pruned_network):
        nodes = set(self.bn.get_all_variables())
        leaves = self.get_leaf_nodes(nodes)
        print(leaves)
        #remaining_nodes = self.query.union(self.evidence)
        while leaves is not empty:
            pruned_network.del_var(leaves[0])
            leaves = pruned_network.get_leaf_nodes(nodes)
            print("<<<<< Loop")
        return pruned_network

    def get_leaf_nodes(self, all_nodes: object) -> object:
        leaf_nodes = []
        for node in all_nodes:
            if len(self.bn.get_children(node)) == 0:
                leaf_nodes.append(node)
        return leaf_nodes

    def execute_pruning(self):
        NetworkCopy = copy.deepcopy(self.bn)
        edge_pruned_network = self.edge_pruning(NetworkCopy)
        full_pruned_network = self.node_pruning(edge_pruned_network)
        full_pruned_network.draw_structure()
        return full_pruned_network

query = {"X"}
evidence = {"J"}
bnReasoner = NetworkPruning('testing/lecture_example.BIFXML', query, evidence)
bnReasoner.execute_pruning()
bnReasoner.bn.draw_structure()
