from BNReasoner import BNReasoner
from BayesNet import BayesNet
import copy
import random
from typing import Union
from networkx import Graph


def shuffle(variables: list[str], heuristic_orders: list[list[str]], amount: int):
    while len(heuristic_orders) < amount + 2:

        r = random.sample(variables, len(variables))

        if r not in heuristic_orders:
            heuristic_orders.append(r)

    return heuristic_orders[2:]


def connect_non_adj(graph: Graph, node: str):
    neighbors = list(graph.neighbors(node))
    for i in range(len(neighbors)):
        neighbors_i = list(graph.neighbors(neighbors[i]))
        if i + 1 == len(neighbors):
            break
        for j in range(i + 1, len(neighbors)):
            if neighbors[j] not in neighbors_i:
                graph.add_edge(neighbors[i], neighbors[j])


def get_edge(graph: Graph, node: str):
    neighbors = list(graph.neighbors(node))
    edges_added = 0
    for i in range(len(neighbors)):
        neighbors_i = list(graph.neighbors(neighbors[i]))
        if i + 1 == len(neighbors):
            break
        for j in range(i + 1, len(neighbors)):
            if neighbors[j] not in neighbors_i:
                edges_added += 1
    return edges_added


def find_smallest_edge(graph: Graph, variables: list[str]):
    return sorted(variables, key=lambda x: get_edge(graph, x))[0]


def get_smallest_nbr(graph: Graph, variables: list[str]):
    node = variables[0]
    least_amount_neighbors = len(list(graph.neighbors(variables[0])))
    for i in range(1, len(variables)):
        amount_neighbors = len(list(graph.neighbors(variables[i])))
        if amount_neighbors < least_amount_neighbors:
            node = variables[i]
            least_amount_neighbors = amount_neighbors
    return node


class Ordering(BNReasoner):

    def execute(self):
        pass

    def __init__(self, net: Union[str, BayesNet]):
        super().__init__(net)

    def min_degree(self, bn: BayesNet, variables: list[str]):
        variables = copy.deepcopy(variables)
        graph = bn.get_interaction_graph()
        order = []

        for i in range(len(variables)):
            node = get_smallest_nbr(graph, variables)
            connect_non_adj(graph, node)
            graph.remove_node(node)
            variables.remove(node)
            order.append(node)

        return order

    def min_fill(self, bn: BayesNet, variables: list[str]) -> list[str]:
        variables = copy.deepcopy(variables)
        graph = bn.get_interaction_graph()
        order = []

        for i in range(len(variables)):
            node = find_smallest_edge(graph, variables)
            connect_non_adj(graph, node)
            graph.remove_node(node)
            variables.remove(node)
            order.append(node)

        return order


orderer = Ordering("testing/lecture_example.BIFXML")
X = ["Rain?","Winter?","Sprinkler?","Wet Grass?","Slippery Road?"]
result = orderer.min_degree(orderer.bn, X)
print(result)