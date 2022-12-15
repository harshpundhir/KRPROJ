from BayesNet import BayesNet
import networkx as nx
from marginalDistribution import MarginalDistributions
import matplotlib.pyplot as plt


obj = BayesNet()
obj.load_from_bifxml("testing/lecture_example.BIFXML")
nx.draw(obj.get_interaction_graph(), with_labels=True)




