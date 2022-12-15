from BayesNet import BayesNet
import networkx as nx
from marginalDistribution import MarginalDistributions
import matplotlib.pyplot as plt


obj = BayesNet()
obj.load_from_bifxml("testing/hailfinder25.xml")
print(obj.get_all_cpts())
nx.draw(obj.get_interaction_graph(), with_labels=True)




