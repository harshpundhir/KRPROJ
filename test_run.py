from BayesNet import BayesNet
import networkx as nx
from marginalDistribution import MarginalDistributions
import matplotlib.pyplot as plt
from BNReasonerOrig import BNReasoner

# obj = BayesNet()
# obj.load_from_bifxml("testing/lecture_example.BIFXML")
# nx.draw(obj.get_interaction_graph(), with_labels=True)

instance = BNReasoner("testing/lecture_example.BIFXML")
print(instance.marginalize(instance.bn.get_cpt("Wet Grass?"), "Rain?"))


