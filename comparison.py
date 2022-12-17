import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
import time

from BNReasoner2 import BNReasoner
from networkpruning import NetworkPruning


def get_random_evidence_query(network):
    evidence_sample, query_sample = set(), set()
    all_variables = network.get_all_variables()

    for x in range(len(all_variables)):
        if random.randint(0, 100) < 20:
            evidence_sample.add(all_variables.pop(x))
        elif random.randint(0, 100) < 20:
            query_sample.add(all_variables.pop(x))
    return query_sample, evidence_sample


file = 'testing/test.BIFXML'
BNReasoner1 = BNReasoner(file)

time_1, time_2, time_3, time_4 = [], [], [], []

for i in range(100):
    query, evidence = get_random_evidence_query(BNReasoner1)

    start_time1 = time.time()
    pruned_network = NetworkPruning(file, query, evidence).execute()
    pruned_network.marginal_distribution()
    time_1.append(time.time() - start_time1)

    start_time2 = time.time()
    time_2.append(time.time() - start_time2)

    start_time3 = time.time()
    network3 = BNReasoner(file)
    network3.minfill_ordering(query)
    time_3.append(time.time() - start_time3)

    start_time4 = time.time()
    network4 = BNReasoner(file)
    network4.mindegree_ordering(query)
    time_4.append(time.time() - start_time4)

data_experiment1 = pd.concat([time_1['With Network Pruning'], time_2['Without Network Pruning']], axis=1)
data_experiment2 = pd.concat([time_3['Min Fill Heuristic'],time_4['Min Degree Heuristic'],],axis=1)

data_experiment1.columns = ['With Network Pruning', 'Without Network Pruning']
data_experiment2.columns = ['Min Fill Heuristic', 'Min Degree Heuristic']

print(data_experiment1)
plt.figure(figsize=(8, 6)) # (width,height)
plt.ylabel('Number of splits')
plt.xlabel('Model Heuristic')
sns.boxplot(data=data)
plt.title("Comparison of splits for 9x9 sudokus")
plt.show()

