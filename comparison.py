import copy
import random
import time

from typing import Union
from BNReasoner2 import BNReasoner
from BayesNet import BayesNet
from networkpruning import NetworkPruning


def get_random_evidence_query(network):
    evidence_sample, query_sample = set(), set()
    all_variables = network.get_all_variables()

    for x in range(len(all_variables)):
        if random.randint(0, 100) < 20:
            evidence_sample.add(all_variables.pop(x))
        elif random.randint(0, 100) < 20:
            query_sample.add(all_variables.pop(x))
    # if len(query_sample) == 0:
    #     variable = all_variables[random.randint(0, (len(all_variables) - 1))]
    #     query_sample.append(all_variables.pop(variable))
    return query_sample, evidence_sample


file = 'testing/test.BIFXML'
BNReasoner1 = BNReasoner(file)

# Run pruned network
pruned_time_track, normal_pruned_time_track, min_fill_time_track, min_degree_time_track = [], [], [], []

for i in range(100):
    query, evidence = get_random_evidence_query(BNReasoner1)

    start_time1 = time.time()
    BNReasoner1_pruned = NetworkPruning(file, query, evidence)
    BNReasoner1_pruned.execute()
    end_time1 = time.time()
    pruned_time_track.append(end_time1 - start_time1)

    start_time2 = time.time()
    min_fill = Ordering(file)
    result1 = min_fill.min_degree()

print(pruned_time_track)
