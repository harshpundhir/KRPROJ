import copy
from typing import Union

import pandas as pd
from pandas import DataFrame

from BNReasonerOrig import BNReasoner
from BayesNet import BayesNet
from marginalDistribution import multiply_factors, get_cpts_with_var
from networkpruning import NetworkPruning
from ordering import Ordering, shuffle


class MapAndMpe(BNReasoner):
    def __init__(self, net: Union[str, BayesNet], q: set, e: {}):
        super().__init__(net)
        self.query = q
        self.evidence = e
        self.res = {}

    def execute(self):
        pass

    def map(self, order: str):
        network = copy.deepcopy(self.bn)
        evidence_list = set()
        for x in self.evidence.keys():
            evidence_list.add(x)
        network = NetworkPruning(network, self.query, evidence_list).execute()

        if order == 'min_degree':
            ordered = (set(Ordering(self.bn).min_degree(network, network.get_all_variables())) - self.query)
            ordered.update(self.query)
        elif order == 'min_fill':
            ordered = (set(Ordering(self.bn).min_fill(network, network.get_all_variables())) - self.query)
            ordered.update(self.query)
        else:
            # TODO check this
            ordered = (set(shuffle(network, network.get_all_variables(), 0)) - self.query)
            ordered.update(self.query)

        cpt_tables = network.bn.get_all_cpts()
        result, assignment, multiplied_factors = DataFrame, {}, DataFrame
        for variable in ordered:
            factors = get_cpts_with_var(cpt_tables, variable)
            cpt_list = list(factors.values())
            f = pd.DataFrame()
            if len(cpt_list) > 0:
                f = cpt_list[0]

            if len(cpt_list) > 1:
                for index in range(len(factors) - 1):
                    f = multiply_factors(f, cpt_list[index + 1])

            if variable in self.query:
                result, assignment = self.maximize_factors(f, variable)
            elif variable not in self.query:
                result = self.sum_out_var(f, variable)

            for key in factors.keys():
                cpt_tables.pop(key)
            cpt_tables[' '.join(factors.keys())] = result
        return cpt_tables, assignment

    def mpe(self, order: str):
        network = copy.deepcopy(self.bn)
        network = NetworkPruning(network, self.query, self.evidence)
        network.edge_pruning(self.evidence)

        if order == 'min_degree':
            ordered = (set(Ordering(self.bn).min_degree(network, network.get_all_variables())) - self.query)
            ordered.update(self.query)
        elif order == 'min_fill':
            ordered = (set(Ordering(self.bn).min_fill(network, network.get_all_variables())) - self.query)
            ordered.update(self.query)
        else:
            # TODO check this
            ordered = (set(network.get_all_variables()) - self.query)

        cpt_tables = network.bn.get_all_cpts()
        print(cpt_tables)  # remove
        # result, assignment, multiplied_factors = DataFrame, {}, DataFrame
        for variable in ordered:
            factors = get_cpts_with_var(cpt_tables, variable)
            cpt_list = list(factors.values())
            print(cpt_list)  # remove
            f = cpt_list[0]

            if len(cpt_list) > 1:
                for index in range(len(factors) - 1):
                    f = multiply_factors(f, cpt_list[index + 1])

            result, assignment = self.maximize_factors(f, variable)

            for key in factors.keys():
                cpt_tables.pop(key)
            cpt_tables[' '.join(factors.keys())] = result
        # Initialize Dataframe
        trivial = pd.DataFrame({'p': [1.0]})

        # Multiply final expressions
        for key, view in cpt_tables.items():
            trivial.at[0, 'p'] = trivial.iloc[0]['p'] * view['p'].max()

        # print(f'\nThis is the MPE given the evidence {self._evidence}: \n {trivial}')
        return trivial

    @staticmethod
    def maximize_factors(self, factor, variable):
        cols = [col for col in factor.columns[:-1] if col != variable]

        if len(cols) != 0:
            cpt = factor.sort_values(by=['p'])

            unique_vals2 = cpt.drop_duplicates(subset=cols, keep="last")

            assignment = list(unique_vals2[variable])[0]

            unique_vals2 = unique_vals2.drop(columns=variable)
            unique_vals2 = unique_vals2.reset_index(drop=True)

            return unique_vals2, {variable: assignment}

        cpt = factor.loc[factor['p'].idxmax()]
        assignment = cpt[variable]
        cpt = cpt.drop(variable)
        return cpt, {variable: assignment}

    @staticmethod
    def sum_out_var(cpt: pd.DataFrame, variable: str):
        cpt = copy.deepcopy(cpt)

        cpt = cpt.drop(variable)

        rem_vars = [col for col in cpt.columns if col != 'p']

        if rem_vars:
            sum_cpt = cpt.groupby(rem_vars).aggregate({'p': 'sum'})
        else:
            sum_cpt = pd.DataFrame()
            sum_cpt['p'] = 0
        return sum_cpt


query = {"X", "I"}
evidence = {"J": True}
bnReasoner = MapAndMpe('testing/lecture_example2.BIFXML', query, evidence)
reasoner_map = bnReasoner.map('min_degree')
print("mmmmm", reasoner_map)
