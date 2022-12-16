from typing import Union, Dict, List
from copy import deepcopy
from BayesNet import BayesNet
import pandas as pd
import typing


class BNReasoner(BayesNet):
    def __init__(self, net: Union[str, BayesNet]):
        """
        param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        super().__init__()
        self.instantiation = {}
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    # TODO: This is where your methods should go

    @staticmethod
    def marginalize(cpt: pd.DataFrame, variable: str) -> pd.DataFrame:
        columns = [column for column in cpt if (column != 'p' and column != variable)]

        partition_t = cpt[variable] == True
        partition_t_cpt = cpt[partition_t].drop(variable, axis=1)
        partition_f_cpt = cpt[~partition_t].drop(variable, axis=1)

        summed_cpt = pd.concat([partition_t_cpt, partition_f_cpt]).groupby(columns, as_index=False)["p"].sum()
        return summed_cpt

    def maxing_out(self, X: str, factor: pd.DataFrame) -> pd.DataFrame:
        if X not in factor:
            return factor

        factor_name = factor.columns[-1]
        new_columns = list(filter(lambda variable: variable != X, factor.columns))
        new_variables = new_columns[:-1]

        # Max out
        if len(new_variables) == 0:
            row = factor.loc[factor[factor_name].idxmax()]
            value = row[X]
            result = pd.DataFrame(row)
        else:
            result = factor.groupby(new_variables).max(factor_name)

            # Reverse dataframe to preserve initial ordering
            result = result.iloc[::-1]

            # Reset indexes from 0
            result = result.reset_index()

            # Remove the maxed out variable
            value = result[X][0]

        if X in result:
            del result[X]

        # Rename factor
        result = result.rename(columns={factor_name: f"max_{X}={value} > {factor_name}"})
        self.instantiation[X] = value
        return result

    @staticmethod
    def multiply_factors(cpts: list[pd.DataFrame]) -> pd.DataFrame:
        cpt1, cpt2 = cpts
        variable = (set(cpt1.columns) & set(cpt2.columns))
        variable.remove('p')

        for var in variable:
            combined_cpt = pd.merge(left=cpt1, right=cpt2, on=str(var), how='inner')

        combined_cpt['p'] = (combined_cpt['p_x'] * combined_cpt['p_y'])
        combined_cpt.drop(['p_x', 'p_y'], inplace=True, axis=1)
        return combined_cpt

    def edge_pruning(self, prune_nodes):
        for node in prune_nodes:
            copy_edges = []
            children = self.bn.get_children(node)
            for child in children:
                copy_edges.append(tuple((node, child)))
            for edge in copy_edges:
                self.bn.del_edge(edge)
        return

    def get_leaf_nodes(self, nodes):  # : object) -> object
        leaf_nodes = []
        for node in nodes:
            if len(self.bn.get_children(node)) == 0:
                leaf_nodes.append(node)
        return leaf_nodes

    def get_regular_nodes(self, exception_nodes):
        regular_nodes = []
        for node in self.bn.get_all_variables():
            if node not in exception_nodes:
                regular_nodes.append(node)
        return regular_nodes

    def leaf_node_pruning_loop(self, exception_nodes):
        regular_nodes = self.get_regular_nodes(exception_nodes)
        leaves = self.get_leaf_nodes(regular_nodes)
        while leaves:
            for leave in leaves:
                self.bn.del_var(leave)
            regular_nodes = self.get_regular_nodes(exception_nodes)
            leaves = self.get_leaf_nodes(regular_nodes)
        return

    def reduce_cpts(self, cpts: Dict, evidence: Dict):
        """
        Reduces the tables by deleting the rows that are not aligned with the evidence.
        :param cpts: A dictionary containing all the Dataframes that represent the CPTs.
        :param evidence: A dictionary with the variables and assigned thruth values.
        :return: A dictionary of all the reduced Dataframes.
        """
        if not evidence:
            return cpts

        for cpt in cpts:
            for ev in evidence:
                if ev in cpts[cpt]:
                    for i in range(len(cpts[cpt].index) - 1, -1, -1):
                        if cpts[cpt].loc[i, ev] == (not evidence[ev]):
                            cpts[cpt] = cpts[cpt].drop(labels=i, axis=0)
                            cpts[cpt] = cpts[cpt].reset_index(drop=True)
        return cpts

    def map(self, variables: List[str], evidence: Dict):
        """
        Calculates the most likely instantiation of the asked variables based on some evidence.
        :param variables: A list with the variables which the instantiation is desired.
        :param evidence: A dictionary with the variables and assigned thruth values.
        :return: A dictionary with the asked variables and their instantiation
        """
        res, cpts = {}, self.bn.get_all_cpts()
        cpts = self.reduce_cpts(cpts, evidence)
        pos_marg = self.marginal_distributions(variables, evidence, self.min_fill())
        idx = pos_marg['p'].idxmax()
        for var in pos_marg:
            if var in variables:
                res[var] = pos_marg.iloc[idx][var]
        return res

obj = BNReasoner("testing/lecture_example.BIFXML")

# print(obj.compute_map({'Wet Grass?': True}, {'Rain?': True}))
# print(obj.multiply_factors([obj.bn.get_all_cpts()['Sprinkler?'], obj.bn.get_all_cpts()['Rain?']]))
