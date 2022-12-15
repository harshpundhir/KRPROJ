from typing import Union
from BayesNet import BayesNet
import pandas as pd


class BNReasoner(BayesNet):
    def __init__(self, net: Union[str, BayesNet]):
        """
        param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        super().__init__()
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

    @staticmethod
    def max_out(cpt: pd.DataFrame, variable: str) -> pd.DataFrame:
        columns = [column for column in cpt if (column != 'p' and column != variable)]
        if len(columns) > 0:

            df = cpt
            df.reset_index(drop=True, inplace=True)
            df1 = df.groupby(columns, as_index=False)["p"].agg(["max", "idxmax"])

            df1["idxmax"] = df.loc[df1["idxmax"], variable].values

            df1 = df1.rename(columns={"idxmax": variable, "max": "p"}).reset_index()

            cpt = df1

            return cpt
        else:
            cpt = cpt.loc[cpt["p"] == cpt["p"].max()]

        return cpt

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


obj = BNReasoner("testing/lecture_example.BIFXML")

# print(marginalize(obj.get_all_cpts()['Wet Grass?'], 'Rain?'))
#print(obj.multiply_factors([obj.bn.get_all_cpts()['Sprinkler?'], obj.bn.get_all_cpts()['Rain?']]))
