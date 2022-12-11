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

        print(columns)
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
    def multiply_factors(cpts: list[pd.DataFrame], variable) -> pd.DataFrame:
        cpt1, cpt2 = cpts
        combined_cpt = pd.merge(left=cpt1, right=cpt2, on=variable, how='inner')

        combined_cpt['p'] = (combined_cpt['p_x'] * combined_cpt['p_y'])
        combined_cpt.drop(['p_x', 'p_y'], inplace=True, axis=1)
        return combined_cpt


obj = BNReasoner("testing/lecture_example.BIFXML")

# print(marginalize(obj.get_all_cpts()['Wet Grass?'], 'Rain?'))
print(obj.multiply_factors([obj.bn.get_all_cpts()['Sprinkler?'], obj.bn.get_all_cpts()['Rain?']], 'Winter?'))
