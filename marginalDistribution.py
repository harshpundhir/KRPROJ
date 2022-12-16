from typing import Set, Union

import pandas as pd

from BNReasonerOrig import BNReasoner
from BayesNet import BayesNet
from ordering import Ordering


def sum_out(cpt: pd.DataFrame, variable: str) -> pd.DataFrame:
    columns = [column for column in cpt if (
            column != 'p' and column != variable)]

    partition_t = cpt[variable] == True
    partition_t_cpt = cpt[partition_t].drop(variable, axis=1)
    partition_f_cpt = cpt[~partition_t].drop(variable, axis=1)

    summed_cpt = pd.concat([partition_t_cpt, partition_f_cpt]).groupby(
        columns, as_index=False)["p"].sum()

    return summed_cpt


def multiply_factors(cpt1, cpt2) -> pd.DataFrame:
    try:
        common_vars = list(
            set([col for col in cpt1.columns if col != 'p']) & set([col for col in cpt2.columns if col != 'p']))

        cpt1, cpt2 = cpt1, cpt2
        combined_cpt = pd.merge(left=cpt1, right=cpt2,
                                on=common_vars, how='inner')

        combined_cpt['p'] = (combined_cpt['p_x'] * combined_cpt['p_y'])
        combined_cpt.drop(['p_x', 'p_y'], inplace=True, axis=1)
        return combined_cpt
    except Exception as e:
        print(e)


def get_cpts_with_var(all_cpts: dict[str, pd.DataFrame], variable: str):
    cpts = {}
    for var, cpt in all_cpts.items():
        if variable in cpt.columns:
            cpts[var] = cpt
    return cpts


class MarginalDistributions(BNReasoner):
    def __init__(self, net: Union[str, BayesNet], varibales: Set[str], Evidence: dict[str, bool]):
        super().__init__(net)
        self.varibales = varibales
        self.Evidence = Evidence

    def execute(self):
        result = self.marginal_Dist(self.varibales, self.Evidence)
        return result

    # P(Q)
    def marginal_Dist(self, X: Set[str], E: dict[str, bool]) -> pd.DataFrame:
        '''
        Marginal distribution of variables X given E
        X: query varialbes
        E: evidence could be empty : {}
        '''

        order = Ordering(self.bn).min_degree(bn=self.bn, variables=list(
            set(self.bn.get_all_variables()) - X))

        if E != {}:
            reduced_cpts = self.reduce_using_evidence(E)
        else:
            reduced_cpts = self.reduce_using_evidence(E)
        j = 0
        for variable in order:
            all_cpts_with_var = get_cpts_with_var(reduced_cpts, variable=variable)
            all_cpts = list(all_cpts_with_var.values())
            first_cpt = all_cpts[0]

            for i in range(len(all_cpts_with_var) - 1):
                first_cpt = multiply_factors(first_cpt, cpt2=all_cpts[i + 1])
            summed_out_factor = sum_out(cpt=first_cpt, variable=variable)
            for var, cpt in all_cpts_with_var.items():
                del reduced_cpts[var]
            reduced_cpts[str(j)] = summed_out_factor
            j += 1
        factor_list = list(reduced_cpts.values())
        result = factor_list[0]
        for i in range(len(factor_list) - 1):
            result = multiply_factors(result, cpt2=factor_list[i + 1])

        if E != {}:
            result = self.normalize_by_evidence(result, E)
        return result

    def var_elimination(self, X, E):
        pass

    def reduce_using_evidence(self, E):

        all_cpts = self.bn.get_all_cpts()
        for variable, cpt in all_cpts.items():
            for var, evidence in E.items():
                if var in cpt.columns:
                    cpt.loc[cpt[var] != evidence, 'p'] = 0
            all_cpts[variable] = cpt
        return all_cpts

    def normalize_by_evidence(self, cpt, E):
        probablity_e = self.marginal_Dist(
            X=set(E.keys()),
            E=dict()
        )
        for var, assignment in E.items():
            probablity_e = probablity_e[probablity_e[var] == assignment]
        evidence_prob = probablity_e.iloc[0]['p']
        cpt['p'] /= evidence_prob
        return cpt


obj = MarginalDistributions("testing/sleep_paralysis.BIFXML",{'SleepPar'}, {'Techno': True})
variables = obj.bn.get_all_variables()
print(variables)
# variables.remove('Wet Grass?')
# variables.remove('Slippery Road?')

print(obj.marginal_Dist({'SleepPar'}, {'Student': True}))

