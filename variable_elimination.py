import pandas as pd
from BayesNet import BayesNet
from typing import Union
from typing import List
from BNReasoner import BNReasoner
from ordering import Ordering


class VariableEliminator(Ordering):  # accepts BN reasoner file
    def __int__(self, net: Union[str, BayesNet]):
        super().__init__(net)
        # self.ordr = Ordering(net)

    @staticmethod
    def factor_multiplication(f: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
        """ Given two factors, compute the product of the two factors. h = fg
        :param f: factor 1
        :param g: factor 2
        :return: the product of the two factors
        """
        # print(f, "\n", g)
        g.rename(columns={'p': 'p1'}, inplace=True)  # prevent name collision
        common_variables = [v for v in f.columns if v in g.columns]
        h = f.join(g.set_index(common_variables), on=common_variables).reset_index(drop=True)
        h['p1'] = h['p1'] * h['p']  # multiply the probabilities
        h.drop(columns=['p'], inplace=True)
        h.rename(columns={'p1': 'p'}, inplace=True)  # rename back to p
        # print(h)
        return h

    def variable_elimination(self, to_remove: List[str], heuristic='manual') -> pd.DataFrame:
        """ Sum out a set of variables by using variable elimination (according to given order).
        :return: the resulting factor
        """

        tau = pd.DataFrame()
        visited = set()
        if heuristic in ['manual', 'min_fill', 'min_degree']:
            if heuristic == 'manual':
                pass
            elif heuristic == 'min_fill':
                to_remove = self.min_fill(self.bn, to_remove)
            else:
                to_remove = self.min_degree(self.bn, to_remove)
        else:
            raise TypeError("Only 'manual', 'min_fill', 'min_degree' as elimination heuristics are allowed")
        for element in to_remove:
            if tau.empty:
                tau = self.bn.get_cpt(element)
                # print(element, tau)
            to_visit = set(self.bn.get_children(element)) - set(visited)
            for child in to_visit:
                tau = self.factor_multiplication(tau, self.bn.get_cpt(child))
                visited.add(child)
            tau = self.marginalize(tau, element)
            visited.add(element)
        return tau


with open('testResults.txt', 'w') as f:
    f.write("Variable elimination Results")

obj = VariableEliminator("testing/lecture_example.BIFXML")
myset = ["Sprinkler?", "Winter?", "Rain?"]
print(obj.variable_elimination(myset, 'min_degree'))
