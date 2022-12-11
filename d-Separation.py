import copy
from typing import Union
from BNReasoner import BNReasoner
from BayesNet import BayesNet


class DSeparation(BNReasoner):
    def __init__(self, net: Union[str, BayesNet], X: set, Y: set, Z:set):
        super().__init__(net)
        self.X = X
        self.Y = Y
        self.Z = Z

    def execute(self):

        return


bnReasoner = DSeparation('testing/lecture_example2.BIFXML', X, Y, Z)
d_sep = copy.deepcopy(bnReasoner)  # why the fuck here
d_sep.execute()



