import copy
from typing import Union
from BNReasoner import BNReasoner
from BayesNet import BayesNet

class DSeparation(BNReasoner):

    def __init__(self, net: Union[str, BayesNet]):

        super().__init__(net)

    def execute(self):
        bnReasoner = DSeparation('testing/lecture_example.BIFXML', X, Y, Z)

        return