import copy
from typing import Union
from BNReasoner import BNReasoner
from BayesNet import BayesNet


class D_Separation(BNReasoner):
    def __init__(self, net: Union[str, BayesNet], query: set, evidence: {}):
        super().__init__(net)
        self.query = query
        self.evidence = evidence



