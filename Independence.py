import copy
from typing import Union
from BNReasoner import BNReasoner
from BayesNet import BayesNet
from DSeparation import DSeparation


class Independence(BNReasoner):
    def __init__(self, net: Union[str, BayesNet], x: set, y: set, z: set):

        super().__init__(net)
        self.DSep = DSeparation(net, x, y, z)
        self.X = x
        self.Y = y
        self.Z = z

    def execute(self):
        return self.DSep.execute()