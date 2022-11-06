from typing import List
import Population


class ConvectionSelectionInterface:
    """
    Interface implemented by Evolutionary Algorithms that enables Convection Selection to use them.
    """
    populations: List[Population]

    def migrate(self):
        ...
