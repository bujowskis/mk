from typing import List
import IndividualInterface


class ConvectionSelectionInterface:
    """
    Interface implemented by Evolutionary Algorithms that enables Convection Selection to use them.
    """
    number_of_populations: int
    populations: List[]

    def get_population(self) -> List[IndividualInterface]:
        """
        :return: population of a given CS subpopulation
        """
        pass


