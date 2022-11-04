from typing import List
import IndividualInterface


class Population:
    """
    Interface implemented by Populations that enables Evolutionary Algorithms to use them.
    """
    individuals: List[IndividualInterface]

    def get_population(self) -> List[IndividualInterface]:
        """
        :return: list of individuals in the population
        """
        return self.individuals

    def set_population(self) -> None:
        """"""
