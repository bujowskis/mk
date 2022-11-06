from typing import List
import IndividualInterface


class Population:
    """
    Interface implemented by Populations that enables Evolutionary Algorithms to use them.
    """
    individuals: List[IndividualInterface]

    def __init__(self, individuals: List[IndividualInterface]):
        """
        :param individuals: list of individuals in the population
        """
        self.individuals = individuals

    def get_individuals(self) -> List[IndividualInterface]:
        """
        Gets the population's individuals
        :return: list of individuals in the population
        """
        return self.individuals

    def set_individuals(self, individuals: List[IndividualInterface]):
        """
        Sets the population's individuals
        :param individuals: list of individuals
        """
        self.individuals = individuals

    # TODO - get_sorted_individuals() OR sorted(get_individuals) ?
