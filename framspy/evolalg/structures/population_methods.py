import random
import numpy as np
from typing import List

from population import PopulationStructures
from individual import Individual


def remove_excess_individuals_random(individuals: List[Individual], population_size: int) -> List[Individual]:
    """
    Removes a random sample of the individual, if there is more individuals than population_size

    :param individuals: list of individuals
    :param population_size: desired population size
    """
    # todo - tests
    if population_size < len(individuals):
        individuals = random.sample(individuals, population_size)

    return individuals

# todo (future) - more elegant implementation of keeping the desired no. of individuals within the subpopulations

# def fill_population_random_copy(population: PopulationStructures) -> Li


def fill_population_with_random_numerical(
        population: PopulationStructures, dimensions: int, upper_bound: float = -100, lower_bound: float = 100,
) -> PopulationStructures:
    """
    Fills the population with random individuals, sampled uniformly from lower to upper bound, within the given dimensions
    """
    pass
