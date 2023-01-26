import random
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


def fill_population_with_random_numerical(
        population: PopulationStructures, dimensions: int, upper_bound: float = -100, lower_bound: float = 100
) -> PopulationStructures:
    """
    Fills the population with random individuals, sampled uniformly from lower to upper bound, within the given dimensions
    """
    difference_from_target_size = len(population.population) - population.population_size
    if difference_from_target_size < 0:
        population.population.extend([[random.uniform(lower_bound, upper_bound) for _ in range(dimensions)] for _ in range(difference_from_target_size)])
    
    return population


def reinitialize_population_with_random_numerical(
        population: PopulationStructures, dimensions: int, upper_bound: float = -100, lower_bound: float = 100
) -> PopulationStructures:
    """
    Wipes the current population's individuals and fills it with randomly sampled numerical ones
    """
    population.population = []
    population = fill_population_with_random_numerical(population, dimensions, upper_bound, lower_bound)

    return population
