import random
from typing import List

from evolalg.structures.population import PopulationStructures
from evolalg.structures.individual import Individual


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
        population: PopulationStructures, dimensions: int, evaluate,
        upper_bound: float = -100, lower_bound: float = 100
) -> PopulationStructures:
    """
    Fills the population with random individuals, sampled uniformly from lower to upper bound, within the given dimensions
    """
    difference_from_target_size = len(population.population) - population.population_size
    if difference_from_target_size < 0:
        individuals = [Individual() for _ in range(-difference_from_target_size)]
        for individual in individuals:
            individual.set_and_evaluate(
                genotype=[random.uniform(lower_bound, upper_bound) for _ in range(dimensions)],
                evaluate=evaluate
            )
        population.population.extend(individuals)  # FIXME (future) - add evaluate() as individual method

    assert len(population.population) >= population.population_size
    return population


def reinitialize_population_with_random_numerical(
        population: PopulationStructures, dimensions: int, evaluate,
        upper_bound: float = -100, lower_bound: float = 100
) -> PopulationStructures:
    """
    Wipes the current population's individuals and fills it with randomly sampled numerical ones
    """
    population.population = []
    population = fill_population_with_random_numerical(population, dimensions, evaluate, upper_bound, lower_bound)

    assert len(population.population) >= population.population_size
    return population

def get_random_frams_solution(population: PopulationStructures, evaluate) -> PopulationStructures:
    # TODO 
    return population


def reinitialize_population_with_random_frams(population: PopulationStructures, evaluate) -> PopulationStructures:
    pass