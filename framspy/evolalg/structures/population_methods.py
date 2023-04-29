import random
from typing import List

from evolalg.structures.population import PopulationStructures
from evolalg.structures.individual import Individual
from FramsticksLib import FramsticksLib
from evolalg.frams_base.experiment_frams import ExperimentFrams


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


def fill_population_with_random_frams(experiment_frams: ExperimentFrams, framslib: FramsticksLib, genformat: any, population: PopulationStructures, evaluate, constraints, initial_genotype=None) -> PopulationStructures:
    """
    Fills the population with random Framsticks individuals
    """
    initial_genotype = experiment_frams.frams_getsimplest(genetic_format=genformat, initial_genotype=initial_genotype)
    difference_from_target_size = len(population.population) - population.population_size
    parts_min = 2
    parts_max = constraints['max_numparts']
    neurons_min = 1
    neurons_max = constraints['max_numneurons']
    iter_max = 10
    if difference_from_target_size < 0:
        genotypes = [framslib.getRandomGenotype(initial_genotype=initial_genotype, 
                        parts_min=parts_min, parts_max=parts_max, neurons_min=neurons_min,
                        neurons_max=neurons_max, iter_max=iter_max, return_even_if_failed=True) \
                        for _ in range(-difference_from_target_size)]
        individuals = [Individual() for _ in genotypes]
        for idx, individual in enumerate(individuals):
            individual.set_and_evaluate(
                genotype=genotypes[idx],
                evaluate=evaluate
            )
            individual.innovation_in_time = [0.0 for _ in range(experiment_frams.number_of_epochs)]
            individual.innovation_in_time[experiment_frams.current_epoch] = 1.0
            individual.contributor_spops = [0.0 for _ in range(experiment_frams.number_of_populations+1)]

        population.population.extend(individuals)

    assert len(population.population) >= population.population_size
    return population


def reinitialize_population_with_random_frams(experiment_frams: ExperimentFrams, framslib: FramsticksLib, genformat: any, population: PopulationStructures, evaluate, constraints, initial_genotype=None) -> PopulationStructures:
    """
    Wipes the current population's individuals and fills it with randomly sampled Framsticks individuals
    """
    population.population = []
    population = fill_population_with_random_frams(experiment_frams, framslib, genformat, population, evaluate, constraints, initial_genotype)

    assert len(population.population) >= population.population_size
    return population