import random
import numpy as np
from evolalg.structures.individual_methods import ensure_numerical_within_space


def simple_numerical_crossover(gen1, gen2):
    """
    Splits the genotypes in halves and combines randomly chosen first half and the other second half
    """
    division_point = len(gen1) // 2
    output = list(gen1[:division_point]) + list(gen2[division_point:]) if random.getrandbits(1) == 0 else \
        list(gen2[:division_point]) + list(gen1[division_point:])

    return output


def cec2017_numerical_crossover(gen1, gen2):  # NOTE - uniform -> no sd
    """
    Returns genotype that lies in a random place on a straight line drawn between parents
    """
    # https://stackoverflow.com/questions/55537838/given-two-points-of-4-or-more-dimensions-is-it-possible-to-find-a-line-equation
    straight_line_function = lambda g: np.array(gen1) + g * (np.array(gen2) - np.array(gen1))  # NOTE - g = 0.0 results in gen1, g = 1.0 results in gen2
    raw_individual = straight_line_function(random.uniform(0.0, 1.0))
    return ensure_numerical_within_space(genotype=raw_individual, lower=-100, upper=100)
