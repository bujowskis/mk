import numpy as np
import copy
import random
from evolalg.structures.individual_methods import ensure_numerical_within_space


def simple_numerical_mutation(genotype, change_magnitude: float = 5.0):
    """
    Changes one dimension by random number in range <-change_magnitude, change_magnitude>
    """
    output = copy.deepcopy(genotype)
    output[np.random.randint(len(output))] += random.uniform(-change_magnitude, change_magnitude)

    return output


def cec2017_numerical_mutation(genotype, standard_deviation_fraction: float = 0.1):  # fixme - sd value
    """
    Changes all dimensions with displacement according to normal distribution
    """
    raw_individual = [genotype[i] + np.random.normal(0, 100*standard_deviation_fraction) for i in range(len(genotype))]
    return ensure_numerical_within_space(genotype=raw_individual, lower=-100, upper=100)
