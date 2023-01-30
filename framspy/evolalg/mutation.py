import numpy as np
import copy
import random


def simple_numerical_mutation(genotype, change_magnitude: float = 5.0):
    """
    Changes one dimension by random number in range <-change_magnitude, change_magnitude>
    """
    output = copy.deepcopy(genotype)
    output[np.random.randint(len(output))] += random.uniform(-change_magnitude, change_magnitude)  # fixme - make into normal distribution, not uniform

    return output
