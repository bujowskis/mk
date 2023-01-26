import random


def simple_numerical_crossover(gen1, gen2):
    """
    Splits the genotypes in halves and combines randomly chosen first half and the other second half
    """
    division_point = len(gen1) // 2
    output = list(gen1[:division_point]) + list(gen2[division_point:]) if random.getrandbits(1) == 0 else \
        list(gen2[:division_point]) + list(gen1[division_point:])

    return output
