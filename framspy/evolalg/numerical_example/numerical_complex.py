import copy

import numpy as np
import random
import benchmark_functions as bf

from ..base.experiment_abc import ExperimentABC


class ExperimentNumericalComplex(ExperimentABC):
    """
    Note that genotype is a list of flot with length equal to the no. of dimensions of the chosen function
    """
    def __init__(self, hof_size, popsize, save_only_best, benchmark_function: bf.BenchmarkFunction) -> None:
        ExperimentABC.__init__(self, popsize=popsize, hof_size=hof_size, save_only_best=save_only_best)
        self.benchmark_function = benchmark_function

    def mutate(self, gen1):
        # todo - remove assert
        # print(random.uniform(-5.0, 5.0) for _ in range(len(gen1)))
        # print([random.uniform(-5.0, 5.0) for _ in range(len(gen1))])
        # print(gen1)
        output = gen1[:]
        output[np.random.randint(0, len(output))] += random.uniform(-5.0, 5.0)
        assert len(output) == len(gen1)
        return output

    def cross_over(self, gen1, gen2):
        if len(gen1) == 1:
            return gen1 if self.benchmark_function(gen1) > self.benchmark_function(gen2) else gen2
        division_point = np.random.randint(1, len(gen1))  # ensure at least 1 point from each
        # todo - remove assert
        output = gen1[:division_point] + gen2[division_point:] if random.getrandbits(1) == 0 else gen2[:division_point] + gen1[division_point:]
        assert len(output) == len(gen1)
        return output

    def evaluate(self, genotype):
        return self.benchmark_function(genotype)
