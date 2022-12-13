import numpy as np

from ..base.experiment_abc import ExperimentABC
from ..structures.hall_of_fame import HallOfFame


class ExperimentNumerical(ExperimentABC):
    def __init__(self, hof_size, popsize, save_only_best) -> None:
        ExperimentABC.__init__(self,popsize=popsize,
                               hof_size=hof_size,
                               save_only_best=save_only_best)

    def mutate(self, gen1):
        return list(gen1 + np.random.randint(-10, 10, len(gen1)))

    def cross_over(self, gen1, gen2):
        return gen1

    def evaluate(self, genotype):
        return 1/sum([x*x for x in genotype])
