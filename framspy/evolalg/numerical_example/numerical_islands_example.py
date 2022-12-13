import numpy as np

from ..base.experiment_islands_model_abc import ExperimentIslands
from ..structures.hall_of_fame import HallOfFame


class ExperimentNumericalIslands(ExperimentIslands):
    def __init__(self, hof_size, popsize, number_of_populations, migration_interval, save_only_best) -> None:
        ExperimentIslands.__init__(self,popsize=popsize,
                                hof_size=hof_size,
                                number_of_populations=number_of_populations,
                                migration_interval=migration_interval,
                                save_only_best=save_only_best)

    def mutate(self, gen1):
        return gen1 + np.random.randint(-10, 10, len(gen1))

    def cross_over(self, gen1, gen2):
        return gen1

    def evaluate(self, genotype):
        return 1/sum([x*x for x in genotype])
