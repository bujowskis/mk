import numpy as np
from ..base.experiment_abc import ExperimentABC, STATS_SAVE_ONLY_BEST_FITNESS
from ..structures.hall_of_fame import HallOfFame

BAD_FITNESS = None

class ExperimentNumerical(ExperimentABC):
    def __init__(self, hof_size, popsize) -> None:
        self.hof = HallOfFame(hof_size)
        self.popsize = popsize
    
    def mutate(self, gen1):
        return gen1 + np.random.randint(-10,10,len(gen1))

    def cross_over(self, gen1, gen2):
        return gen1

    def evaluate(self, genotype):
        return 1/sum([x*x for x in genotype])

    def save_genotypes(self, filename):
        """Implement if you want to save finall genotypes,in default implementation this function is run once at the end of evolution"""
        pass