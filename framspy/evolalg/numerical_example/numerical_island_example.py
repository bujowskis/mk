import numpy as np
from ..structures.hall_of_fame import HallOfFame
from ..base.experiment_island_model_abc import Experiment_Island

BAD_FITNESS = None

class ExperimentNumericalIsland(Experiment_Island):
    def __init__(self, hof_size, popsize, number_of_populations, migration_interval,archive_size=0) -> None:
        self.hof = HallOfFame(hof_size)
        self.popsize = popsize
        self.number_of_populations = number_of_populations
        self.migration_interval = migration_interval
        self.archive_size = archive_size
    
    def mutate(self, gen1):
        return gen1 + np.random.randint(-10,10,len(gen1))

    def cross_over(self, gen1, gen2):
        return gen1

    def evaluate(self, genotype):
        return 1/sum([x*x for x in genotype])

    def save_genotypes(self, filename):
        """Implement if you want to save finall genotypes,in default implementation this function is run once at the end of evolution"""
        pass