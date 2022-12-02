# FIXME - relative import outside of package
from ..structures.individual import Individual
from ..structures.population import PopulationStructures
from ..frams_base.experiment_frams import ExperimentFrams
from ..base.experiment_island_model_abc import Experiment_Island


class ExperimentFramsIslands(Experiment_Island, ExperimentFrams):
    def __init__(self, frams_lib, optimization_criteria, hof_size, popsize, constraints, genformat,
                 number_of_populations, migration_interval, archive_size=0) -> None:
        super().__init__(frams_lib, optimization_criteria, hof_size, popsize, genformat, constraints)
        self.archive_size = archive_size
        self.number_of_populations = number_of_populations
        self.migration_interval = migration_interval

    def _initialize_evolution(self, initialgenotype):
        self.current_generation = 0
        self.timeelapsed = 0
        self.stats = []  # stores the best individuals, one from each generation across all populations
        initial_individual = Individual()
        initial_individual.setAndEvaluate(
            self.frams_getsimplest('1' if self.genformat is None else self.genformat, initialgenotype), self.evaluate)
        self.stats.append(initial_individual.rawfitness)
        [  # todo - vectorized "for"? (equivalent of for loop) - ADAM
            self.populations.append(
                PopulationStructures(
                    initial_individual=initial_individual,
                    archive_size=self.archive_size,
                    popsize=self.popsize)
            ) for _ in range(self.number_of_populations)
        ]
