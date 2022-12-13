
from ..base.experiment_islands_model_abc import ExperimentIslands
from ..frams_base.experiment_frams import ExperimentFrams
from ..structures.individual import Individual
from ..structures.population import PopulationStructures
from ..utils import merge_two_parsers


class ExperimentFramsIslands(ExperimentIslands, ExperimentFrams):
    def __init__(self, frams_lib, optimization_criteria, hof_size,
                 popsize, constraints, genformat,
                 number_of_populations, migration_interval, save_only_best) -> None:
        ExperimentFrams.__init__(self, frams_lib=frams_lib, optimization_criteria=optimization_criteria,
                                 hof_size=hof_size, popsize=popsize,
                                 genformat=genformat, save_only_best=save_only_best, constraints=constraints)

        self.number_of_populations = number_of_populations
        self.migration_interval = migration_interval

    def initialize_evolution(self, initialgenotype):
        self.current_generation = 0
        self.time_elapsed = 0
        # stores the best individuals, one from each generation across all populations
        self.stats = []
        initial_individual = Individual()
        initial_individual.set_and_evaluate(self.frams_getsimplest(
            '1' if self.genformat is None else self.genformat, initialgenotype), self.evaluate)
        self.stats.append(initial_individual.rawfitness)
        self.populations= [PopulationStructures(initial_individual=initial_individual,
                                              popsize=self.popsize)
                         for _ in range(self.number_of_populations)]
    @staticmethod
    def get_args_for_parser():
        parser1 = ExperimentFrams.get_args_for_parser()
        parser2 = ExperimentIslands.get_args_for_parser()
        return merge_two_parsers(parser1, parser2)