from ..base.experiment_niching_abc import ExperimentNiching
from ..structures.individual import Individual
from ..structures.population import PopulationStructures
from ..utils import merge_two_parsers
from .experiment_frams import ExperimentFrams


class ExperimentFramsNiching(ExperimentFrams, ExperimentNiching):
    def __init__(self, frams_lib, optimization_criteria, hof_size, popsize, constraints, normalize, dissim, fit, genformat, archive_size, save_only_best) -> None:
        ExperimentFrams.__init__(self, hof_size=hof_size,
                                 popsize=popsize,
                                 frams_lib=frams_lib,
                                 constraints=constraints,
                                 optimization_criteria=optimization_criteria,
                                 genformat=genformat,
                                 save_only_best=save_only_best
                                 )
        ExperimentNiching.__init__(self, hof_size=hof_size,
                                   popsize=popsize,
                                   fit=fit,
                                   normalize=normalize,
                                   save_only_best=save_only_best,
                                   archive_size=archive_size
                                   )
        self.dissim = dissim
        

    def initialize_evolution(self, genformat, initialgenotype):
        self.current_generation = 0
        self.time_elapsed = 0
        self.stats = []  # stores the best individuals, one from each generation
        initial_individual = Individual()
        initial_individual.set_and_evaluate(self.frams_getsimplest('1' if genformat is None else genformat, initialgenotype), self.evaluate)
        self.hof.add(initial_individual)
        self.stats.append(initial_individual.rawfitness if self.save_only_best else initial_individual)
        self.population_structures = PopulationStructures(initial_individual=initial_individual, archive_size=self.archive_size, popsize=self.popsize)
        if self.fit == "nsga2":
            self.do_nsga2_dissim(self.population_structures.population)
        if self.fit == "nslc":
            self.do_nslc_dissim(self.population_structures.population)

    def dissimilarity(self, population):
        return self.frams_lib.dissimilarity([i.genotype for i in population], self.dissim)


    @staticmethod
    def get_args_for_parser():
        p1 = ExperimentFrams.get_args_for_parser()
        p2 = ExperimentNiching.get_args_for_parser()
        return merge_two_parsers(p1, p2)

