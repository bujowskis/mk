# TODO - relative import outside package
from ..structures.individual import Individual
from ..structures.population import PopulationStructures
from ..base.experiment_niching_abc import ExperimentNiching, DeapFitness
from .experiment_frams import STATS_SAVE_ONLY_BEST_FITNESS, ExperimentFrams


class ExperimentFramsNiching(ExperimentFrams, ExperimentNiching):
    def __init__(self, frams_lib, optimization_criteria, hof_size, popsize, constraints, normalize, dissim, fit, genformat, archive_size) -> None:
        super().__init__(frams_lib, optimization_criteria, hof_size, popsize, constraints)
        self.normalize = normalize
        self.dissim = dissim
        self.fit = fit
        self.genformat = genformat
        self.archive_size = archive_size

    # TODO - signature doesn't match
    def _initialize_evolution(self, genformat, initialgenotype):
        self.current_generation = 0
        self.timeelapsed = 0
        self.stats = []  # stores the best individuals, one from each generation
        initial_individual = Individual()
        initial_individual.setAndEvaluate(self.frams_getsimplest('1' if genformat is None else genformat, initialgenotype), self.evaluate)
        self.hof.add(initial_individual)
        self.stats.append(initial_individual.rawfitness if STATS_SAVE_ONLY_BEST_FITNESS else initial_individual)
        self.current_population = PopulationStructures(initial_individual=initial_individual,archive_size=self.archive_size,popsize=self.popsize)
        if self.fit == "nsga2":
            self.do_nsga2_dissim(self.current_population.population)
        if self.fit == "nslc":
            self.do_nslc_dissim(self.current_population.population)

    def dissimilarity(self, population):
        return self.frams_lib.dissimilarity([i.genotype for i in population], self.dissim)
