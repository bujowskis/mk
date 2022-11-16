import time

from ..structures.individual import Individual
from ..structures.population import PopulationStructures
from ..base.experiment_niching_abc import ExperimentNiching, DeapFitness
from .experiment_frams import STATS_SAVE_ONLY_BEST_FITNESS, ExperimentFrams


class ExperimentFramsNiching(ExperimentFrams, ExperimentNiching):
    def __init__(self,frams_lib, optimization_criteria, hof_size, popsize, constraints, normalize, dissim, fit, genformat, archive_size) -> None:
        super().__init__(frams_lib, optimization_criteria, hof_size, popsize, constraints)
        self.normalize = normalize
        self.dissim = dissim
        self.fit = fit
        self.genformat = genformat
        self.archive_size = archive_size

    def _initialize_evolution(self, genformat, initialgenotype):
        self.current_genneration = 0
        self.timeelapsed = 0
        self.stats = []  # stores the best individuals, one from each generation
        initial_individual = Individual(self.evaluate)
        initial_individual.setAndEvaluate(self.frams_getsimplest('1' if genformat is None else genformat, initialgenotype))
        self.hof.add(initial_individual)
        self.stats.append(initial_individual.rawfitness if STATS_SAVE_ONLY_BEST_FITNESS else initial_individual)
        self.current_population = PopulationStructures(evaluate=self.evaluate,initial_individual=initial_individual,archive_size=self.archive_size,popsize=self.popsize)
        if self.fit == "nsga2":
            self.do_nsga2_dissim(self.current_population.population)
        if self.fit == "nslc":
            self.do_nslc_dissim(self.current_population.population)

    def evolve(self, hof_savefile, generations, initialgenotype, pmut, pxov, tournament_size):
        file_name = self.get_state_filename(hof_savefile)
        state = self.load_state(file_name)
        if state is not None:  # loaded state from file
            self.current_genneration += 1  # saved generation has been completed, start with the next one
            print("...Resuming from saved state: population size = %d, hof size = %d, stats size = %d, archive size = %d, generation = %d/%d" % (len(self.current_population.population), len(self.hof), len(self.stats),  (len(self.current_population.archive)),self.current_genneration, generations))  # self.current_genneration (and g) are 0-based, parsed_args.generations is 1-based
        else:
            self._initialize_evolution(self.genformat,initialgenotype)

        time0 = time.process_time()
        for g in range(self.current_genneration, generations):
            if self.fit != "raw" and self.fit !="nsga2" and self.fit !="nslc":
                self.do_niching(self.current_population)

            if type(self.current_population.population[0].fitness) == DeapFitness:
                self.current_population.population = self.make_new_population_nsga2(self.current_population.population, pmut, pxov)
            else:
                self.current_population.population = self.make_new_population(self.current_population.population, pmut, pxov, tournament_size)

            self.update_stats(g, self.current_population.population)
            
            if hof_savefile is not None:
                self.current_genneration=g
                self.timeelapsed += time.process_time() - time0
                self.save_state(file_name) 

        return self.current_population.population, self.stats
    
    def dissimilarity(self, population):
        return self.frams_lib.dissimilarity([i.genotype for i in population], self.dissim)