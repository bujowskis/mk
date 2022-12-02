from abc import ABC
from typing import List
import time

# TODO - resolve namespace package
from ..structures.individual import Individual
from ..structures.population import PopulationStructures
from .experiment_abc import ExperimentABC


# FIXME - CamelCase
class Experiment_Island(ExperimentABC, ABC):
    # TODO - init with the params below? (__init__)
    number_of_populations: int = 5
    popsize: int = 100
    populations: List[PopulationStructures] = []
    migration_interval = 10  # TODO - int OR method of migration? (fixed number of generations, event-driven, etc.)

    def migrate_populations(self):
        pool_of_all_individuals = []
        for p in self.populations:
            pool_of_all_individuals.extend(p.population)

        sorted_individuals = sorted(pool_of_all_individuals, key=lambda x: x.rawfitness)
        
        for i in range(self.number_of_populations):
            shift = i*self.popsize
            self.populations[i].population = sorted_individuals[shift:shift+self.popsize]

    def _initialize_evolution(self, initialgenotype):
        self.current_generation = 0
        self.timeelapsed = 0
        self.stats = []  # stores the best individuals, one from each generation
        initial_individual = Individual()
        initial_individual.setAndEvaluate(initialgenotype, self.evaluate)
        self.stats.append(initial_individual.rawfitness)
        [self.populations.append(PopulationStructures(initial_individual=initial_individual,
                                                      archive_size=self.archive_size,
                                                      popsize=self.popsize))
        for _ in range(self.number_of_populations)]

    def get_state(self):
        return [self.timeelapsed, self.current_generation, self.populations, self.hof, self.stats]

    def set_state(self, state):
        # FIXME - outside __init__
        self.timeelapsed, self.current_generation, self.populations, hof_, self.stats = state
        for h in sorted(hof_, key=lambda x: x.rawfitness):  # sorting: ensure that we add from worst to best so all individuals are added to HOF
            self.hof.add(h)

    def evolve(self, hof_savefile, generations, initialgenotype, pmut, pxov, tournament_size):
        file_name = self.get_state_filename(hof_savefile)
        state = self.load_state(file_name)
        if state is not None:  # loaded state from file
            self.current_generation += 1  # saved generation has been completed, start with the next one
            print("...Resuming from saved state: population size = %d, hof size = %d, stats size = %d, archive size = %d, generation = %d/%d" % (len(self.populations[0].population), len(self.hof), len(self.stats),  (len(self.populations[0].archive)),self.current_generation, generations))  # self.current_generation (and g) are 0-based, parsed_args.generations is 1-based
        else:
            self._initialize_evolution(initialgenotype)

        time0 = time.process_time()
        for g in range(self.current_generation, generations):
            for p in self.populations:  # evolve each population
                p.population = self.make_new_population(p.population, pmut, pxov, tournament_size)

            if g % self.migration_interval == 0:
                self.migrate_populations()

            # TODO - for sure could be optimized, instead of
            pool_of_all_individuals = []
            [pool_of_all_individuals.extend(p.population) for p in self.populations]
            self.update_stats(g, pool_of_all_individuals)

            if hof_savefile is not None:
                self.timeelapsed += time.process_time() - time0
                self.save_state(file_name) 

        return self.current_population.population, self.stats
