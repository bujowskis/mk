from abc import ABC
from typing import List
import time

from ..structures.individual import Individual
from ..structures.population import PopulationStructures
from .experiment_abc import ExperimentABC


class Experiment_Island(ExperimentABC, ABC):
    
    number_of_populations = 5
    popsize = 100
    populations: List[PopulationStructures] =[]
    migration_interval = 10

    def migrate_populations(self):
        print("migration")
        pool_of_all_individuals = []
        for p in self.populations:
            pool_of_all_individuals.extend(p.population)

        sorted_individuals = sorted(pool_of_all_individuals, key=lambda x: x.rawfitness)
        
        for i in range(self.number_of_populations):
            shift = i*self.popsize
            self.populations[i].population = sorted_individuals[shift:shift+self.popsize]


    def _initialize_evolution(self, initialgenotype):
        self.current_genneration = 0
        self.timeelapsed = 0
        self.stats = []  # stores the best individuals, one from each generation
        initial_individual = Individual(self.evaluate)
        initial_individual.setAndEvaluate(initialgenotype)
        self.stats.append(initial_individual.rawfitness)
        [self.populations.append(PopulationStructures(evaluate=self.evaluate,
                                                      initial_individual=initial_individual,
                                                      archive_size=self.archive_size,
                                                      popsize=self.popsize))
        for _ in range(self.number_of_populations)]

    def get_state(self):
        return [self.timeelapsed, self.current_genneration, self.populations, self.hof, self.stats]

    def set_state(self,state):
        self.timeelapsed, self.current_genneration, self.populations, hof_,self.stats = state
        for h in sorted(hof_, key=lambda x: x.rawfitness):  # sorting: ensure that we add from worst to best so all individuals are added to HOF
            self.hof.add(h)

    def evolve(self,hof_savefile, generations, initialgenotype, pmut, pxov, tournament_size):
        file_name = self.get_state_filename(hof_savefile)
        state = self.load_state(file_name)
        if state is not None:  # loaded state from file
            self.current_genneration += 1  # saved generation has been completed, start with the next one
            print("...Resuming from saved state: population size = %d, hof size = %d, stats size = %d, archive size = %d, generation = %d/%d" % (len(self.populations[0].population), len(self.hof), len(self.stats),  (len(self.populations[0].archive)),self.current_genneration, generations))  # self.current_genneration (and g) are 0-based, parsed_args.generations is 1-based
        else:
            self._initialize_evolution(initialgenotype)
        time0 = time.process_time()
        for g in range(self.current_genneration, generations):
            for p in self.populations:
                p.population = self.make_new_population(p.population, pmut, pxov, tournament_size)

            if g%self.migration_interval==0:
                self.migrate_populations()

            pool_of_all_individuals = []
            [pool_of_all_individuals.extend(p.population) for p in self.populations]
            self.update_stats(g,pool_of_all_individuals)
            if hof_savefile is not None:
                self.timeelapsed += time.process_time() - time0
                self.save_state(file_name) 

        return self.current_population.population, self.stats