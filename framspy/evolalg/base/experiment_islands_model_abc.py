import time
from abc import ABC
from typing import List

from ..structures.individual import Individual
from ..structures.population import PopulationStructures
from .experiment_abc import ExperimentABC


class ExperimentIslands(ExperimentABC, ABC):

    number_of_populations = 5
    popsize = 100
    populations: List[PopulationStructures] = []
    migration_interval = 10

    def __init__(self, popsize, hof_size, number_of_populations, migration_interval, save_only_best) -> None:
        super().__init__(popsize=popsize, hof_size=hof_size, save_only_best=save_only_best)
        self.number_of_populations=number_of_populations
        self.migration_interval=migration_interval

    def migrate_populations(self):
        print("Performing base migration")
        pool_of_all_individuals = []
        for p in self.populations:
            pool_of_all_individuals.extend(p.population)
        print(f"Pool of individuals: {len(pool_of_all_individuals)}")
        sorted_individuals = sorted(
            pool_of_all_individuals, key=lambda x: x.rawfitness)
        print(f"Best indiviudal for new islands:")
        for i in range(self.number_of_populations):
            shift = i*self.popsize
            self.populations[i].population = sorted_individuals[shift:shift+self.popsize]
            print(i, self.populations[i].population[-1].rawfitness)

    def initialize_evolution(self, initialgenotype):
        self.current_generation = 0
        self.time_elapsed = 0
        self.stats = []  # stores the best individuals, one from each generation
        initial_individual = Individual()
        initial_individual.set_and_evaluate(initialgenotype, self.evaluate)
        self.stats.append(initial_individual.rawfitness)
        [self.populations.append(PopulationStructures(initial_individual=initial_individual,
                                                      popsize=self.popsize))
         for _ in range(self.number_of_populations)]

    def get_state(self):
        return [self.time_elapsed, self.current_generation, self.populations, self.hof, self.stats]

    def set_state(self, state):
        self.time_elapsed, self.current_generation, self.populations, hof_, self.stats = state
        # sorting: ensure that we add from worst to best so all individuals are added to HOF
        for h in sorted(hof_, key=lambda x: x.rawfitness):
            self.hof.add(h)

    def evolve(self, hof_savefile, generations, initialgenotype, pmut, pxov, tournament_size):
        file_name = self.get_state_filename(hof_savefile)
        state = self.load_state(file_name)
        if state is not None:  # loaded state from file
            # saved generation has been completed, start with the next one
            self.current_generation += 1
            print("...Resuming from saved state: population size = %d, hof size = %d, stats size = %d, generation = %d/%d" % (len(self.populations[0].population), len(
                self.hof), len(self.stats), self.current_generation, generations))  # self.current_generation (and g) are 0-based, parsed_args.generations is 1-based
        else:
            self.initialize_evolution(initialgenotype)
        time0 = time.process_time()
        for g in range(self.current_generation, generations):
            for p in self.populations:
                p.population = self.make_new_population(
                    p.population, pmut, pxov, tournament_size)

            if g % self.migration_interval == 0:
                print("---------Start of migration-------")
                self.migrate_populations()
                print("---------End of migration---------")

            pool_of_all_individuals = []
            [pool_of_all_individuals.extend(p.population)
             for p in self.populations]
            self.update_stats(g, pool_of_all_individuals)
            if hof_savefile is not None:
                self.current_generation = g
                self.time_elapsed += time.process_time() - time0
                self.save_state(file_name)

        if hof_savefile is not None:
            self.save_genotypes(hof_savefile)

        return self.hof, self.stats

    @staticmethod
    def get_args_for_parser():
        parser = ExperimentABC.get_args_for_parser()

        parser.add_argument("-islands",type=int, default=5,
                            help="Number of subpopulations (islands)")
        parser.add_argument("-generations_migration",type=int, default=10,
                            help="Number of generations separating migration events when genotypes migrate between subpopulations (islands)")
        return parser