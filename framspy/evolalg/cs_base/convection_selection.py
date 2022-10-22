from abc import ABC
from typing import List
import time

from ..structures.individual import Individual
from ..structures.population import PopulationStructures
from ..base.experiment_abc import ExperimentABC


# todo - import and use numpy arrays, refactor code

# todo - remove after done with experiment_frams_islands.py
# noinspection DuplicatedCode
class ExperimentIsland(ExperimentABC, ABC):
    # TODO - take these as arguments
    number_of_populations = 5
    popsize = 100
    migration_interval = 10
    populations: List[PopulationStructures] = []

    def migrate_populations(self):
        print("migration")
        pool_of_all_individuals = []
        for p in self.populations:
            pool_of_all_individuals.extend(p.population)

        # TODO - could override `Individual` compare core function
        sorted_individuals = sorted(pool_of_all_individuals, key=lambda x: x.rawfitness)

        for i in range(self.number_of_populations):
            shift = i * self.popsize
            self.populations[i].population = sorted_individuals[shift:shift + self.popsize]

    def _initialize_evolution(self, initial_genotype):
        self.current_generation = 0
        self.time_elapsed = 0
        self.stats = []  # stores the best individuals, one from each generation
        initial_individual = Individual(self.evaluate)
        initial_individual.setAndEvaluate(initial_genotype)
        self.stats.append(initial_individual.rawfitness)
        [self.populations.append(PopulationStructures(
            evaluate=self.evaluate,
            initial_individual=initial_individual,
            archive_size=self.archive_size,  # TODO - resolve archive_size
            popsize=self.popsize)
        ) for _ in range(self.number_of_populations)]

    def get_state(self):
        return [self.time_elapsed, self.current_generation, self.populations, self.hof, self.stats]

    def set_state(self, state):
        # todo - resolve time_elapsed out of scope
        self.time_elapsed, self.current_generation, self.populations, hof_, self.stats = state
        for h in sorted(hof_, key=lambda x: x.rawfitness):
            self.hof.add(h)  # sorting: ensure that we add from worst to best so all individuals are added to HOF

    def evolve(self, hof_savefile, generations, initial_genotype, pmut, pxov, tournament_size):
        file_name = self.get_state_filename(hof_savefile)
        state = self.load_state(file_name)
        if state is not None:  # loaded state from file
            self.current_generation += 1  # saved generation has been completed, start with the next one
            print(
                "...Resuming from saved state: population size = %d, hof size = %d, stats size = %d, archive size = %d, generation = %d/%d" % (
                    len(self.populations[0].population), len(self.hof), len(self.stats),
                    (len(self.populations[0].archive)),
                    self.current_generation,
                    generations)
            )  # self.current_generation (and g) are 0-based, parsed_args.generations is 1-based
        else:
            self._initialize_evolution(initial_genotype)
        time0 = time.process_time()
        for g in range(self.current_generation, generations):
            for p in self.populations:
                p.population = self.make_new_population(p.population, pmut, pxov, tournament_size)

            if g % self.migration_interval == 0:
                self.migrate_populations()

            pool_of_all_individuals = []
            [pool_of_all_individuals.extend(p.population) for p in self.populations]
            self.update_stats(g, pool_of_all_individuals)
            if hof_savefile is not None:
                self.time_elapsed += time.process_time() - time0
                self.save_state(file_name)

        return self.current_population.population, self.stats


"""
The main idea of the Convection Selection:
- Subpopulations (Slaves) - evolutionary algorithms in their own
    - similar fitness values between individuals in one Slave
- (Super)population (Master) - migrate individuals between Subpopulations
    - (all) genotypes sorted according to fitness

First method - equiWidth
- divide the fitness range equally between Slaves
- supposed the fitness range is [0, 1], 5 Slaves,
    - Slave1 = [0.0, 0.2]
    - Slave2 = (0.2, 0.4]
    - (...)
    - Slave5 = (0.8, 1.0]
- it may happen some Slave's population is empty
- worst-case scenario - it's the same as running a single evolutionary algorithm with no distribution

Second method - equiNumber
- divide the individuals equally between slaves
- supposed the population [i0, i1, (...), i9] is sorted according to fitness (worst-best)
    - Slave1 population = [i0, i1]
    - Slave2 population = [i2, i3]
    - (...)
    - Slave5 population = [i8, i9]
- the individuals are always spread equally between
- more complex
"""

# Initially:
# Slave1 state = [0.05, 0.08, 0.12]
# (...)

#   1. Slave1Step [0.0, 0.2] -> state = [0.1, 0.15, 0.23]
#   2. Slave2Step (0.2, 0.4]
#   (...)
#   5. Slave5Step (0.8, 1.0]
#   6. Master
#       6.1. Check Slave1 state = [0.1, 0.15, **0.23**]
#       6.2. Move "too good" individuals up -> Slave1 state = [0.1, 0.15, None]; Slave2 state = state + [0.23]
#       (6.3.) Inject new, random/some-other-way, into Slave1
#   (gen += 1)

"""
CS

class EA:
    self.population
    def step()

    def getState()
    def setState()
"""
