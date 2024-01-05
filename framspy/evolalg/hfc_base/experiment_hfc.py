import time
from abc import ABC, abstractmethod
from typing import List
from numpy import inf, std

from evolalg.cs_base.experiment_convection_selection import ExperimentConvectionSelection

from evolalg.utils import get_state_filename


class ExperimentHFC(ExperimentConvectionSelection, ABC):
    """
    Implementation of synchronous HFC, tailored to be comparable with CS
    """
    def __init__(self, popsize, hof_size, number_of_populations, migration_interval, save_only_best) -> None:
        # todo - input validation
        super().__init__(
            popsize=popsize,
            hof_size=hof_size,
            number_of_populations=number_of_populations,
            migration_interval=migration_interval,
            save_only_best=save_only_best
        )
        self.admission_buffers: List[List] = [[] for _ in range(self.number_of_populations)]
        self.admission_thresholds: List[float] = [-inf for _ in range(self.number_of_populations)]

    def migrate_populations(self):
        """
        HFC "only up" admission threshold migration
        """
        self.recalculate_admission_thresholds()

        # remove / move to admission buffers
        for pop_idx in range(self.number_of_populations):
            for individual in self.populations[pop_idx].population:
                # we don't explicitly remove "broken" individuals from their population
                for admission_threshold_idx in reversed(range(pop_idx+1, self.number_of_populations)):
                    if individual.fitness >= self.admission_thresholds[admission_threshold_idx]:
                        self.admission_buffers[admission_threshold_idx].append(individual)
                        self.populations[pop_idx].population.remove(individual)
                        break

        # handle admission buffers
        for pop_idx in range(self.number_of_populations):
            self.admission_buffers[pop_idx] = sorted(self.admission_buffers[pop_idx], key=lambda i: i.fitness, reverse=True)[:self.populations[pop_idx].population_size//2]
            self.populations[pop_idx].population.extend(self.admission_buffers[pop_idx])
            self.populations[pop_idx].population = sorted(self.populations[pop_idx].population, key=lambda i: i.fitness, reverse=True)[:self.populations[pop_idx].population_size]

        # add random individuals to the worst subpopulation
        self.add_to_worst()

    @abstractmethod
    def add_to_worst(self):
        """
        Handles adding new random individuals to the worst subpopulation
        """
        pass

    def recalculate_admission_thresholds(self):
        """
        Called after evolve(), works on initialized admission thresholds
        Recalculates the admission thresholds according to equiwidth scheme, leaving entry and avg threshold untouched
        """
        pool_of_all_individuals = []
        [pool_of_all_individuals.extend(p.population) for p in self.populations]
        fitnesses_of_individuals = [individual.fitness for individual in pool_of_all_individuals]
        # EXAMPLE - Passing parameters for HFC-ADM
        lower_bound, upper_bound = self.get_bounds(
            pool_of_all_individuals, fitnesses_of_individuals,
            set_worst_to_fixed=True, set_best_to_stdev=False
        )
        lower_bound = self.admission_thresholds[1]
        population_width = (upper_bound - lower_bound) / (self.number_of_populations - 1)
        for i in range(2, self.number_of_populations):
            self.admission_thresholds[i] = lower_bound + (i - 1) * population_width

    def get_bounds(self, pool_of_all_individuals, fitnesses_of_individuals, set_worst_to_fixed: bool, set_best_to_stdev: bool):
        # HFC-ADM approach
        if set_worst_to_fixed and set_best_to_stdev:
            avg_random_fitness = sum(fitnesses_of_individuals)/len(pool_of_all_individuals)
            lower_bound = avg_random_fitness
            upper_bound = max(fitnesses_of_individuals) - std(fitnesses_of_individuals)
        # Equiwidth approach
        elif not set_worst_to_fixed and not set_best_to_stdev:
            lower_bound = min(fitnesses_of_individuals)
            upper_bound = max(fitnesses_of_individuals)
        # Something in between HFC-ADM and Equiwidth approaches
        elif set_worst_to_fixed and not set_best_to_stdev:
            avg_random_fitness = sum(fitnesses_of_individuals) / len(pool_of_all_individuals)
            lower_bound = avg_random_fitness
            upper_bound = max(fitnesses_of_individuals)
        else:
            raise Exception('not implemented')

        return lower_bound, upper_bound

    def evolve(
            self, hof_savefile, generations, initialgenotype, pmut, pxov, tournament_size,
            try_from_saved_file: bool = True  # to enable in-code disabling of loading saved savefile
    ):
        self.setup_evolution(hof_savefile, initialgenotype, try_from_saved_file)

        # CALIBRATION STAGE
        pool_of_all_individuals = []
        [pool_of_all_individuals.extend(p.population) for p in self.populations]
        fitnesses_of_individuals = [individual.fitness for individual in pool_of_all_individuals]
        self.admission_thresholds[0] = -inf
        # EXAMPLE - Passing parameters for HFC-ADM
        self.admission_thresholds[1], self.admission_thresholds[-1] = self.get_bounds(
            pool_of_all_individuals, fitnesses_of_individuals,
            set_worst_to_fixed=False, set_best_to_stdev=False
        )
        lower_bound = self.admission_thresholds[1]
        upper_bound = self.admission_thresholds[-1]
        population_width = (upper_bound - lower_bound) / (self.number_of_populations - 1)
        for i in range(2, self.number_of_populations):
            self.admission_thresholds[i] = lower_bound + (i-1)*population_width

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
                self.current_generation = g
                self.time_elapsed += time.process_time() - time0
                self.save_state(get_state_filename(hof_savefile))

        if hof_savefile is not None:
            self.save_genotypes(hof_savefile)

        return self.hof, self.stats
