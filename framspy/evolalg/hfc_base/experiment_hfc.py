import time
from abc import ABC
from typing import List
from numpy import inf, std

from evolalg.cs_base.experiment_convection_selection import ExperimentConvectionSelection

from ..utils import get_state_filename


class ExperimentHFC(ExperimentConvectionSelection, ABC):
    admission_buffers: List[List]
    admission_thresholds: List[float]

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
        self.admission_buffers = [[] for _ in range(self.number_of_populations)]
        self.admission_thresholds = [-inf for _ in range(self.number_of_populations)]

    def migrate_populations(self):
        """
        HFC "only up" admission threshold migration
        """
        self.recalculate_admission_thresholds()

        # remove / move to admission buffers
        for pop_idx in range(self.number_of_populations):
            for individual in self.populations[pop_idx].population:
                if individual.fitness < self.admission_thresholds[pop_idx]:  # fitness so bad individual doesn't belong
                    self.populations[pop_idx].population.remove(individual)
                    break
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

    def recalculate_admission_thresholds(self):
        """
        Called after evolve(), works on initialized admission thresholds
        Recalculates the admission thresholds according to equiwidth scheme, leaving entry and avg threshold untouched
        """
        # FIXME - potential problem of widen range
        pool_of_all_individuals = []
        [pool_of_all_individuals.extend(p.population) for p in self.populations]
        lower_bound: float = self.admission_thresholds[1]
        upper_bound = max(pool_of_all_individuals, key=lambda x: x.fitness)
        population_width = (upper_bound - lower_bound) / self.number_of_populations - 2
        for i in range(2, self.number_of_populations):
            self.admission_thresholds[i] = lower_bound + (i - 1) * population_width

    def evolve(
            self, hof_savefile, generations, initialgenotype, pmut, pxov, tournament_size,
            try_from_saved_file: bool = True  # to enable in-code disabling of loading saved savefile
    ):
        self.setup_evolution(hof_savefile, initialgenotype, try_from_saved_file)

        # CALIBRATION STAGE - HFC-ADM
        pool_of_all_individuals = []
        [pool_of_all_individuals.extend(p.population) for p in self.populations]

        fitnesses_of_individuals = [individual.fitness for individual in pool_of_all_individuals]
        avg_random_fitness = sum(fitnesses_of_individuals)/len(pool_of_all_individuals)
        self.admission_thresholds[0], self.admission_thresholds[1] = -inf, avg_random_fitness
        self.admission_thresholds[-1] = max(fitnesses_of_individuals) - std(fitnesses_of_individuals)

        lower_bound = avg_random_fitness
        upper_bound = self.admission_thresholds[-1]
        population_width = (upper_bound - lower_bound) / self.number_of_populations - 2 
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
                self.save_state(get_state_filename())

        if hof_savefile is not None:
            self.save_genotypes(hof_savefile)

        return self.hof, self.stats
