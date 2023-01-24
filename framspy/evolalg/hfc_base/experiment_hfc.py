import time
from abc import ABC
from typing import List

from evolalg.cs_base.experiment_convection_selection import ExperimentConvectionSelection

from ..utils import get_state_filename


class ExperimentHFC(ExperimentConvectionSelection, ABC):
    admission_buffers: List[List]
    admission_thresholds: List

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
        self.admission_buffers = [[] for _ in range(len(self.populations))]
        self.admission_thresholds = [None for _ in range(len(self.populations))]

    def migrate_populations(self):
        """
        HFC "only up" admission threshold migration
        """
        # admission_buffers = [[] for i in range(len(self.populations))]
        # admission_thresholds
        # todo
        ...

    def recalculate_admission_thresholds(self):
        """
        Called after evolve(), works on initialized admission thresholds
        Recalculates the admission thresholds according to equiwidth scheme, leaving entry and avg threshold untouched
        """
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

        # CALIBRATION STAGE
        # admission thresholds based on HFC
        pool_of_all_individuals = []
        [pool_of_all_individuals.extend(p.population) for p in self.populations]
        avg_random_fitness = sum([individual.fitness for individual in pool_of_all_individuals])/len(pool_of_all_individuals)
        self.admission_thresholds[0], self.admission_thresholds[1] = None, avg_random_fitness
        # admission thresholds based on equiwidth
        lower_bound = avg_random_fitness  # it makes no sense to go below average fitness of random individual
        upper_bound = max(pool_of_all_individuals, key=lambda x: x.fitness)
        population_width = (upper_bound - lower_bound) / self.number_of_populations - 2  # account for entry and first subpop
        for i in range(2, self.number_of_populations):
            self.admission_thresholds[i] = lower_bound + (i-1)*population_width
        # at this point, the admission thresholds are the following:
        # None, avg, avg + 1*population_width, avg + 2*population_width, ..., avg + (number_of_populations-1)*population_width

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
