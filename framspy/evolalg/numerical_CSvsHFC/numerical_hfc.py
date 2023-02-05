from pandas import DataFrame
import time
import numpy as np

from evolalg.hfc_base.experiment_hfc import ExperimentHFC
from evolalg.utils import get_state_filename
from evolalg.structures.population_methods import reinitialize_population_with_random_numerical, fill_population_with_random_numerical
from evolalg.mutation import simple_numerical_mutation
from evolalg.crossover import simple_numerical_crossover


class ExperimentNumericalHFC(ExperimentHFC):
    def __init__(
            self, popsize, hof_size, number_of_populations, migration_interval, save_only_best,
            benchmark_function, dimensions: int, results_directory_path: str
    ):
        super().__init__(popsize, hof_size, number_of_populations, migration_interval, save_only_best)
        self.benchmark_function = benchmark_function
        self.dimensions = dimensions
        self.results_directory_path = results_directory_path

    def evolve(
            self, hof_savefile, generations, initialgenotype, pmut, pxov, tournament_size,
            try_from_saved_file: bool = True  # to enable in-code disabling of loading saved savefile
    ):
        self.setup_evolution(hof_savefile, initialgenotype, try_from_saved_file)
        for pop_idx in range(len(self.populations)):
            self.populations[pop_idx] = reinitialize_population_with_random_numerical(
                population=self.populations[pop_idx],
                dimensions=self.dimensions,
                evaluate=self.evaluate
            )

        df = DataFrame(columns=['generation', 'total_popsize', 'worst_fitness', 'best_fitness'])

        # CALIBRATION STAGE
        pool_of_all_individuals = []
        [pool_of_all_individuals.extend(p.population) for p in self.populations]
        fitnesses_of_individuals = [individual.fitness for individual in pool_of_all_individuals]
        self.admission_thresholds[0] = -np.inf
        # EXAMPLE - Passing parameters for HFC-ADM
        self.admission_thresholds[1], self.admission_thresholds[-1] = self.get_bounds(
            pool_of_all_individuals, fitnesses_of_individuals,
            set_worst_to_fixed=True, set_best_to_stdev=True
        )
        lower_bound = self.admission_thresholds[1]
        upper_bound = self.admission_thresholds[-1]
        population_width = (upper_bound - lower_bound) / self.number_of_populations - 2
        for i in range(2, self.number_of_populations):
            self.admission_thresholds[i] = lower_bound + (i - 1) * population_width

        time0 = time.process_time()
        for g in range(self.current_generation, generations):
            for p in self.populations:
                p.population = self.make_new_population(p.population, pmut, pxov, tournament_size)

            if g % self.migration_interval == 0:
                self.migrate_populations()

            pool_of_all_individuals = []
            [pool_of_all_individuals.extend(p.population) for p in self.populations]
            self.update_stats(g, pool_of_all_individuals)
            cli_stats = self.get_cli_stats()
            df.loc[len(df)] = [cli_stats[0], cli_stats[1], cli_stats[2][0], cli_stats[3][0]]

        return self.hof, self.stats, df

    def add_to_worst(self):
        self.populations[0] = fill_population_with_random_numerical(
            population=self.populations[0],
            dimensions=self.dimensions,
            evaluate=self.evaluate
        )

    def evaluate(self, genotype):
        if any(x < -100 or x > 100 for x in genotype):
            return -np.inf
        cec2017_genotype = np.array([genotype])
        return self.benchmark_function(cec2017_genotype)

    def mutate(self, gen1):
        return simple_numerical_mutation(gen1)

    def cross_over(self, gen1, gen2):
        return simple_numerical_crossover(gen1, gen2)
