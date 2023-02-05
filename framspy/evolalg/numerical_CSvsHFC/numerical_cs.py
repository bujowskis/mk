from pandas import DataFrame, Series
import numpy as np
from evolalg.cs_base.experiment_convection_selection import ExperimentConvectionSelection
from evolalg.structures.population_methods import reinitialize_population_with_random_numerical
from evolalg.mutation import simple_numerical_mutation
from evolalg.crossover import simple_numerical_crossover


class ExperimentNumericalCSRun(ExperimentConvectionSelection):
    def __init__(
            self, popsize, hof_size, number_of_populations, migration_interval, save_only_best,
            benchmark_function, results_directory_path, dimensions
    ):
        super().__init__(popsize, hof_size, number_of_populations, migration_interval, save_only_best)
        self.benchmark_function = benchmark_function
        self.results_directory_path = results_directory_path
        self.dimensions = dimensions

    def evolve(
            self, hof_savefile, generations, initialgenotype, pmut, pxov, tournament_size,
            try_from_saved_file: bool = True  # to enable in-code disabling of loading saved savefile
    ):
        self.setup_evolution(hof_savefile, initialgenotype, try_from_saved_file)
        for pop_idx in range(len(self.populations)):
            self.populations[pop_idx] = reinitialize_population_with_random_numerical(
                population=self.populations[pop_idx], dimensions=self.dimensions, evaluate=self.evaluate
            )

        df = DataFrame(columns=['generation', 'total_popsize', 'worst_fitness', 'best_fitness'])

        for g in range(self.current_generation, generations):
            for p in self.populations:
                p.population = self.make_new_population(p.population, pmut, pxov, tournament_size)

            if g % self.migration_interval == 0:
                self.migrate_populations()

            pool_of_all_individuals = []
            [pool_of_all_individuals.extend(p.population) for p in self.populations]
            self.update_stats(g, pool_of_all_individuals)
            cli_stats = self.get_cli_stats()
            df.loc[len(df)] = [cli_stats[0], cli_stats[1], cli_stats[2], cli_stats[3]]

        return self.hof, self.stats, df

    def cross_over(self, gen1, gen2):
        return simple_numerical_crossover(gen1, gen2)

    def evaluate(self, genotype):
        if any(x < -100 or x > 100 for x in genotype):
            return -np.inf
        cec2017_genotype = np.array([genotype])
        return self.benchmark_function(cec2017_genotype)

    def mutate(self, gen1):
        return simple_numerical_mutation(gen1)
