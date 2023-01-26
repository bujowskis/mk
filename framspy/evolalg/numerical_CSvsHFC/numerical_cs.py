from pandas import DataFrame, Series
import numpy as np
import random
import copy
from evolalg.cs_base.experiment_convection_selection import ExperimentConvectionSelection
from evolalg.structures.population_methods import reinitialize_population_with_random_numerical

class ExperimentNumericalCSRun(ExperimentConvectionSelection):
    def __init__(
            self, popsize, hof_size, number_of_populations, migration_interval, save_only_best,
            benchmark_function, results_directory_path, dimensions
    ):
        super().__init__(popsize, hof_size, number_of_populations, migration_interval, save_only_best, dimensions)
        self.benchmark_function = benchmark_function
        self.results_directory_path = results_directory_path
        self.dimensions = dimensions

    def evolve(
            self, hof_savefile, generations, initialgenotype, pmut, pxov, tournament_size,
            try_from_saved_file: bool = True  # to enable in-code disabling of loading saved savefile
    ):
        self.setup_evolution(hof_savefile, initialgenotype, try_from_saved_file)
        function_bound = 100**self.dimensions
        self.reinitialize_population_with_random_numerical(self.population_structures, self.dimensions, upper_bound=function_bound, lower_bound=-function_bound)

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
            df.loc[len(df)] = [cli_stats[0], cli_stats[1], cli_stats[2][0], cli_stats[3][0]]

        df.to_csv(f'{self.results_directory_path}/numerical_cs-{self.benchmark_function.__name__}-{len(initialgenotype)}-{self.migration_interval}-{len(self.populations)}.csv')

        return self.hof, self.stats, df

    def cross_over(self, gen1, gen2):
        division_point = len(gen1) // 2
        output = list(gen1[:division_point]) + list(gen2[division_point:]) if random.getrandbits(1) == 0 else \
            list(gen2[:division_point]) + list(gen1[division_point:])
        return output

    def evaluate(self, genotype):
        cec2017_genotype = np.array([genotype])
        return self.benchmark_function(cec2017_genotype)

    def mutate(self, gen1):
        output = copy.deepcopy(gen1)
        output[np.random.randint(len(output))] += random.uniform(-5.0, 5.0)
        return output
