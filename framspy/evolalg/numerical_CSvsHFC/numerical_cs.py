from pandas import DataFrame
import time
from evolalg.cs_base.experiment_convection_selection import ExperimentConvectionSelection
from evolalg.structures.population_methods import reinitialize_population_with_random_numerical
from evolalg.mutation import cec2017_numerical_mutation
from evolalg.crossover import cec2017_numerical_crossover
from evolalg.utils import evaluate_cec2017, get_state_filename


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
        time0 = time.process_time()

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
            self.update_stats(g, pool_of_all_individuals)
            if hof_savefile is not None:
                self.current_generation = g
                self.time_elapsed += time.process_time() - time0
                self.save_state(get_state_filename(hof_savefile))
        if hof_savefile is not None:
            self.save_genotypes(hof_savefile)

        return self.hof, self.stats, df

    def cross_over(self, gen1, gen2):
        return cec2017_numerical_crossover(gen1, gen2)

    def evaluate(self, genotype):
        return evaluate_cec2017(genotype, self.benchmark_function)

    def mutate(self, gen1):
        return cec2017_numerical_mutation(gen1)
