from pandas import DataFrame
from evolalg.cs_base.experiment_convection_selection import ExperimentConvectionSelection


class ExperimentCSRun(ExperimentConvectionSelection):
    def __init__(
            self, popsize, hof_size, number_of_populations, migration_interval, save_only_best,
            benchmark_function, results_directory_path
    ):
        super().__init__(popsize, hof_size, number_of_populations, migration_interval, save_only_best)
        self.benchmark_function = benchmark_function
        self.results_directory_path = results_directory_path

    def evolve(
            self, hof_savefile, generations, initialgenotype, pmut, pxov, tournament_size,
            try_from_saved_file: bool = True  # to enable in-code disabling of loading saved savefile
    ):
        self.setup_evolution(hof_savefile, initialgenotype, try_from_saved_file)
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

        df.to_csv(f'{self.results_directory_path}/cs_equinumber-{self.benchmark_function.__name__}-{len(initialgenotype)}-{self.migration_interval}-{len(self.populations)}.csv')

        return self.hof, self.stats
