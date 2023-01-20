from pandas import DataFrame
import numpy as np
import random
from evolalg.cs_base.experiment_convection_selection import ExperimentConvectionSelection


class ExperimentCSRun(ExperimentConvectionSelection):
    def __init__(
            self, popsize, hof_size, number_of_populations, migration_interval, save_only_best,
            benchmark_function
    ):
        super().__init__(popsize, hof_size, number_of_populations, migration_interval, save_only_best)
        self.benchmark_function = benchmark_function

    def evolve(
            self, hof_savefile, generations, initialgenotype, pmut, pxov, tournament_size,
            try_from_saved_file: bool = True  # to enable in-code disabling of loading saved savefile
    ):
        self.setup_evolution(hof_savefile, initialgenotype, try_from_saved_file)
        df = DataFrame()

        for g in range(self.current_generation, generations):
            for p in self.populations:
                p.population = self.make_new_population(p.population, pmut, pxov, tournament_size)

            if g % self.migration_interval == 0:
                self.migrate_populations()

            pool_of_all_individuals = []
            [pool_of_all_individuals.extend(p.population) for p in self.populations]
            self.update_stats(g, pool_of_all_individuals)
            df.append(self.stats)

        df.to_csv(f'../../results/cs_equinumber-{self.benchmark_function.__name__}-{len(initialgenotype)}.csv')

        return self.hof, self.stats

    def cross_over(self, gen1, gen2):
        division_point = len(gen1) // 2
        list(gen1[:division_point]) + list(gen2[division_point:]) if random.getrandbits(1) == 0 else \
            list(gen2[:division_point]) + list(gen1[division_point:])

    def evaluate(self, genotype):
        return self.benchmark_function(genotype)

    def mutate(self, gen1):
        output = gen1[:]
        output[np.random.randint(0, len(output))] += random.uniform(-5.0, 5.0)
        assert len(output) == len(gen1)
        return output
