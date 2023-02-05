import itertools

import numpy as np
import random
from evolalg.numerical_CSvsHFC.numerical_cs import ExperimentNumericalCSRun
from evolalg.numerical_CSvsHFC.numerical_hfc import ExperimentNumericalHFC
from cec2017.functions import f1, f4, f6, f9, f10


def main():
    # fixed parameters
    hof_size = 10
    popsize = 100  # todo - probably interested in the configuration
    pmut = 0.8
    pxov = 0.2
    generations = 1_000  #100_000 FIXME - turn it back for experiment runs
    tournament_size = 5
    results_directory_path = 'results/numerical_CSvsHFC/'

    # configurations we are interested in
    migration_interval_list = [10]
    number_of_populations_list = [5]
    # todo - popsize
    # todo - tournament
    pmuts = [1.0, 0.8]  # todo
    pxovs = [0.0, 0.2]  # todo
    # todo - mutations for numerical benchmarks
    # todo - crossovers for numerical benchmarks

    dimensions = [2, 10, 20, 30, 50, 100]
    functions = [f1, f4, f6, f9, f10]

    # Seeds for multiple runs with reproducible results
    # FIXME - incorporate seeds
    seeds = [seed for seed in range(10)]

    # FIXME - CS, HFC - same parameters -> make it so that we have comparable results "as we go" (CS and HFC, 2 dim, same params instead of CS 2 dim, 10 dim, ...)
    for combination in list(itertools.product(*[
        functions, dimensions, seeds, migration_interval_list, number_of_populations_list
    ])):
        benchmark_function, dimension, seed, migration_interval, number_of_populations = combination

        random.seed(seed); np.random.seed(seed)
        # experiment = ExperimentNumericalCSRun(  # FIXme - equiwidth, not equinumber
        #     popsize, hof_size, number_of_populations, migration_interval,
        #     save_only_best=True, benchmark_function=benchmark_function, results_directory_path=results_directory_path,
        #     dimensions=dimension
        # )
        # hof, stats, df = experiment.evolve(
        #     hof_savefile=None,
        #     generations=generations,
        #     tournament_size=tournament_size,
        #     pmut=pmut,
        #     pxov=pxov,
        #     try_from_saved_file=False,
        #     initialgenotype=np.zeros(dimension)
        # )
        # df.to_csv(f'{results_directory_path}numerical_CSvsHFC-cs-{benchmark_function.__name__}-{dimension}-{seed}-{migration_interval}-{number_of_populations}.csv')

        random.seed(seed); np.random.seed(seed)
        experiment = ExperimentNumericalHFC(
            popsize, hof_size, number_of_populations, migration_interval,
            save_only_best=True, benchmark_function=benchmark_function, results_directory_path=results_directory_path,
            dimensions=dimension
        )
        hof, stats, df = experiment.evolve(
            hof_savefile=None,
            generations=generations,
            tournament_size=tournament_size,
            pmut=pmut,
            pxov=pxov,
            try_from_saved_file=False,
            initialgenotype=np.zeros(dimension)
        )
        df.to_csv(
            f'{results_directory_path}numerical_CSvsHFC-hfc-{benchmark_function.__name__}-{dimension}-{seed}-{migration_interval}-{number_of_populations}.csv')
        break  # fixme

if __name__ == "__main__":
    main()