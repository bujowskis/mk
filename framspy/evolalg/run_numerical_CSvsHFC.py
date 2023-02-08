import itertools

import numpy as np
import random
from evolalg.numerical_CSvsHFC.numerical_cs_equiwidth import ExperimentNumericalCSEquiwidth
from evolalg.numerical_CSvsHFC.numerical_hfc import ExperimentNumericalHFC
from cec2017.functions import f1, f3, f4, f5, f6, f7, f8, f9, f10, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30


# FIXME - make script runnable from CLI
# FIXME - add special measures into # .csv
def main():
    NUMBER_OF_REPETITIONS = 30
    hof_size = 10
    evaluations = 100_000
    # f21, f22, f23, f24, f25, f26, f27, f28
    functions = [f30, f1, f3, f4, f5, f6, f7, f8, f9, f10]  # FIXME - f29 [nan, nan)
    dimension = 30
    parameters_default = {"migration_interval": 10, "populations": 25, "subpopsize": 50, "pmut": 0.8, "pxov": 0.2, "tournament_size": 5}
    parameters_optional = {"migration_interval": 2, "populations": 5, "subpopsize": 100, "pmut": 1.0, "pxov": 0.0, "tournament_size": 20}
    # [default, optional]
    # migration_interval_list = [10, 2]
    # number_of_populations_list = [25, 5]
    # subpopsize = [50, 100] 
    # pmuts = [0.8, 1.0]  
    # pxovs = [0.2, 0.0]  
    # tournament_size = [5, 20]
    mutation_sd_fraction = [.01, .1]  # fixme - mutation sd_fraction
    results_directory_path = 'results/numerical_CSvsHFC/'

    # seeds = [seed for seed in range(10)]

    # FIXME - CS, HFC - same parameters -> make it so that we have comparable results "as we go" (CS and HFC, 2 dim, same params instead of CS 2 dim, 10 dim, ...)
    # for combination in list(itertools.product(*[
    #     functions, iterations
    # ])):
    #     break  # fixme

    # for combination in list(itertools.product(*[
    #     functions, dimensions, seeds, migration_interval_list, number_of_populations_list
    # ])):
    #     benchmark_function, dimension, seed, migration_interval, number_of_populations = combination

    #     # random.seed(seed); np.random.seed(seed)
    #     experiment = ExperimentNumericalCSEquiwidth(
    #         popsize, hof_size, number_of_populations, migration_interval,
    #         save_only_best=True, benchmark_function=benchmark_function, results_directory_path=results_directory_path,
    #         dimensions=dimension
    #     )
    #     hof, stats, df = experiment.evolve(
    #         hof_savefile=f'HoF/numerical_CSvsHFC/cs-ew/HoF-cs-{benchmark_function.__name__}-{dimension}-{seed}-{migration_interval}-{number_of_populations}.gen',
    #         generations=generations,
    #         tournament_size=tournament_size,
    #         pmut=pmut,
    #         pxov=pxov,
    #         try_from_saved_file=False,
    #         initialgenotype=np.zeros(dimension)
    #     )
    #     df.to_csv(f'{results_directory_path}numerical_CSvsHFC-cs-{benchmark_function.__name__}-{dimension}-{seed}-{migration_interval}-{number_of_populations}.csv')

    #     # random.seed(seed); np.random.seed(seed)
    #     experiment = ExperimentNumericalHFC(
    #         popsize, hof_size, number_of_populations, migration_interval,
    #         save_only_best=True, benchmark_function=benchmark_function, results_directory_path=results_directory_path,
    #         dimensions=dimension
    #     )
    #     hof, stats, df = experiment.evolve(
    #         hof_savefile=f'HoF/numerical_CSvsHFC/hfc/HoF-hfc-{benchmark_function.__name__}-{dimension}-{seed}-{migration_interval}-{number_of_populations}.gen',
    #         generations=generations,
    #         tournament_size=tournament_size,
    #         pmut=pmut,
    #         pxov=pxov,
    #         try_from_saved_file=False,
    #         initialgenotype=np.zeros(dimension)
    #     )
    #     df.to_csv(f'{results_directory_path}numerical_CSvsHFC-hfc-{benchmark_function.__name__}-{dimension}-{seed}-{migration_interval}-{number_of_populations}.csv')
    #     break  # fixme - remove for experiments


    for repetition in range(NUMBER_OF_REPETITIONS):
        for fraction in mutation_sd_fraction:
            for benchmark_function in functions:
                migration_interval, number_of_populations, subpopsize, pmut, pxov, tournament_size = parameters_default.values()
                #generations = int(np.ceil(evaluations / (number_of_populations * subpopsize)))
                #print(f'generations: {generations}')
                generations = 500

                experiment = ExperimentNumericalCSEquiwidth(
                    subpopsize, hof_size, number_of_populations, migration_interval,
                    save_only_best=True, benchmark_function=benchmark_function, results_directory_path=results_directory_path,
                    dimensions=dimension
                )
                hof, stats, df = experiment.evolve(
                    hof_savefile=f'HoF/numerical_CSvsHFC/cs-ew/HoF-cs-{benchmark_function.__name__}-{dimension}-{repetition}-{fraction}-{migration_interval}-{number_of_populations}-{subpopsize}-{pmut}-{pxov}-{tournament_size}.gen',
                    generations=generations,
                    tournament_size=tournament_size,
                    pmut=pmut,
                    pxov=pxov,
                    try_from_saved_file=False,
                    initialgenotype=np.zeros(dimension)
                )
                df.to_csv(f'{results_directory_path}numerical_CSvsHFC-cs-{benchmark_function.__name__}-{dimension}-{repetition}-{fraction}-{migration_interval}-{number_of_populations}-{subpopsize}-{pmut}-{pxov}-{tournament_size}.csv')

                experiment = ExperimentNumericalHFC(
                    subpopsize, hof_size, number_of_populations, migration_interval,
                    save_only_best=True, benchmark_function=benchmark_function, results_directory_path=results_directory_path,
                    dimensions=dimension
                )
                hof, stats, df = experiment.evolve(
                    hof_savefile=f'HoF/numerical_CSvsHFC/hfc/HoF-hfc-{benchmark_function.__name__}-{dimension}-{repetition}-{fraction}-{migration_interval}-{number_of_populations}-{subpopsize}-{pmut}-{pxov}-{tournament_size}.gen',
                    generations=generations,
                    tournament_size=tournament_size,
                    pmut=pmut,
                    pxov=pxov,
                    try_from_saved_file=False,
                    initialgenotype=np.zeros(dimension)
                )
                df.to_csv(f'{results_directory_path}numerical_CSvsHFC-hfc-{benchmark_function.__name__}-{dimension}-{repetition}-{fraction}-{migration_interval}-{number_of_populations}-{subpopsize}-{pmut}-{pxov}-{tournament_size}.csv')


if __name__ == "__main__":
    main()
