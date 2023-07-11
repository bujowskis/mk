import numpy as np
from evolalg.numerical_CSvsHFC.numerical_cs_equiwidth import ExperimentNumericalCSEquiwidth
from evolalg.numerical_CSvsHFC.numerical_hfc import ExperimentNumericalHFC
from .base.experiment_islands_model_abc import ExperimentIslands
from cec2017.functions import f1, f3, f4, f5, f6, f7, f8, f9, f10, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30


def main():
    parsed_args = ExperimentIslands.get_args_for_parser().parse_args()
    repetition, fraction, function_str, dimension, tournament_size = parsed_args.runnum, parsed_args.mutsize, parsed_args.funcnum, parsed_args.dims, parsed_args.tsize
    hof_size = 10
    evaluations = 100_000
    functions = [f21, f22, f23, f24, f25, f26, f27, f28, f29, f30, f1, f3, f4, f5, f6, f7, f8, f9, f10]

    for function in functions:
        if function.__name__ == function_str:
            benchmark_function = function
    parameters_default = {"migration_interval": 10, "populations": 25, "subpopsize": 50, "pmut": 0.8, "pxov": 0.2}
    # parameters_optional = {"migration_interval": 2, "populations": 5, "subpopsize": 100, "pmut": 1.0, "pxov": 0.0, "tournament_size": 20}
    results_directory_path = 'results/numerical_CSvsHFC/'

    migration_interval, number_of_populations, subpopsize, pmut, pxov = parameters_default.values()
    generations = 20

    experiment = ExperimentNumericalHFC(
        subpopsize, hof_size, number_of_populations, migration_interval,
        save_only_best=True, benchmark_function=benchmark_function, results_directory_path=results_directory_path,
        dimensions=dimension, mutsize=fraction
    )
    hof, stats, df = experiment.evolve(
        hof_savefile=f'HoF/numerical_CSvsHFC/hfc/HoF-hfc-{function_str}-{dimension}-{repetition}-{fraction}-{migration_interval}-{number_of_populations}-{subpopsize}-{pmut}-{pxov}-{tournament_size}.gen',
        generations=generations,
        tournament_size=tournament_size,
        pmut=pmut,
        pxov=pxov,
        try_from_saved_file=False,
        initialgenotype=np.zeros(dimension)
    )
    df.to_csv(f'results/numerical_CSvsHFC/hfc/numerical_CSvsHFC-hfc-{function_str}-{dimension}-{repetition}-{fraction}-{migration_interval}-{number_of_populations}-{subpopsize}-{pmut}-{pxov}-{tournament_size}.csv')


if __name__ == "__main__":
    main()
