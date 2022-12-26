import numpy as np
import benchmark_functions as bf

from .numerical_example.numerical_complex import ExperimentNumericalComplex

AVAILABLE_FUNCTIONS = [
    'Ackley',
    'EggHolder',
    'Schaffer2',  # note - only 2 dim
]


# note - there may be a need to restrict the boundaries
def main():
    args_parser = ExperimentNumericalComplex.get_args_for_parser()
    args_parser.add_argument('-dimensions', type=int, default=2)
    args_parser.add_argument('-benchmark_function', type=str, default='null')
    parsed_args = args_parser.parse_args()
    # print("Argument values:", ", ".join(['%s=%s' % (arg, getattr(parsed_args, arg)) for arg in vars(parsed_args)]))

    initialgenotype = np.random.uniform(low=0.5, high=13.3, size=(parsed_args.dimensions,))
    if parsed_args.benchmark_function not in AVAILABLE_FUNCTIONS:
        raise Exception(f'wrong function - choose one from the following: {AVAILABLE_FUNCTIONS}')
    benchmark_function = \
        bf.Ackley(n_dimensions=parsed_args.dimensions, opposite=True) if parsed_args.benchmark_function == 'Ackley' else \
        bf.EggHolder(n_dimensions=parsed_args.dimensions, opposite=True) if parsed_args.benchmark_function == 'EggHolder' else \
        bf.Schaffer2(opposite=True)

    print('Best individuals:')
    experiment = ExperimentNumericalComplex(
        hof_size=parsed_args.hof_size,
        popsize=parsed_args.popsize,
        save_only_best=parsed_args.save_only_best,
        benchmark_function=benchmark_function
    )

    hof, stats = experiment.evolve(
        hof_savefile=parsed_args.hof_savefile,
        generations=parsed_args.generations,
        initialgenotype=initialgenotype,
        pmut=parsed_args.pmut,
        pxov=parsed_args.pxov,
        tournament_size=parsed_args.tournament
    )

    # for ind in hof: print(ind.genotype, ind.rawfitness)


if __name__ == "__main__":
    main()
