from .numerical_example.numerical_example import ExperimentNumerical


def main():
    parsed_args = ExperimentNumerical.get_args_for_parser().parse_args()
    print("Argument values:", ", ".join(
        ['%s=%s' % (arg, getattr(parsed_args, arg)) for arg in vars(parsed_args)]))

    initialgenotype = [100, 100, 100, 100]
    print('Best individuals:')
    experiment = ExperimentNumerical(
        hof_size=parsed_args.hof_size,
        popsize=parsed_args.popsize,
        save_only_best=parsed_args.save_only_best)

    hof, stats = experiment.evolve(hof_savefile=parsed_args.hof_savefile,
                                   generations=parsed_args.generations,
                                   initialgenotype=initialgenotype,
                                   pmut=parsed_args.pmut,
                                   pxov=parsed_args.pxov,
                                   tournament_size=parsed_args.tournament)
    print(len(hof))
    for ind in hof:
        print(ind.genotype, ind.rawfitness)


if __name__ == "__main__":
    main()
