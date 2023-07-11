from FramsticksLib import FramsticksLib

from .frams_base.experiment_frams_islands import ExperimentFramsIslands


def main():
    # random.seed(123)  # see FramsticksLib.DETERMINISTIC below, set to True if you want full determinism
    # must be set before FramsticksLib() constructor call
    FramsticksLib.DETERMINISTIC = False
    parsed_args =ExperimentFramsIslands.get_args_for_parser().parse_args()
    print("Argument values:", ", ".join(
        ['%s=%s' % (arg, getattr(parsed_args, arg)) for arg in vars(parsed_args)]))
    # multiple criteria not supported here. If needed, use FramsticksEvolution.py
    opt_criteria = parsed_args.opt.split(",")
    framsLib = FramsticksLib(
        parsed_args.path, parsed_args.lib, parsed_args.sim.split(";"))
    constrains = {"max_numparts": parsed_args.max_numparts,
                  "max_numjoints": parsed_args.max_numjoints,
                  "max_numneurons": parsed_args.max_numneurons,
                  "max_numconnections": parsed_args.max_numconnections,
                  "max_numgenochars": parsed_args.max_numgenochars,
                  }

    print('Best individuals:')
    experiment = ExperimentFramsIslands(frams_lib=framsLib,
                                        optimization_criteria=opt_criteria,
                                        hof_size=parsed_args.hof_size,
                                        constraints=constrains,
                                        genformat=parsed_args.genformat,
                                        popsize=parsed_args.popsize,
                                        migration_interval=parsed_args.generations_migration,
                                        number_of_populations=parsed_args.islands,
                                        save_only_best=parsed_args.save_only_best)

    experiment.evolve(hof_savefile=parsed_args.hof_savefile,
                      generations=parsed_args.generations,
                      initialgenotype=parsed_args.initialgenotype,
                      pmut=parsed_args.pmut,
                      pxov=parsed_args.pxov,
                      tournament_size=parsed_args.tournament)


if __name__ == "__main__":
    main()
