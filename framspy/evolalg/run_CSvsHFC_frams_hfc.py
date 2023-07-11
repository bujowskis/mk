from FramsticksLib import FramsticksLib

from .frams_base.experiment_frams_islands import ExperimentFramsIslands
from .frams_CSvsHFC.frams_hfc import ExperimentFramsHFC


def main():
    FramsticksLib.DETERMINISTIC = False
    parsed_args = ExperimentFramsIslands.get_args_for_parser().parse_args()
    print("Argument values:", ", ".join(
        ['%s=%s' % (arg, getattr(parsed_args, arg)) for arg in vars(parsed_args)]))

    opt_criteria = parsed_args.opt.split(",")
    sim_file = "eval-allcriteria-mini.sim;deterministic.sim;sample-period-2.sim;only-body.sim"
    framsLib = FramsticksLib(parsed_args.path, parsed_args.lib, sim_file.split(";"))
    constrains = {"max_numparts": parsed_args.max_numparts,
                  "max_numjoints": parsed_args.max_numjoints,
                  "max_numneurons": parsed_args.max_numneurons,
                  "max_numconnections": parsed_args.max_numconnections,
                  "max_numgenochars": parsed_args.max_numgenochars,
                  }

    repetition, tournament_size, generations = parsed_args.runnum, parsed_args.tsize, parsed_args.generations
    hof_size = 10

    parameters_default = {"migration_interval": 10, "populations": 25, "subpopsize": 50, "pmut": 0.8, "pxov": 0.2}
    # parameters_optional = {"migration_interval": 2, "populations": 5, "subpopsize": 100, "pmut": 1.0, "pxov": 0.0, "tournament_size": 20}
    results_directory_path = 'results/frams/'

    migration_interval, number_of_populations, subpopsize, pmut, pxov = parameters_default.values()

    experiment = ExperimentFramsHFC(frams_lib=framsLib,
                                        optimization_criteria=opt_criteria,
                                        genformat=parsed_args.genformat,
                                        hof_size=hof_size,
                                        constraints=constrains,
                                        popsize=subpopsize,
                                        migration_interval=migration_interval,
                                        number_of_populations=number_of_populations,
                                        save_only_best=parsed_args.save_only_best,
                                        results_directory_path=results_directory_path)

    hof, stats, df = experiment.evolve(hof_savefile=f'HoF/frams/hfc/frams_HoF_hfc-{sim_file}-{repetition}-{migration_interval}-{number_of_populations}-{subpopsize}-{pmut}-{pxov}-{tournament_size}.gen',
                    generations=generations,
                    initialgenotype=parsed_args.initialgenotype,
                    pmut=pmut,
                    pxov=pxov,
                    genformat=parsed_args.genformat,
                    tournament_size=tournament_size)
    df.to_csv(f'results/frams/hfc/frams_CSvsHFC_hfc-{sim_file}-{repetition}-{migration_interval}-{number_of_populations}-{subpopsize}-{pmut}-{pxov}-{tournament_size}.csv')
    
if __name__ == "__main__":
    main()
