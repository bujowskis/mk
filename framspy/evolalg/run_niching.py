import argparse
import sys

# FIXME - relative import outside package
from FramsticksLib import FramsticksLib
from .utils import ensureDir
from .frams_base.experiment_frams_niching import ExperimentFramsNiching


def parseArguments():
    parser = argparse.ArgumentParser(
        description='Run this program with "python -u %s" if you want to disable buffering of its output.' % sys.argv[
            0])
    parser.add_argument('-path', type=ensureDir, required=True, help='Path to Framsticks CLI without trailing slash.')
    parser.add_argument('-lib', required=False,
                        help='Library name. If not given, "frams-objects.dll" or "frams-objects.so" is assumed depending on the platform.')
    parser.add_argument('-sim', required=False, default="eval-allcriteria.sim",
                        help="The name of the .sim file with settings for evaluation, mutation, crossover, and similarity estimation. If not given, \"eval-allcriteria.sim\" is assumed by default. Must be compatible with the \"standard-eval\" expdef. If you want to provide more files, separate them with a semicolon ';'.")

    parser.add_argument('-genformat', required=False,
                        help='Genetic format for the simplest initial genotype, for example 4, 9, or B. If not given, f1 is assumed.')
    parser.add_argument('-initialgenotype', required=False,
                        help='The genotype used to seed the initial population. If given, the -genformat argument is ignored.')

    parser.add_argument('-opt', required=True,
                        help='optimization criteria: vertpos, velocity, distance, vertvel, lifespan, numjoints, numparts, numneurons, numconnections (or other as long as it is provided by the .sim file and its .expdef).')  # For multiple criteria optimization, separate the names by the comma.')
    parser.add_argument('-popsize', type=int, default=50, help="Population size, default: 50.")
    parser.add_argument('-generations', type=int, default=5, help="Number of generations, default: 5.")
    parser.add_argument('-tournament', type=int, default=5, help="Tournament size, default: 5.")
    parser.add_argument('-pmut', type=float, default=0.7, help="Probability of mutation, default: 0.7")
    parser.add_argument('-pxov', type=float, default=0.2, help="Probability of crossover, default: 0.2")
    parser.add_argument('-hof_size', type=int, default=10, help="Number of genotypes in Hall of Fame. Default: 10.")
    parser.add_argument('-hof_savefile', required=False,
                        help='If set, Hall of Fame will be saved in Framsticks file format (recommended extension *.gen). This also activates saving state (checpoint) file and auto-resuming from the saved state, if this file exists.')

    parser.add_argument('-max_numparts', type=int, default=None, help="Maximum number of Parts. Default: no limit")
    parser.add_argument('-max_numjoints', type=int, default=None, help="Maximum number of Joints. Default: no limit")
    parser.add_argument('-max_numneurons', type=int, default=None, help="Maximum number of Neurons. Default: no limit")
    parser.add_argument('-max_numconnections', type=int, default=None,
                        help="Maximum number of Neural connections. Default: no limit")
    parser.add_argument('-max_numgenochars', type=int, default=None,
                        help="Maximum number of characters in genotype (including the format prefix, if any). Default: no limit")

    parser.add_argument("-dissim", type=int, default="frams",
                        help="Dissimilarity measure type. Availible -2:emd, -1:lev, 1:frams1 (default), 2:frams2")
    parser.add_argument("-fit", type=str, default="raw",
                        help="Fitness type, availible  types: niching, novelty, nsga2, nslc and raw (default)")
    parser.add_argument("-archive", type=int, default=50, help="Maximum archive size")
    parser.add_argument("-normalize", type=str, default="max",
                        help="What normalization use for dissimilarity matrix, max (default), sum and none")
    # parser.add_argument("-knn",type=int,default=0,help="Nearest neighbours parameter for local novelty/niching, if knn==0 global is performed.Default:0")

    return parser.parse_args()


def main():
    # random.seed(123)  # see FramsticksLib.DETERMINISTIC below, set to True if you want full determinism
    FramsticksLib.DETERMINISTIC = False  # must be set before FramsticksLib() constructor call
    parsed_args = parseArguments()
    print("Argument values:", ", ".join(['%s=%s' % (arg, getattr(parsed_args, arg)) for arg in vars(parsed_args)]))
    opt_criteria = parsed_args.opt.split(
        ",")  # multiple criteria not supported here. If needed, use FramsticksEvolution.py
    framsLib = FramsticksLib(parsed_args.path, parsed_args.lib, parsed_args.sim.split(";"))
    constrains = {
        "max_numparts": parsed_args.max_numparts,
        "max_numjoints": parsed_args.max_numjoints,
        "max_numneurons": parsed_args.max_numneurons,
        "max_numconnections": parsed_args.max_numconnections,
        "max_numgenochars": parsed_args.max_numgenochars,
    }

    print('Best individuals:')
    experiment = ExperimentFramsNiching(
        frams_lib=framsLib,
        optimization_criteria=opt_criteria,
        hof_size=parsed_args.hof_size,
        constraints=constrains,
        normalize=parsed_args.normalize,
        dissim=parsed_args.dissim,
        fit=parsed_args.fit,
        genformat=parsed_args.genformat,
        popsize=parsed_args.popsize,
        archive_size=parsed_args.archive
    )

    experiment.evolve(
        hof_savefile=parsed_args.hof_savefile,
        generations=parsed_args.generations,
        initialgenotype=parsed_args.initialgenotype,
        pmut=parsed_args.pmut,
        pxov=parsed_args.pxov,
        tournament_size=parsed_args.tournament
    )


if __name__ == "__main__":
    main()
