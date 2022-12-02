import argparse
import sys

from .utils import ensureDir
from .numerical_example.numerical_island_example import ExperimentNumericalIsland

    

def parseArguments():
    parser = argparse.ArgumentParser(description='Run this program with "python -u %s" if you want to disable buffering of its output.' % sys.argv[0])
    # parser.add_argument('-initialgenotype', type=str, required=False, help='The genotype used to seed the initial population. If given, the -genformat argument is ignored.')

    parser.add_argument('-popsize', type=int, default=50, help="Population size, default: 50.")
    parser.add_argument('-generations', type=int, default=5, help="Number of generations, default: 5.")
    parser.add_argument('-tournament', type=int, default=5, help="Tournament size, default: 5.")
    parser.add_argument('-pmut', type=float, default=0.7, help="Probability of mutation, default: 0.7")
    parser.add_argument('-pxov', type=float, default=0.2, help="Probability of crossover, default: 0.2")
    parser.add_argument('-hof_size', type=int, default=10, help="Number of genotypes in Hall of Fame. Default: 10.")
    parser.add_argument('-hof_savefile', required=False, help='If set, Hall of Fame will be saved in Framsticks file format (recommended extension *.gen). This also activates saving state (checpoint) file and auto-resuming from the saved state, if this file exists.')
  
    parser.add_argument("-archive",type=int,default=0,help="Maximum archive size")
    parser.add_argument("-islands",type=int,default=5,help="Number of subpopulations (islands)")
    parser.add_argument("-generations_migration",type=int,default=10,help="Number of generations separating migration events when genotypes migrate between subpopulations (islands)")

    return parser.parse_args()


def main():
    parsed_args = parseArguments()
    print("Argument values:", ", ".join(['%s=%s' % (arg, getattr(parsed_args, arg)) for arg in vars(parsed_args)]))

    initialgenotype = [100,100,100,100]
    print('Best individuals:')
    experiment = ExperimentNumericalIsland(
                           hof_size=parsed_args.hof_size,
                           popsize=parsed_args.popsize,
                           migration_interval=parsed_args.generations_migration,
                           number_of_populations=parsed_args.islands,
                           archive_size=parsed_args.archive)
                           
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
