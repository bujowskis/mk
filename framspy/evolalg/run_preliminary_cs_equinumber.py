from evolalg.preliminary_experiments.cs_equinumber import ExperimentCSRun
from cec2017.functions import f1, f4, f6, f9, f10


def main():
    # fixed parameters
    hof_size = 10
    popsize = 100
    pmut = 0.7
    pxov = 0.2
    generations = 100_000
    tournament_size = 5

    # configurations we are interested in
    migration_interval_list = [10]
    number_of_populations_list = [5]

    dimensions = [2, 10, 20, 30, 50, 100]
    functions = [f1, f4, f6, f9, f10]

    for fun in functions:
        for dimension in dimensions:
            for migration_interval in migration_interval_list:
                for number_of_populations in number_of_populations_list:
                    # experiment setup

                    with open(file=f'', mode='w') as f:
                    initialgenotype = initialgenotype,

    initialgenotype = initialgenotype,


if __name__ == "__main__":
    main()
