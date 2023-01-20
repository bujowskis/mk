import numpy as np
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
                    experiment = ExperimentCSRun(popsize, hof_size, number_of_populations, migration_interval, save_only_best=True, benchmark_function=fun)
                    experiment.evolve(
                        hof_savefile=None,
                        generations=generations,
                        tournament_size=tournament_size,
                        pmut=pmut,
                        pxov=pxov,
                        try_from_saved_file=False,
                        # FIXME - CEC benchmarks "breaking news" change breaks the code...
                        initialgenotype=np.array(np.zeros(dimension))  # np.random.uniform(0,0,size=(3, dimension))
                    )


if __name__ == "__main__":
    main()
