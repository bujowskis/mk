from evolalg.cs_base.experiment_convection_selection import ExperimentConvectionSelection


class ExperimentCSRun(ExperimentConvectionSelection):
    def __init__(
            self, popsize, hof_size, number_of_populations, migration_interval, save_only_best,
            dimensions
    ):
        super().__init__(popsize, hof_size, number_of_populations, migration_interval, save_only_best)

    def cross_over(self, gen1, gen2):
        pass

    def evaluate(self, genotype):
        pass

    def mutate(self, gen1):
        pass
