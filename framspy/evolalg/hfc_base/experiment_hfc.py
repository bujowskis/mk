import time
from abc import ABC
from typing import List

from evolalg.structures.individual import Individual
from evolalg.structures.population import PopulationStructures
from evolalg.cs_base.experiment_convection_selection import ExperimentConvectionSelection

from evolalg.utils import get_state_filename


class ExperimentHFC(ExperimentConvectionSelection, ABC):
    """
    Implementation of HFC, tailored to be comparable with CS
    """
    # parameters
    number_of_populations: int  # = 5
    popsize: int  # = 100
    migration_interval: int  # = 10
    # internal members
    populations: List[PopulationStructures] = []  # = []

    def __init__(self, popsize, hof_size, number_of_populations, migration_interval, save_only_best) -> None:
        # todo - input validation
        super().__init__(
            popsize=popsize,
            hof_size=hof_size,
            save_only_best=save_only_best
        )
        self.number_of_populations = number_of_populations
        self.migration_interval = migration_interval

    def migrate_populations(self):
        """
        HFC "only up" admission threshold migration
        """

