import random

import numpy as np

from ..base.remove_diagonal import remove_diagonal


class PopulationStructures:
    def __init__(self,  initial_individual, archive_size=0, popsize=100) -> None:
        self.population_size = popsize
        self.population = [initial_individual.copy()
                           for _ in range(self.population_size)]
        self.archive = []
        self.archive_size = archive_size

    def change_individual(self, dissim):
        no_diagonal = remove_diagonal(dissim.copy())
        minval = np.min(no_diagonal)
        to_remove_pair = list(set(np.where(no_diagonal == minval)[0]))
        last_index = len(dissim)-1
        if last_index in to_remove_pair:
            return last_index
        else:
            return random.choice(to_remove_pair)

    def update_archive(self, dissim_matrix, population_archive):
        if self.archive_size < 1:
            return
        current_archive_ind = [i for i in range(
            self.population_size, len(dissim_matrix))]
        for i in range(self.population_size):
            current_archive_ind = current_archive_ind + [i]
            temp_dissim_matrix = dissim_matrix[current_archive_ind,
                                               ][:, current_archive_ind]
            if len(current_archive_ind) > self.archive_size:
                to_remove = self.change_individual(temp_dissim_matrix)
                current_archive_ind.pop(to_remove)

        self.archive = np.array(population_archive)[
            current_archive_ind].tolist()
