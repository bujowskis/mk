import numpy as np
from intervals import closed, openclosed
from abc import ABC

from evolalg.cs_base.experiment_convection_selection import ExperimentConvectionSelection
from evolalg.structures.population_methods import remove_excess_individuals_random

# how to define the absolute range bounds? (min and max from pool of all individuals?)
#   YES (current min, current max - not historical)

# how to handle case with all uniform individuals?
#   either check for this case and invoke equinumber
#   or handled with proper implementation

# how to handle case with marginal difference in min and max?
#   (handled in the below)

# how to handle case with no individuals in subpopulation?
#   (notes: "receives from nearest low non-empty" - how exactly?)
#   look for the bottom explanations (may use as a reference code KacperPerz/evolalg - after our implementation)


class ExperimentConvectionSelectionEquiwidth(ExperimentConvectionSelection, ABC):
    def migrate_populations(self):
        """
        Equiwidth migration - populations with distribution of fixed ranges of fitness
        """
        # 1.0, 1.2, 1.5, 2.0, 4.0
        # [1.0, 2.0], (2.0, 3.0]
        # ^ if this is inclusive, the copying mechanism handles some edge cases todo - test if that's the case
        # OR invoke equinumber in edge case max == min
        # todo - check for case with 1 pop
        # fixme - use the ub as next lb
        # todo - check if works for empty subpopulation case

        # get the needed information about the populations
        pool_of_all_individuals = []
        for p in self.populations:
            pool_of_all_individuals.extend(p.population)
        lower_bound = min(pool_of_all_individuals, key=lambda x: x.fitness).fitness
        upper_bound = max(pool_of_all_individuals, key=lambda x: x.fitness).fitness
        population_width = (upper_bound - lower_bound) / self.number_of_populations

        # create subpopulations' fitness ranges
        population_cuts = [lower_bound + x*population_width for x in range(self.number_of_populations)]
        population_cuts[-1] = upper_bound  # ensures no python float errors
        subpopulations_fitness_ranges = {
            openclosed(population_cuts[x], population_cuts[x+1]): [] for x in range(len(population_cuts)-1)  # FIXME - 1 element shorter
        }
        subpopulations_fitness_ranges[closed(population_cuts[0], population_cuts[1])] = subpopulations_fitness_ranges.pop(openclosed(population_cuts[0], population_cuts[1]))  # includes the lower_bound individual's fitness

        # Code below can be optimized this way:                  
        # having, e.g., ranges_subpopulations = {[0.1, 0.2]: 0, ..., [0.7, 0.8]: <number_of_populations>]}

        # individual_groups = dict()
        # for i in pool_of_all_individuals:
            ## individual_groups[i] = np.where(i.fitness in ranges_subpopulations.keys(), ranges_subpopulatons.values())

        # then we'd have, e.g., individual_groups = {<individual>: 0, ...}, meaning the <individual> is assigned to range group 0

        # place individuals in the respective subpopulations
        for i in pool_of_all_individuals:  # fixme (future) - needs optimization, O(n^2) for now
            for fitness_range in subpopulations_fitness_ranges.keys():
                if i.fitness in fitness_range:
                    subpopulations_fitness_ranges[fitness_range].append(i)
                    continue
        
        # ensure all subpopulations are non-empty
        sorted_sub_pop_ranges = sorted(subpopulations_fitness_ranges.keys())
        for i in range(1, len(sorted_sub_pop_ranges)):  # note that the first
            if not subpopulations_fitness_ranges[sorted_sub_pop_ranges[i]]:
                subpopulations_fitness_ranges[sorted_sub_pop_ranges[i]] = subpopulations_fitness_ranges[sorted_sub_pop_ranges[i-1]][:]

        # remove the excess individuals
        # NOTE - removing before ensuring non-empty subpopulations would be more efficient
        #   however, we potentially lose diversity doing this (since random removal may preserve different individuals
        #   from population p[n-1] in p[n]
        for sub_pop_range in sorted_sub_pop_ranges:
            subpopulations_fitness_ranges[sub_pop_range] = remove_excess_individuals_random(
                individuals=subpopulations_fitness_ranges[sub_pop_range],
                population_size=self.populations[0].population_size
            )

        # fill populations with not enough individuals
        # NOTE - seems to be handled by experiment_abc.make_new_population todo - make sure

        # assign migrated subpopulations  # FIXME - not the same
        print(len(self.populations))
        print(len(subpopulations_fitness_ranges))
        for i in range(len(self.populations)):
            self.populations[i] = subpopulations_fitness_ranges[sorted_sub_pop_ranges[i]]
