from experiment_convection_selection import ExperimentConvectionSelection


# todo - how to define the absolute range bounds? (min and max from pool of all individuals?)
# todo - how to handle case with all uniform individuals?
# todo - how to handle case with marginal difference in min and max?
# todo - how to handle case with no individuals in subpopulation?
#   (notes: "receives from nearest low non-empty" - how exactly?)

# todo - check if works for empty subpopulation case

class ExperimentConvectionSelectionEquiwidth(ExperimentConvectionSelection):
    def mutate(self, gen1):
        # todo - use modular
        pass

    def cross_over(self, gen1, gen2):
        # todo - use modular
        pass

    def evaluate(self, genotype):
        # todo - use modular
        pass

    def migrate_populations(self):
        """
        Equiwidth migration - populations with distribution of fixed ranges of fitness
        """
        pool_of_all_individuals = []
        for p in self.populations:
            pool_of_all_individuals.extend(p.population)
        sorted_individuals = sorted(pool_of_all_individuals, key=lambda x: x.rawfitness)  # todo - this or max() and min() (may be more efficient to use sorted) - + dynamic programming (slicing)

        lower_bound = sorted_individuals[0].fitness
        upper_bound = sorted_individuals[len(sorted_individuals)-1].fitness
        population_width = (upper_bound - lower_bound) / self.number_of_populations
        # population_ranges = [(lower_bound + x*population_width, lower_bound + (x+1)*population_width)
        #                      for x in range(self.number_of_populations)]
        cutting_points_values = [lower_bound + x*population_width for x in range(1, self.number_of_populations)]  # todo - check for case with 1 pop
        cutting_points_values.append(upper_bound)  # fixme - not needed, if handled properly

        # note - python_intervals package?
        # todo - needs optimization, multiple approaches possible
        # fixme - implement, such that it handles [] populations correctly
        subpopulation_index = 0
        subpopulation_start_index = 0
        for idx in range(len(sorted_individuals)):
            if sorted_individuals[idx].fitness >= cutting_points_values[subpopulation_index]:
                subpopulation_index += 1
