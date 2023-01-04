from experiment_convection_selection import ExperimentConvectionSelection

# todo - try running in SLURM / super computers (slower than SLURM, but a workaround if sth wrong with SLURM)
#   GECKO - numerical benchmarks (fast enough) framsticks - too slow
#   writing the paper - the most time-consuming, but may go for CEC2004(?) benchmarks (DEAP benchmark functions?)

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

# check if works for empty subpopulation case
#   just copy individuals from the next worser, non-empty subpopulation - don't delete from lower; # todo - implement as reusable function
#   steady-state would delete the excess individuals - random deletion should improve diversity # todo - implement as reusable function
#   todo (future) - more elegant implementation of keeping the desired no. of individuals within the subpopulations

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
        # note - may contact KacperPerz/evolalg
        pool_of_all_individuals = []
        for p in self.populations:
            pool_of_all_individuals.extend(p.population)
        sorted_individuals = sorted(pool_of_all_individuals, key=lambda x: x.rawfitness)  # todo - this or max() and min() (may be more efficient to use sorted) - + dynamic programming (slicing)

        # 1.0, 1.2, 1.5, 2.0, 4.0
        # [1.0, 2.0], (2.0, 3.0]
        # ^ if this is inclusive, the copying mechanism handles some edge cases todo - test if that's the case
        # OR invoke equinumber in edge case max == min

        lower_bound = sorted_individuals[0].fitness
        upper_bound = sorted_individuals[len(sorted_individuals)-1].fitness
        population_width = (upper_bound - lower_bound) / self.number_of_populations
        # population_ranges = [(lower_bound + x*population_width, lower_bound + (x+1)*population_width)
        #                      for x in range(self.number_of_populations)]
        cutting_points_values = [lower_bound + x*population_width for x in range(1, self.number_of_populations)]  # todo - check for case with 1 pop
        # fixme - use the ub as next lb
        cutting_points_values.append(upper_bound)  # fixme - not needed, if handled properly

        # note - python_intervals package?
        # todo - needs optimization, multiple approaches possible
        # fixme - implement, such that it handles [] populations correctly
        subpopulation_index = 0
        subpopulation_start_index = 0
        for idx in range(len(sorted_individuals)):
            if sorted_individuals[idx].fitness >= cutting_points_values[subpopulation_index]:
                subpopulation_index += 1
