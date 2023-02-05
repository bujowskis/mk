from ..constants import BAD_FITNESS


class Individual:
    only_positive_fitness = False  # fixme :)).....

    def __init__(self):
        self.genotype = None
        # used for stats. This is raw fitness value, None = not evaluated or invalid genotype
        self.rawfitness: float = None
        # used in selection and can be modified e.g. by diversity or niching techniques
        self.fitness: float = None

    def copyFrom(self, individual):  # "copying constructor"
        self.genotype = individual.genotype
        self.rawfitness = individual.rawfitness
        self.fitness = individual.fitness
        return self

    def copy(self):
        new_copy = Individual()
        new_copy.genotype = self.genotype
        new_copy.rawfitness = self.rawfitness
        new_copy.fitness = self.fitness
        return new_copy

    def set_and_evaluate(self, genotype, evaluate):
        self.genotype = genotype
        fitness = evaluate(genotype)
        if fitness is not BAD_FITNESS:  # BAD_FITNESS indicates that the genotype was not valid or some other problem occurred during evaluation
            if self.only_positive_fitness:
                if fitness < 0:
                    fitness = 0
        self.fitness = self.rawfitness = fitness

    def __str__(self):
        try:
            return "%g\t%g\t'%s'" % (self.rawfitness, self.fitness, self.genotype)
        except:
            return "%g\t'%s'" % (self.rawfitness, self.genotype)

    def __gt__(self, other):
        return self.rawfitness > other.rawfitness
