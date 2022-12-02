class Individual:
    only_positive_fitness = True  # TODO - accept as param? (unelegant, moved into cmd param)
    """
    (mention fitness may be float or dict of float fitnesses / vector)
    """

    def __init__(self):
        self.genotype = None
        self.rawfitness = None  # used for stats. This is raw fitness value, None = not evaluated or invalid genotype
        self.fitness = None  # used in selection and can be modified e.g. by diversity or niching techniques

    def copyFrom(self, individual):  # "copying constructor"
        self.genotype = individual.genotype
        self.rawfitness = individual.rawfitness
        self.fitness = individual.fitness
        return self

    def copy(self):  # FIXME - deep_copy()?
        new_copy = Individual()
        new_copy.genotype = self.genotype
        new_copy.rawfitness = self.rawfitness
        new_copy.fitness = self.fitness
        return new_copy

    def setAndEvaluate(self, genotype, evaluate):
        self.genotype = genotype
        fitness = evaluate(genotype)
        if fitness is not None:  # BAD_FITNESS is None and indicates that the genotype was not valid or some other problem occurred during evaluation
            fitness = fitness[0]  # get first criterion
            if self.only_positive_fitness:
                if fitness < 0:
                    fitness = 0
        self.fitness = self.rawfitness = fitness

    # TODO - other methods (lt, eq, etc.)? (though sufficient)
    def __str__(self):
        try:
            return "%g\t%g\t'%s'" % (self.rawfitness, self.fitness, self.genotype)
        except:  # fixme
            return "%g\t'%s'" % (self.rawfitness, self.genotype)

    def __gt__(self, other):
        return self.rawfitness > other.rawfitness
