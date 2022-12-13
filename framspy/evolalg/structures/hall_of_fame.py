class HallOfFame:
    """A simple function that keeps the specified number of individuals, adding only better ones.
    The list remains sorted from best to worst."""

    def __init__(self, hofsize):
        self.hofsize = hofsize
        self.hof = []

    def __iter__(self):
        return iter(self.hof)

    def __len__(self):
        return len(self.hof)

    def add(self, individual):
        if len(self.hof) < 1:  # empty hof?
            self.hof.append(individual)  # then add the first individual
        else:  # we have some individuals in hof?
            # only add if the new one is better than the first stored in hof
            if individual.rawfitness > self.hof[0].rawfitness:
                self.hof.insert(0, individual)  # add as first
        while len(self.hof) > self.hofsize:  # exceeded desired hof capacity?
            self.hof.pop()  # delete last (=worst)
