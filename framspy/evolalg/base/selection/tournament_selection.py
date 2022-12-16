
def tournament_selection(individuals, tournament_size, random_index_sequence):
    """
    Tournament selection, returns the index of the best individual from those taking part in the tournament
    """
    # TODO - note that RIS should be a member, and this should be a class that inherits from selection
    best_index = None
    for i in range(tournament_size):
        rnd_index = random_index_sequence.getNext()
        if best_index is None or individuals[rnd_index].fitness > best_index.fitness:
            best_index = individuals[rnd_index]
    return best_index
