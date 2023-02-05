def ensure_numerical_within_space(genotype, lower: int = -100, upper: int = 100):
    """
    Ensures values of each dimension in the genotype are within the search space lower and upper bounds
    "Bounces back" each dimension that falls outside of the bounds by the difference
    """
    if not lower < upper:
        raise ValueError("not lower < upper")
    for i in range(len(genotype)):
        if genotype[i] > upper:
            genotype[i] = upper - (abs(genotype[i]) - abs(upper))
        elif genotype[i] < lower:
            genotype[i] = lower + (abs(genotype[i]) - abs(lower))

    return genotype