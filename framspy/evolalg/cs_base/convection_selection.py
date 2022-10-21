"""
The main idea of the Convection Selection:
- Slaves - evolutionary algorithms in their own
    - similar fitness values between individuals in one Slave
- Master - migrate individuals between Slaves
    - (all) genotypes sorted according to fitness

First method - equiWidth
- divide the fitness range equally between Slaves
- supposed the fitness range is [0, 1], 5 Slaves,
    - Slave1 = [0.0, 0.2]
    - Slave2 = (0.2, 0.4]
    - (...)
    - Slave5 = (0.8, 1.0]
- it may happen some Slave's population is empty
- worst-case scenario - it's the same as running a single evolutionary algorithm with no distribution

Second method - equiNumber
- divide the individuals equally between slaves
- supposed the population [i0, i1, (...), i9] is sorted according to fitness (worst-best)
    - Slave1 population = [i0, i1]
    - Slave2 population = [i2, i3]
    - (...)
    - Slave5 population = [i8, i9]
- the individuals are always spread equally between
- more complex
"""

# Initially:
# Slave1 state = [0.05, 0.08, 0.12]
# (...)

#   1. Slave1Step [0.0, 0.2] -> state = [0.1, 0.15, 0.23]
#   2. Slave2Step (0.2, 0.4]
#   (...)
#   5. Slave5Step (0.8, 1.0]
#   6. Master
#       6.1. Check Slave1 state = [0.1, 0.15, **0.23**]
#       6.2. Move "too good" individuals up -> Slave1 state = [0.1, 0.15, None]; Slave2 state = state + [0.23]
#       (6.3.) Inject new, random/some-other-way, into Slave1
#   (gen += 1)
