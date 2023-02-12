# Convection Selection Młoda Kadra Project
- Framsticks50rc23

## NOTES TO SELF
- when using PyCharm, opening `framspy`

## Setup - Windows
### Structure / workspace
- scripts work if they go directly into framspy folder
- `todo` - what if I want to use framspy anywhere else, create scripts outside framspy dir?
  - (basically, that's how it's done when using all kinds of modules, packages)

### Environment Variables
```commandline
setx FRAMS_PATH "C:\Users\ADMIN\PycharmProjects\mk\framspy"
```
```commandline
setx DIR_WITH_FRAMS_LIBRARY "C:\Users\ADMIN\PycharmProjects\mk\framspy"
```
```commandline
setx FRAMSTICKS_DIR "C:\Users\ADMIN\Downloads\Framsticks50rc23"
```

## How stuff works
- [gdoc](https://docs.google.com/document/d/1LAyeRrTTnjC1XVllBG8wzMZP7a-6MYHS5lWjI2oFwi0/edit?usp=sharing)

## Base of articles:

- [Niching](https://drive.google.com/file/d/1XP7q9zo72OYlNCa-IFHaI9lHTtRJomYN/view)

- [Diversification Methods](https://drive.google.com/file/d/1XI1p5CiWTVcgzPiBgKKXUSb0-4IlgrNB/view)

- [Convection Selection Experiments](http://www.framsticks.com/files/common/TournamentBasedConvectionSelectionEvolutionary.pdf)

- [Multithreading Explained](http://www.framsticks.com/files/common/MultithreadedEvolutionaryDesign.pdf)


In the convection selection schemes, individuals are first sorted according to
their fitness. Then each subpopulation receives a subset of individuals that fall
within a range of fitness values. In our experiments, two methods of determining
fitness ranges are considered. In the first method denoted EqualWidth (Fig. 1c),
the entire fitness range has been divided into equal intervals (as many as there
are subpopulations); if there are no individuals in some fitness range, the corresponding subpopulation receives individuals from the nearest lower non-empty
fitness interval. In the second method denoted EqualNumber (Fig. 1d), once the
individuals are sorted according to their fitness, they are divided into as many
sets as there are subpopulations so that each subpopulation receives the same
number of individuals.

<img src="https://github.com/bujowskis/mk/blob/master/Convection%20Selection%20Scheme.jpg" width="500" height="250" />

Fig. 1: An illustration of four compared selection schemes. The fitness of 20 individuals is shown as red circles, and 4 subpopulations are depicted as green
boxes. (a) Standard evolutionary algorithm with a single population. (b) Random assignment of individuals to subpopulations. (c) Convection selection with
fitness intervals of equal width. (d) Convection selection with fitness intervals
yielding equal number of individuals.

## Note

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

#### Example:

Initially:
Slave1 state = [0.05, 0.08, 0.12]

(...)

1. Slave1Step [0.0, 0.2] -> state = [0.1, 0.15, 0.23]
2. Slave2Step (0.2, 0.4]

(...)

5. Slave5Step (0.8, 1.0]
6. Master

      6.1. Check Slave1 state = [0.1, 0.15, **0.23**]

      6.2. Move "too good" individuals up -> Slave1 state = [0.1, 0.15, None]; Slave2 state = state + [0.23]

      (6.3.) Inject new, random/some-other-way, into Slave1

      (gen += 1)
