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
