"""
NOTE - some ideas on how to approach modularity
"""

import experiment_abc

"""
@abstractmethod
make_new_pop:
    pass
"""

class Foo(experiment_abc):
    def __init__(self):
        self.make_new_pop = evolalg.make_new_pops.method1
        self.select = evolalg.selection.method1
        ...


"""
make_new_pop:
    (...)
"""

class Foo2(experiment_abc):
    def __init__(self):
        super()

class Foo3(experiment_abc):
    def __init__(self):
        super()
        self.make_new_pop = evolalg.make_new_pops.method1