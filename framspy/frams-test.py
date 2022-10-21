"""Simple examples of using the "frams" module to communicate directly with the Framsticks library (dll/so).

For an introduction to Framsticks, its usage and scripting, see https://www.youtube.com/playlist?list=PLkPlXm7pOPatTl3_Gecx8ZaCVGeH4UV1L
For a list of available classes, objects, methods and fields, see http://www.framsticks.com/files/classdoc/
For a number of examples of scripting, see the "scripts" directory in the Framsticks distribution."""

import sys
import json
import frams

frams.init(*(sys.argv[1:]))  # pass whatever args we have, init() is the right place to deal with different scenarios:
# frams.init() - should try to figure out everything (and might fail)
# frams.init('path/to/lib') - load the library from the specified directory and configure Framsticks path as "data" inside this directory
# frams.init('path/to/lib','-d/tmp/workdir/data') - as above, but set the working (writable) directory somewhere else (see also -D)
# frams.init('path/to/lib','-Lframs-objects-alt.dll') - use specified library location and non-default file name

print('Available objects:', dir(frams))
print()


def extValueDetails(v):
	"""A helper function to display basic information about a variable of type ExtValue."""
	return '\t"' + str(v) + '"    frams type=' + str(v._type()) + '    frams class=' + str(v._class()) + '    python type=' + str(type(v._value()))


dic_as_string = '[100,2.2,"abc",[null,[],{}],XYZ[9,8,7]]'
print("We have the following string:\n\t'%s'" % dic_as_string)
print("Looks like a serialized dictionary, let's ask Framsticks String.deserialize() to do its job.")
v = frams.String.deserialize(dic_as_string)
print("Framsticks String.deserialize() returned\n\t", type(v))
print("More specifically, it is:")
print(extValueDetails(v))
print("Even though it is ExtValue (Framsticks' Vector), it supports iteration like a python vector, so let's inspect its elements:")
for e in v:
	print(extValueDetails(e))

print("\nNow let's play with the Framsticks simulator. Let's create a Genotype object and set fields in its custom 'data' dictionary.")
g = frams.GenePools[0].add('X')
g.name = "Snakis Py"
g.data['custom'] = 123.456
g.data['a'] = 'b'  # implicit conversion, looks like python dictionary but still converts '3' and '4' to ExtValue
dic = frams.Dictionary.new()  # let's create a Dictionary object from Framsticks
dic.set('1', '2')  # calling set() from Framsticks Dictionary
dic['3'] = '4'  # implicit conversion, looks like python dictionary but still converts '3' and '4' to ExtValue
g.data['d'] = dic
print(extValueDetails(g))

print("Let's add a few mutants and display their data:")
for more in range(5):
	frams.GenePools[0].add(frams.GenMan.mutate(g.geno))

for g in frams.GenePools[0]:
	print("\t%d. name='%s'\tgenotype='%s'\tdata=%s" % (g.index._value(), str(g.name), str(g.genotype), str(g.data)))

print("Let's now change some property of the simulation. Current water level is", frams.World.wrldwat)
frams.World.wrldwat = 0.5
print("Now water level is", frams.World.wrldwat)
frams.World.wrldwat = frams.World.wrldwat._value() + 0.7
print("Now water level is", frams.World.wrldwat)

initial_genotype = 'X(X,RX(X[T],X[G]))'  # simple body with touch and gyroscope sensors
print("Let's perform a few simulation steps of the initial genotype:", initial_genotype)
frams.ExpProperties.initialgen = initial_genotype
frams.ExpProperties.p_mut = 0  # no mutation (the selection procedure will clone our initial genotype)
frams.ExpProperties.p_xov = 0  # no crossover (the selection procedure will clone our initial genotype)
frams.Populations[0].initial_nn_active = 1  # immediate simulation of neural network - no "waiting for stabilization" period
frams.World.wrldg = 5  # gravity=5x default, let it fall quickly

frams.Simulator.init()  # adds initial_genotype to gene pool (calls onInit() from standard.expdef)
frams.Simulator.start()  # this does not actually start the simulation, just sets the "Simulator.running" status variable
step = frams.Simulator.step  # cache reference to avoid repeated lookup in the loop (just for performance)
for s in range(15):
	step()  # first step performs selection and revives one genotype according to standard.expdef rules
	creature = frams.Populations[0][0]  # FramScript Creature object
	mechpart0 = creature.getMechPart(0)
	print('Step# = %d' % frams.Simulator.stepNumber._value(),
	      '\tSimulated_creatures =', frams.Populations[0].size._value(),
	      "\tpart0_xyz = (% .2f,% .2f,% .2f)" % (mechpart0.x._value(), mechpart0.y._value(), mechpart0.z._value()),
	      "\ttouch = % .3f\tgyro = % .3f" % (creature.getNeuro(0).state._value(), creature.getNeuro(1).state._value()))
frams.Simulator.stop()

# changing expdef
testgenotype = "XrrX[G][-1:80][|,-1:0.9]X[|,-2:-21]"
evaluations = 100
print("\nLet's change the experiment definition (expdef) and evaluate genotype '%s' %d times." % (testgenotype, evaluations))
frams.Simulator.expdef = "standard-eval"
frams.ExpProperties.evalcount = evaluations
frams.ExpProperties.cr_v = 1
frams.ExpProperties.evalplan = ":velocity,vertpos,fit_stdev,time"
frams.GenePools[0].add(testgenotype)
frams.ExpProperties.evalsavefile = ""  # no need to store results in a file - we will get evaluations directly from Genotype's "data" field
frams.Simulator.init()
frams.Simulator.start()
# step = frams.Simulator.step  # cache reference to avoid repeated lookup in the loop (just for performance)
# while frams.Simulator.running._int():  # standard-eval.expdef sets running to 0 when the evaluation is complete
#	step()
frams.Simulator.eval("while(Simulator.running) Simulator.step();")  # loop in FramScript much faster than loop in python
for g in frams.GenePools[0]:  # loop over all genotypes, even though we know we added only one
	serialized_dict = frams.String.serialize(g.data[frams.ExpProperties.evalsavedata._value()])
	print(json.loads(serialized_dict._string()))

# sampling a Model in 3D
geno = "RXX(X,CXXX)"
print("\nNow build a Model from the genotype '%s' and sample it in 3D, then print a 2D projection" % geno)
import numpy as np

matrix = np.zeros((20, 20, 20), dtype=int)  # 3D matrix, "voxels"
m = frams.ModelGeometry.forModel(frams.Model.newFromString(geno));
m.geom_density = 20;
for p in m.voxels():
	# print('%f %f %f ' % (p.x._value(), p.y._value(), p.z._value()))
	matrix[int(p.x._value() * 5 + 2), int(p.y._value() * 5 + 5), int(p.z._value() * 5 + 6)] += 1  # scaling and offsets adjusted manually to fit the matrix nicely
matrix = np.sum(matrix, axis=1)  # sum along axis, make 2D from 3D ("projection")
np.set_printoptions(formatter={'int': lambda x: ('.' if x == 0 else str(x // 18))})  # print zeros as dots, x//18 to fit a larger range into a single digit
print(matrix)
np.set_printoptions()  # avoids straange python errors: frams.py, line 48, in __del__ AttributeError: 'NoneType' object has no attribute 'extFree'

#
#
# Note that implementing a complete expdef, especially a complex one, entirely in python may be inconvenient or impractical
# because you do not have access to "event handlers" like you have in FramScript - onStep(), onBorn(), onDied(), onCollision() etc.,
# so you would have to check various conditions in python in each simulation step to achieve the same effect.
