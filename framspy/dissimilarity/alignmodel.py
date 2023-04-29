import math
import numpy as np

def wcentre(matrix, weights):
    sw = weights.sum()
    swx = (matrix*weights).sum(axis=1)
    swx /= sw
    return (matrix.transpose()-swx).transpose()*np.sqrt(weights)
    
def weightedMDS(distances, weights):
    n = len(weights)
    distances = distances**2
    for i in range(2):
        distances = wcentre(distances, weights)
        distances = distances.T
    distances *= -0.5
    _, eigenvalues, vh = np.linalg.svd(distances)
    W = (vh/np.sqrt(weights)).T
    S = np.zeros((n,n))
    np.fill_diagonal(S, eigenvalues)
    S = S**0.5
    dcoords = W.dot(S)
    coords = np.zeros((n, 3))
    coords[:,0]=dcoords[:,0]
    for i in range(1,3):
        if n>i:
            coords[:,i]=dcoords[:,i]
    return coords
    

def align(model, fixedZaxis=False):
	numparts=model.numparts._value()
	distmatrix = np.zeros((numparts, numparts), dtype=float)
	for p1 in range(numparts):
		for p2 in range(numparts): #TODO optimize, only calculate a triangle
			P1=model.getPart(p1)
			P2=model.getPart(p2)
			if fixedZaxis:
				#fixed vertical axis, so pretend all points are on the xy plane
				z_dist = 0
			else:
				z_dist = (P1.z._value()-P2.z._value())**2
			distmatrix[p1,p2]=math.sqrt((P1.x._value()-P2.x._value())**2+(P1.y._value()-P2.y._value())**2+z_dist)
	
	if model.numjoints._value() > 0:
		weightvector=np.zeros((numparts), dtype=int)
	else:
		weightvector=np.ones((numparts), dtype=int)
	
	for j in range(model.numjoints._value()):
		J=model.getJoint(j)
		weightvector[J.p1._value()]+=1
		weightvector[J.p2._value()]+=1
	weightvector=weightvector.astype(float) # convert to float once, since later it would be promoted to float so many times anyway...
	coords = weightedMDS(distmatrix, weightvector)

	# update parts positions
	n = len(weightvector)
	for p in range(numparts):
		P = model.getPart(p)
		P.x = coords[p, 0]
		if n > 1:
			P.y = coords[p, 1]
		if n > 2:
			if not fixedZaxis:
				P.z = coords[p, 2]


	if fixedZaxis:
		if np.shape(coords)[1] > 2:
		#restore original z coordinate
			for p in range(numparts):
				P=model.getPart(p)
				coords[p,2]=P.z._value()

