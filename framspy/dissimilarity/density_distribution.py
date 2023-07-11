import numpy as np
from pyemd import emd
from ctypes import cdll
from ctypes.util import find_library
from alignmodel import align

class DensityDistribution:
    """Two dissimilarity measures based on the spatial distribution of two Models. The Model bounding box is divided into a grid of equally-sized cuboids, the number of which is the 'resolution' parameter cubed. Then the Model surface is covered with points; the density of the surface sampling is determined by the 'density' parameter. There are two versions of the measure. In the default version ('frequency'=False), a signature of each cuboid is the centroid and the number of samples. In the 'frequency'=True version, FFT is computed from the vector containing the number of samples in each cuboid. The final result of the dissimilarity measure is the distance between the signatures and it can be computed using EMD, L1, or L2 norms (the 'metric' parameter).
    """
    
    libm = cdll.LoadLibrary(find_library('m')) # for disabling/enabling floating point exceptions (division by zero occurs in the EMD library)
    EPSILON = 0.0001
    
    def __init__(self, frams_module=None, density = 10, resolution = 8, reduce_empty=True, frequency=False, metric = 'emd', fixedZaxis=False, verbose=False):
        """ __init__
        Args:
            density (int, optional): density of samplings for frams.ModelGeometry. Defaults to 10.
            resolution (int, optional): How many intervals are used in each dimension to partition surface samples of Models in the 3D space. 
                The higher the value, the more detailed the comparison and the longer the calculations. Defaults to 3.
            reduce_empty (bool, optional): If we should use reduction to remove blank samples. Defaults to True.
            frequency (bool, optional): If we should use frequency distribution. Defaults to False.
            metric (string, optional): The distance metric that should be used ('emd', 'l1', or 'l2'). Defaults to 'emd'.
            fixedZaxis (bool, optional): If the z axis should be fixed during alignment. Defaults to False.
            verbose (bool, optional): Turning on logging, works only for calculateEMDforGeno. Defaults to False.            
        """
        if frams_module is None:
            raise ValueError('Framsticks module not provided!')
        self.frams = frams_module

        self.density = density
        self.resolution = resolution
        self.verbose = verbose
        self.reduce_empty = reduce_empty
        self.frequency = frequency
        self.metric = metric
        self.fixedZaxis = fixedZaxis


    def calculateNeighberhood(self,array,mean_coords):
        """ Calculates number of elements for given sample and set ups the center of this sample
        to the center of mass (calculated by mean of every coordinate)
        Args:
            array ([[float,float,float],...,[float,float,float]]): array of voxels that belong to given sample.
            mean_coords ([float,float,float]): default coordinates that are the
                middle of the sample (used when number of voxels in sample is equal to 0)

        Returns:
            weight [int]: number of voxels in a sample
            coordinates [float,float,float]: center of mass for a sample
        """
        weight = len(array)
        if weight > 0:
            point = [np.mean(array[:,0]),np.mean(array[:,1]),np.mean(array[:,2])]
            return weight, point
        else:
            return 0, mean_coords


    def calculateDistPoints(self,point1, point2):
        """ Returns euclidean distance between two points
        Args (distribution):
            point1 ([float,float,float]) - coordinates of first point
            point2 ([float,float,float]) - coordinates of second point
        Args (frequency):
            point1 (float) - value of the first sample
            point2 (float) - value of the second sample

        Returns:
            [float]: euclidean distance
        """
        if self.frequency:
            return abs(point1-point2)
        else:
            return np.sqrt(np.sum(np.square(point1-point2)))


    def calculateDistanceMatrix(self,array1, array2):
        """
        Args:
            array1 ([type]): array of size n with points representing the first Model 
            array2 ([type]): array of size n with points representing the second Model

        Returns:
            np.array(np.array(,dtype=float)): distance matrix n*n 
        """
        n = len(array1)
        distMatrix = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                distMatrix[i][j] = self.calculateDistPoints(array1[i], array2[j])
        return np.array(distMatrix)


    def reduceEmptySignatures_Frequency(self,s1,s2):
        """Removes samples from signatures if corresponding samples for both models have weight 0.
        Args:
            s1 (np.array(,dtype=np.float64)): values of samples
            s2 (np.array(,dtype=np.float64)): values of samples

        Returns:
            s1new (np.array(,dtype=np.float64)): coordinates of samples after reduction
            s2new (np.array(,dtype=np.float64)): coordinates of samples after reduction
        """
        lens = len(s1)
        indices = []
        for i in range(lens):
            if s1[i]==0 and s2[i]==0:
                    indices.append(i)

        return np.delete(s1, indices), np.delete(s2, indices)


    def reduceEmptySignatures_Density(self,s1,s2):
        """Removes samples from signatures if corresponding samples for both models have weight 0. 
        Args:
            s1 ([np.array(,dtype=np.float64),np.array(,dtype=np.float64)]): [coordinates of samples, weights]
            s2 ([np.array(,dtype=np.float64),np.array(,dtype=np.float64)]): [coordinates of samples, weights]

        Returns:
            s1new ([np.array(,dtype=np.float64),np.array(,dtype=np.float64)]): [coordinates of samples, weights] after reduction
            s2new ([np.array(,dtype=np.float64),np.array(,dtype=np.float64)]): [coordinates of samples, weights] after reduction
        """
        lens = len(s1[0])
        indices = []
        for i in range(lens):
            if s1[1][i]==0 and s2[1][i]==0:
                indices.append(i)

        s1 = [np.delete(s1[0], indices, axis=0), np.delete(s1[1], indices, axis=0)]
        s2 = [np.delete(s2[0], indices, axis=0), np.delete(s2[1], indices, axis=0)]
        return s1, s2


    def getSignatures(self,array,edges3,steps3):
        """Generates signature for array representing the Model. Signature is composed of list of points [x,y,z] (float) and list of weights (int).

        Args:
            array (np.array(np.array(,dtype=float))): array with voxels representing the Model
            edges3 ([np.array(,dtype=float),np.array(,dtype=float),np.array(,dtype=float)]): lists with edges for each step for each axis in order x,y,z
            steps3 ([float,float,float]): [size of interval for x axis, size of interval for y axis, size of interval for y axis] 

        Returns (distribution):
           signature [np.array(,dtype=np.float64),np.array(,dtype=np.float64)]: returns signatuere [np.array of points, np.array of weights]
        Returns (frequency):
           signature np.array(,dtype=np.float64): returns signatuere np.array of coefficients
        """
        edges_x,edges_y,edges_z = edges3
        step_x,step_y,step_z=steps3
        feature_array = []
        weight_array = []
        step_x_half = step_x/2
        step_y_half = step_y/2
        step_z_half = step_z/2
        for x in range(len(edges_x[:-1])):
            for y in range(len(edges_y[:-1])) :
                for z in range(len(edges_z[:-1])):
                    rows=np.where((array[:,0]> edges_x[x]) &
                                  (array[:,0]<= edges_x[x+1]) &
                                  (array[:,1]> edges_y[y]) &
                                  (array[:,1]<= edges_y[y+1]) &
                                  (array[:,2]> edges_z[z]) &
                                  (array[:,2]<= edges_z[z+1]))
                    if self.frequency:
                        feature_array.append(len(array[rows]))
                    else:
                        weight, point = self.calculateNeighberhood(array[rows],[edges_x[x]+step_x_half,edges_y[y]+step_y_half,edges_z[z]+step_z_half])
                        feature_array.append(point)
                        weight_array.append(weight)

        if self.frequency:
            samples = np.array(feature_array,dtype=np.float64)
            return abs(np.fft.fft(samples))
        else:
            return [np.array(feature_array,dtype=np.float64), np.array(weight_array,dtype=np.float64)]


    def getSignaturesForPair(self,array1,array2):
        """Generates signatures for given pair of models represented by array of voxels.
        We calculate space for given models by taking the extremas for each axis and dividing the space by the resolution.
        This divided space generate us samples which contains points. Each sample will have new coordinates which are mean of all points from it and weight which equals to the number of points.
       
        Args:
            array1 (np.array(np.array(,dtype=float))): array with voxels representing model1
            array2 (np.array(np.array(,dtype=float))): array with voxels representing model2

        Returns:
            s1 ([np.array(,dtype=np.float64),np.array(,dtype=np.float64)]): [coordinates of samples, weights] 
            s2 ([np.array(,dtype=np.float64),np.array(,dtype=np.float64)]): [coordinates of samples, weights]
        """

        min_x = np.min([np.min(array1[:,0]),np.min(array2[:,0])])
        max_x = np.max([np.max(array1[:,0]),np.max(array2[:,0])])
        min_y = np.min([np.min(array1[:,1]),np.min(array2[:,1])])
        max_y = np.max([np.max(array1[:,1]),np.max(array2[:,1])])
        min_z = np.min([np.min(array1[:,2]),np.min(array2[:,2])])
        max_z = np.max([np.max(array1[:,2]),np.max(array2[:,2])])

        # We request self.resolution+1 samples since we need self.resolution intervals
        edges_x,step_x = np.linspace(min_x,max_x,self.resolution+1,retstep=True)
        edges_y,step_y = np.linspace(min_y,max_y,self.resolution+1,retstep=True)
        edges_z,step_z = np.linspace(min_z,max_z,self.resolution+1,retstep=True)
        
        for edges in (edges_x, edges_y, edges_z):  # EPSILON subtracted to deal with boundary voxels (one-sided open intervals and comparisons in loops in function getSignatures())
            edges[0] -= self.EPSILON

        edges3 = (edges_x,edges_y,edges_z)
        steps3 = (step_x,step_y,step_z)
        
        s1 = self.getSignatures(array1,edges3,steps3)
        s2 = self.getSignatures(array2,edges3,steps3)    
        
        return s1,s2


    def getVoxels(self,geno):
        """Generates voxels for genotype using frams.ModelGeometry

        Args:
            geno (string): representation of Model in one of the formats supported by Framsticks, http://www.framsticks.com/a/al_genotype.html

        Returns:
            np.array([np.array(,dtype=float)]: list of voxels representing the Model.
        """
        model = self.frams.Model.newFromString(geno)
        align(model, self.fixedZaxis)
        model_geometry = self.frams.ModelGeometry.forModel(model)

        model_geometry.geom_density = self.density
        voxels = np.array([np.array([p.x._value(),p.y._value(),p.z._value()]) for p in model_geometry.voxels()])
        return voxels


    def calculateDissimforVoxels(self, voxels1, voxels2):
        """Calculates EMD for pair of voxels representing models.
        Args:
            voxels1 np.array([np.array(,dtype=float)]: list of voxels representing model1.
            voxels2 np.array([np.array(,dtype=float)]: list of voxels representing model2.

        Returns:
            float: dissim for pair of list of voxels
        """
        numvox1 = len(voxels1)
        numvox2 = len(voxels2)    

        s1, s2 = self.getSignaturesForPair(voxels1, voxels2)

        reduce_fun = self.reduceEmptySignatures_Frequency if self.frequency else self.reduceEmptySignatures_Density
        if self.reduce_empty:
            s1, s2 = reduce_fun(s1,s2)

            if not self.frequency:
                if numvox1 != sum(s1[1]) or numvox2 != sum(s2[1]):
                    print("Voxel reduction didn't work properly")
                    print("Base voxels fig1: ", numvox1, " fig2: ", numvox2)
                    print("After reduction voxels fig1: ", sum(s1[1]), " fig2: ", sum(s2[1]))
                    raise RuntimeError("Voxel reduction error!")
        
        if self.metric == 'l1':
            if self.frequency:
                out = np.linalg.norm((s1-s2), ord=1)
            else:
                out = np.linalg.norm((s1[1]-s2[1]), ord=1)

        elif self.metric == 'l2':
            if self.frequency:
                out = np.linalg.norm((s1-s2))
            else:
                out = np.linalg.norm((s1[1]-s2[1]))

        elif self.metric == 'emd':
            if self.frequency:
                num_points = len(s1)
                dist_matrix = self.calculateDistanceMatrix(range(num_points),range(num_points))
            else:
                dist_matrix = self.calculateDistanceMatrix(s1[0],s2[0])

            self.libm.fedisableexcept(0x04)  # change default flag value - don't cause exceptions when dividing by 0 (pyemd does it)

            if self.frequency:
                out = emd(s1,s2,np.array(dist_matrix,dtype=np.float64))
            else:
                out = emd(s1[1],s2[1],dist_matrix)

            self.libm.feclearexcept(0x04) # restoring default flag values...
            self.libm.feenableexcept(0x04)

        else:
            raise ValueError("Wrong metric '%s'"%self.metric)

        return out


    def calculateDissimforGeno(self, geno1, geno2):
        """Calculates EMD for a pair of genotypes.
        Args:
            geno1 (string): representation of model1 in one of the formats supported by Framsticks, http://www.framsticks.com/a/al_genotype.html
            geno2 (string): representation of model2 in one of the formats supported by Framsticks, http://www.framsticks.com/a/al_genotype.html

        Returns:
            float: dissim for pair of strings representing models.
        """     

        voxels1 = self.getVoxels(geno1)
        voxels2 = self.getVoxels(geno2)

        out = self.calculateDissimforVoxels(voxels1, voxels2)

        if self.verbose == True:
            print("Intervals: ", self.resolution)
            print("Geno1:\n",geno1)
            print("Geno2:\n",geno2)
            print("EMD:\n",out)

        return out


    def getDissimilarityMatrix(self,listOfGeno):
        """
        Args:
            listOfGeno ([string]): list of strings representing genotypes in one of the formats supported by Framsticks, http://www.framsticks.com/a/al_genotype.html

        Returns:
            np.array(np.array(,dtype=float)): dissimilarity matrix of EMD for given list of genotypes
        """
        numOfGeno = len(listOfGeno)
        dissimMatrix = np.zeros(shape=[numOfGeno,numOfGeno])
        listOfVoxels = [self.getVoxels(g) for g in listOfGeno]
        for i in range(numOfGeno):
            for j in range(numOfGeno):
                dissimMatrix[i,j] = self.calculateDissimforVoxels(listOfVoxels[i], listOfVoxels[j])
        return dissimMatrix
