import numpy as np
import random

class k_means:

    seed = 3
    max_iters = 10000
    step = 0

    @classmethod
    def set_seed(cls, seed):
        cls.seed = seed

    def __init__(self, k):
        """ """ 
        self.k = k
        self.iteration = 0

    def _initialize_centroids(self):
        random.seed(self.seed)

        centroid_indices = random.sample(range(self.m),self.k)
        self.centroid_vect = self.X[:,centroid_indices]

    def _compute_L2_norm_distance(self, x_i):
        dist_vect = ((x_i - self.centroid_vect) ** 2).sum(axis = 0)
        allocated_center_indx = np.argmin(dist_vect, axis = 0)
        return allocated_center_indx

    def _allocate(self):
        labels_vect = np.zeros((self.m,1))

        for entry in range(self.m):
            x_i = self.X[:,entry].reshape(2,1)
            self.x_i_example = x_i
            allocated_center_indx = self._compute_L2_norm_distance(x_i)
            labels_vect[entry]= allocated_center_indx

        self.labels_vect = labels_vect

    def _calibrate(self):
        centroid_temp = np.empty((self.p,self.k))

        for label_i in range(self.k):
            label_bool_mask = (self.labels_vect.T == label_i).reshape(self.m)
            centroid = np.mean(self.X[:,label_bool_mask], axis=1)
            centroid_temp[:,label_i] = centroid

        self._check_convergence(centroid_temp)

        self.centroid_vect = centroid_temp

    def _check_convergence(self,vect):
        if self.step > 1:
            self.converged = np.all(self.centroid_vect==vect)
        else:
            self.converged = False

    def _early_stop(self):
        if self.iteration == self.max_iters:
            self.executed = True
        else:
            self.executed = False

    def _step(self):
        self.step += 1
        #print("\n","Step:", self.step)
        self._allocate()
        self._calibrate()

    def fit(self,X):
        self.X = np.array(X).T
        self.m = self.X.shape[1]
        self.p = self.X.shape[0]

        self._initialize_centroids()

        self.converged = False
        self.iteration = 0
        while self.converged == False and self.iteration < self.max_iters:
            self._step()
            self.iteration += 1
