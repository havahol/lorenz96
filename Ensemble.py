import numpy as np
import Lorenz96

class Ensemble:
    
    def __init__(self, Ne, physical_model=True, physical_model_params=None):
        
        self.Ne = Ne
        
        self.ensemble = [None]*self.Ne
        for i in range(self.Ne):
            self.ensemble[i] = Lorenz96.Lorenz96(**physical_model_params)
            
        self.mean = None
        self.it = 0
        
    def perturb_all_states(self, loc=0, scale=1):
        for member in self.ensemble:
            member.X[self.it,:] += np.random.normal(loc=loc, scale=scale, size=member.nx)
            
        
    def step(self, T, keras_step=True):
        for member in self.ensemble:
            member.step(T)
        self.update_mean()
        self.it = self.ensemble[0].it
    
    def update_mean(self):
        self.mean = np.zeros_like(self.ensemble[0].X)
        for member in self.ensemble:
            self.mean += member.X
        self.mean /= self.Ne
            
    def getActualMean(self):
        self.update_mean()
        return self.mean
    
    def getCurrentMean(self):
        return self.mean[self.it, :]
        