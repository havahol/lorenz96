import numpy as np
import time


class Lorenz96:
    
    def __init__(self, nx=8, ny=8, nz=8, num_layers=3, 
                 end_T=5, store_Y=False, store_Z=False,
                 init_X=None, init_Y=None, init_Z=None
                 ):
        
        self.nx, self.ny, self.nz = nx, ny, nz
        self.num_layers = num_layers
        
        if self.num_layers < 3:
            self.nz=1
        if self.num_layers < 2:
            self.ny=1
            
        if init_X is not None:
            assert(len(init_X) == self.nx), 'wrong size of init X'
        
        self.dt = 0.005
            
        self.end_T = end_T
        self.num_timesteps = int(self.end_T / 0.005)
        
        # Always store X
        self.X = np.zeros((self.num_timesteps+1, self.nx))
        
        # iteration counter
        self.it = 0
        self.t = 0.0
        
        # Initialzation:
        if init_X is not None:
            self.X[0,:] = init_X.copy()
        else:
            self.X[0,:] = np.random.normal(loc=0, scale=1, size=self.nx)
        
        if init_Y is not None:
            self.Y = init_Y.copy()
        else:
            self.Y = np.random.normal(loc=0, scale=1, size=(self.nx, self.ny))
        if init_Z is not None:
            self.Z = init_Z.copy()
        else:
            self.Z = np.random.normal(loc=0, scale=0.05, size=(self.nx,self.ny,self.nz))
        
        # Parameters (following those from Düben and Bauer, 2018)
        self.b, self.c, self.e, self.d = 10.0, 10.0, 10.0, 10.0
        self.h, self.g = 1.0, 1.0
        self.F = 20.0
        
        
        self.store_Y, self.store_Z = store_Y, store_Z
        if self.num_timesteps > 1000:
            if self.store_Y or self.store_Z:
                print('Ignoring store_Y and store_Z, since simulation will be too long')
            self.store_Y = False
            self.store_Z = False
        self.all_Y1, self.all_Z1 = None, None
        if self.store_Y:
            self.all_Y1 = np.zeros((self.num_timesteps+1, self.ny))
            self.all_Y1[0,:] = self.Y[0,:]
        if self.store_Z:
            self.all_Z1 = np.zeros((self.num_timesteps+1, self.nz))
            self.all_Z1[0,:] = self.Z[0, 0, :]
            
    def step(self, T, verbose=False, report_every=10000):
        """
        Step model forward until time T
        """
        
        T_in_iterations = int(T/self.dt)
        num_iterations = T_in_iterations - self.it
        
        tic = time.time()
        
        if self.it + num_iterations > self.num_timesteps:
            print('Asked for too many timesteps \n(allocated, already completed, asked for, would result in, will only do)\n', 
                  self.num_timesteps, self.it, num_iterations, self.it+num_iterations, self.num_timesteps - self.it)
            num_iterations = self.num_timesteps - self.it
        
        for i in range(num_iterations):
            self.X[self.it+1,:], self.Y, self.Z = self._rk4()
            
            if self.store_Y:
                self.all_Y1[self.it+1,:] = self.Y[0,:]
            if self.store_Z:
                self.all_Z1[self.it+1,:] = self.Z[0,0,:]
            
            self.it += 1
            
            if verbose and i > 0:
                if i%report_every == 0:
                    progress_in_percent = 100*i/num_iterations
                    toc = time.time()
                    print(str(int(progress_in_percent)).zfill(2)+' % after time (in s): ' + str(toc-tic))
    
    
    def _rk4(self):
        
        x1, y1, z1 = self._L96(self.X[self.it, :], self.Y, self.Z)
        
        x2, y2, z2 = self._L96(self.X[self.it, :] + x1*self.dt/2.0,
                               self.Y + y1*self.dt/2.0,
                               self.Z + z1*self.dt/2.0)
        
        x3, y3, z3 = self._L96(self.X[self.it, :] + x2*self.dt/2.0,
                               self.Y + y2*self.dt/2.0,
                               self.Z + z2*self.dt/2.0)
        
        x4, y4, z4 = self._L96(self.X[self.it, :] + x3*self.dt,
                               self.Y + y3*self.dt,
                               self.Z + z3*self.dt)
        
        new_x = self.X[self.it, :] + self.dt*(x1 + 2*x2 + 2*x3 + x4)/6.0
        new_y = self.Y + self.dt*(y1 + 2*y2 + 2*y3 + y4)/6.0
        new_z = self.Z + self.dt*(z1 + 2*z2 + 2*z3 + z4)/6.0
        
        return new_x, new_y, new_z

       
        
    def _L96(self, X, Y, Z):
        new_Y, new_Z = 0, 0
        
        new_X = np.roll(X,1)*(np.roll(X,-1) - np.roll(X,2)) - X + self.F
        if self.num_layers > 1:
            new_X -= (self.h*self.c/self.b)*Y.sum(axis=1)
            new_Y = -self.c*self.b*np.roll(Y,-1)*(np.roll(Y,-2) - np.roll(Y,1)) - self.c*Y + (self.h*self.c/self.b)*X[:, None] 
        if self.num_layers > 2:
            new_Y -= (self.h*self.e/self.d)*Z.sum(axis=2)
            new_Z = self.e*self.d*np.roll(Z,1)*(np.roll(Z,-1) - np.roll(Z,2)) - self.g*self.e*Z + (self.h*self.e/self.d)*Y[:,:,None]

        return new_X, new_Y, new_Z
        
