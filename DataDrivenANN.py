#Import packages we need
import numpy as np
import pandas as pd

import math

import tensorflow.keras 
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import RMSprop, SGD, Adagrad, Adadelta
import json

import time

def create_model_ANN():
    # Create Neural-network structure using the global structure
    model = Sequential()
    model.add(Dense(8, input_dim=8, activation='tanh'))
    model.add(Dense(100, activation='tanh'))
    model.add(Dense(100, activation='tanh'))
    model.add(Dense(100, activation='tanh'))
    model.add(Dense(100, activation='tanh'))
    model.add(Dense(8, activation='tanh'))
    
    # Compile model with stochastic gradient descent
    model.compile(loss='mean_absolute_error', optimizer='SGD', metrics=['mae'])

    return model


class DataDrivenLorenz96:
    
    def __init__(self, data_driven_model, nx=8,
                 end_T=5, init_X=None, 
                 mean=0, stddev=1, is_normalized=True
                 ):
        
        assert(data_driven_model.output_shape[1] == nx), 'nx does not match the data driven model'
        
        self.model = data_driven_model
        
        self.nx = nx
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

        self.mean, self.stddev = mean, stddev

        # Initialzation:
        if init_X is not None:
            if is_normalized:
                self.X[0,:] = init_X
            else:
                self.X[0,:] = (init_X-self.mean)/self.stddev
        else:
            self.X[0,:] = np.random.normal(loc=0, scale=1, size=self.nx)
        
         
        self.time_scheme = 1
        
        self.prev_dx = 0
        self.prev_prev_dx = 0
        
    def step(self, T, keras_step=True, use_multiprocessing=False, batch_size=1):
        
        T_in_iterations = int(T/self.dt)
        num_iterations = T_in_iterations - self.it
        
        tic = time.time()
        
        if self.it + num_iterations > self.num_timesteps:
            print('Asked for too many timesteps \n(allocated, already completed, asked for, would result in, will only do)\n', 
                  self.num_timesteps, self.it, num_iterations, self.it+num_iterations, self.num_timesteps - self.it)
            num_iterations = self.num_timesteps - self.it

        state = np.zeros((1,self.nx))

        for i in range(num_iterations):
            
            state[:] = self.X[self.it, :]
            
            #print(self.X[self.it,:].shape)
            #print(self.X[self.it,:])
            if keras_step:
                # batch_size = 1 seems to be best 
                #    (at least as long state only represent a single state, and not an ensemble)
                latest_dx = self.model.predict(state, batch_size=batch_size, use_multiprocessing=use_multiprocessing) #, use_multiprocessing=True)
            else:
                latest_dx = self._manual_evaluate_NN(state)
            
            if self.time_scheme == 1:
                dx = latest_dx
                self.time_scheme = 2
            elif self.time_scheme == 2:
                dx = 1.5*latest_dx - 0.5*self.prev_dx
                self.time_scheme = 3
            else:
                dx = (23.0/12.0)*latest_dx-(4.0/3.0)*self.prev_dx+(5.0/12.0)*self.prev_prev_dx
            
            self.X[self.it+1,:] = self.X[self.it,:] + dx
            
            self.prev_prev_dx = self.prev_dx
            self.prev_dx = latest_dx
            self.it += 1
        
    def _manual_evaluate_NN(self, state):
        for layer in self.model.layers:
            #print('starting layer')
            #w = layer.weights[0].numpy()
            #b = layer.bias.numpy()
            #state = np.tanh(np.dot(state, w) + b)
            
            state = np.tanh(np.dot(state, layer.weights[0].numpy()) + layer.bias.numpy())
        return state  
            
    def getActualState(self):
        return self.X[self.it, :]*self.stddev + self.mean
    
    def getActualTimeseries(self):
        return self.X*self.stddev + self.mean

    def setCurrentState(self, X, is_normalized=True):
        if is_normalized:
            self.X[self.it, :] = X
        else:
            self.X[self.it, :] = (X-self.mean)/self.stddev
        
    
class DataDrivenEnsemble:
    
    def __init__(self, Ne, datadriven_model_params=None):
        
        self.Ne = Ne
        
        self.ensemble = [None]*self.Ne
        for i in range(self.Ne):
            self.ensemble[i] = DataDrivenLorenz96(**datadriven_model_params)
            
        self.mean = None
        self.it = 0
        
    def perturb_all_states(self, loc=0, scale=1):
        for member in self.ensemble:
            perturbedState = member.getActualState() + np.random.normal(loc=loc, scale=scale, size=member.nx)
            member.setCurrentState(perturbedState, is_normalized=False)
            
        
    def step(self, T, keras_step=True):
        for member in self.ensemble:
            member.step(T, keras_step=keras_step)
        self.update_mean()
        self.it = self.ensemble[0].it
    
    def update_mean(self):
        self.mean = np.zeros_like(self.ensemble[0].X)
        for member in self.ensemble:
            self.mean += member.X
        self.mean /= self.Ne
            
    def getActualMean(self):
        return self.mean*self.ensemble[0].stddev + self.ensemble[0].mean
    
    def getCurrentMean(self):
        return self.getActualMean()[self.it, :]
    
    