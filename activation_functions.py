'''
Created on 18 de dez de 2015

@author: Fellipe
'''
import numpy as np
from scipy.special import expit

class ActivationFunction():
    def __call__(self,z):
        raise Exception('Not implemented')
    def derivative(self,z):
        raise Exception('Not implemented')
    
class Sigmoid(ActivationFunction):
    def __call__(self,z):
        """The sigmoid function."""
        return expit(z)
    def derivative(self,z):
        """Derivative of the sigmoid function."""
        return self(z)*(1-self(z))

class Tanh(ActivationFunction):
    def __call__(self,z):
        """The sigmoid function."""
        return np.tanh(z)
    def derivative(self,z):
        """Derivative of the sigmoid function."""
        return 1-self(z)**2