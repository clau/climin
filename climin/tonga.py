# -*- coding: utf-8 -*-

import scipy
import numpy as np
import numpy.linalg
import scipy.optimize
import itertools
import math

from base import Minimizer, repeat_or_iter


class tonga(Minimizer):

    def __init__(self, wrt, fgrad, fjacobian, damping, blocksizes,
                 gamma=0.995, b=50, k=5,
                 args=None, logfunc=None):
        super(tonga, self).__init__(
            wrt, args=args, logfunc=logfunc)

        self.fgrad = fgrad
        self.fjacobian = fjacobian
        self.damping = damping
        self.blocksizes = blocksizes
        self.gamma = gamma
        self.gamma_sqrt = math.sqrt(self.gamma)
        self.b = b #TODO find a meaningful name
        self.k = k

          

        
    def __iter__(self):
        X_m1 = np.zeros((1, self.wrt.size))
        oldGrad = np.zeros((self.b -1, self.wrt.size))
        step = scipy.empty(self.wrt.size)

        args, kwargs = self.args.next()

        
        for i, (next_args, next_kwargs) in enumerate(self.args):            
            offset = 0
            if (i==0):
                gradient_mean = self.fgrad(self.wrt, *args, **kwargs)
                X = scipy.empty((self.wrt.size,1)) 
                X[:,0] = gradient_mean
            elif ((i%self.b)!=0):
                gradient_mean = self.fgrad(self.wrt, *args, **kwargs)
                X = scipy.empty((self.wrt.size, X_m1[0].size + 1))
                X[:,:-1] = X_m1 * self.gamma_sqrt
                X[:,-1] = gradient_mean
            else:
                gradient = self.fjacobian(self.wrt, *args, **kwargs)
                gradient_mean = gradient.mean(axis=0)
                X = scipy.empty((self.wrt.size, self.k + self.b)) 

                
            for size in self.blocksizes:
                if (i%self.b==0) and (i>0):

                    factor = [self.gamma_sqrt**power for power in range(self.b-1,0,-1)]
                    X[offset:offset+size, self.k:self.k+self.b-1] = factor * oldGrad.T[offset:offset+size]
                    
                    X[offset:offset+size, self.k+self.b-1] = self.gamma_sqrt**(self.b+i) * gradient_mean[offset:offset+size]
                    
                    grad = gradient[:, offset:offset+size]
                    covariance = scipy.dot(grad.T, grad)
                    V, U = np.linalg.eig(covariance)
                    ind = V.argsort()
                    X[offset:offset+size, :self.k] = self.gamma * (U[:, ind])[:, -self.k:]
                                                                 

                x = X[offset:offset+size]
                step[offset:offset+size] = scipy.dot(x, scipy.linalg.inv(scipy.dot(x.T,x)+ self.damping*scipy.eye(len(x[0]))))[:,-1]
                offset += size
                

            #storing the old gradients
            if ((i%self.b)==0) and (i>0):
                oldGrad = np.zeros((self.b -1, self.wrt.size))
            else:
                oldGrad[i%self.b-1] = gradient_mean
                
            self.wrt -= step
            X_m1 = X

            
            
            yield {
                'gradient':gradient_mean,
                'args':args,
                'kwargs':kwargs,
                'n_iter':i,
                'step':step,
            }
        
            args, kwargs = next_args, next_kwargs

            