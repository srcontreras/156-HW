### Losses
import numpy as np
class MSE:
    def loss(self, predictions, y):

        l = np.mean((y - predictions)**2)
        return l
    def loss_gradient(self, predictions, y):
        grad = -2*(y - predictions)/y.shape[1]
        return grad.T
    
class Cross_Entropy:
    
    def loss(self, predictions, y):
        l = -np.sum(y * np.log(predictions + 1e-9)) / y.shape[1]
        return l
    def loss_gradient(self, predictions, y):
        l = -np.sum(y * np.log(predictions + 1e-9)) / y.shape[1]
        grad = -(1/y.shape[1]) * (y/(predictions + 1e-9))
        #compute here the cross entropy loss l as computed before, the result, grad, should be of dimension n x m, where m is the number of
        #points in the batch
        return grad.T