import numpy as np
from abc import ABC , abstractmethod

class generate_loss_function(ABC):
    def parameters_init(self,y,y_pred):
        self.y,self.y_pred = y,y_pred

    @abstractmethod
    def mean_squared_error(self):
        ...

    @abstractmethod
    def mean_absolute_error(self):
        ...

    @abstractmethod
    def log_loss(self):
        ...
        
    @abstractmethod    
    def hinge_loss(self):
        ...

class classic_loss_function(generate_loss_function):
    def __init__(self) -> None:
        super().__init__()
        # print('loading classic.')

    def mean_squared_error(self):
        return np.mean(np.square(self.y_pred - self.y))

    def mean_absolute_error(self):
        return np.mean(np.abs(self.y_pred - self.y))

    def log_loss(self):
        return np.mean(self.y * np.log(self.y_pred) + (1-self.y) * np.log(1-self.y_pred))
    
    def hinge_loss(self):
        return np.sum(np.max(0,1-self.y*self.y_pred))    

class regularization_loss_function(classic_loss_function):
    def __init__(self) -> None:
        super().__init__()
        # print('loading regularization.')

    def parameters_init(self,y,y_pred,num_samples,theta,regularization_param):
        super().parameters_init(y,y_pred)
        self.num_samples,self.theta,self.regularization_param = num_samples,theta,regularization_param
    
    def regularization(self):
        return (self.regularization_param / self.num_samples) * np.sum(np.square(self.theta))

    def mean_squared_error(self):
        return super().mean_squared_error() +\
            self.regularization()

    def mean_absolute_error(self):
        return super().mean_absolute_error() +\
            self.regularization()

    def log_loss(self):
        return super().log_loss() +\
            self.regularization()