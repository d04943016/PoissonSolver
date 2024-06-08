import numpy as np
from abc import ABC, abstractmethod

from typing import Optional, Callable

class UpdaterTemplate(ABC):
    """ UpdaterTemplate is an abstract class for updating the parameters of the model. """
    def __init__(self, size:int):
        """ 
        parameters
        ----------
        size : the size of the vector
        
        """
        self.size = size
        self.reset()

    @abstractmethod
    def reset(self):
        """ Reset the updater """
        raise NotImplementedError

    @abstractmethod
    def update(self, vec:np.ndarray, delta:np.ndarray) -> np.ndarray:
        """ 
        Update the parameters of the model 
        
        usage
        -----
        vec = updater(vec, delta)

        Parameters
        ----------
        vec : the vector to be updated
        delta : the delta of the vector
        
        """
        raise NotImplementedError
    
    def __call__(self, vec:np.ndarray, delta:np.ndarray):
        """ 
        Update the parameters of the model 
        
        usage
        -----
        vec = updater(vec, delta)

        Parameters
        ----------
        vec : the vector to be updated
        delta : the delta of the vector
        
        """
        return self.update(vec, delta)
    
    @property
    def size(self):
        return self.__size
    
    @size.setter
    def size(self, size:int):
        self.__size = size
        self.reset()

class SimpleUpdater(UpdaterTemplate):
    """ Simple Updater """
    def __init__(self, size:int, lr:float=1.0):
        """
        Parameters
        ----------
        size : the size of the vector
        lr : the learning rate

        """
        self.lr = lr
        super().__init__(size)

    def reset(self):
        """ Reset the updater """
        pass

    def update(self, vec:np.ndarray, delta:np.ndarray) -> np.ndarray:
        """ 
        Update the parameters of the model 
        
        usage
        -----
        vec = updater(vec, delta)

        Parameters
        ----------
        vec : the vector to be updated
        delta : the delta of the vector
        
        """
        if vec.shape != delta.shape:
            raise ValueError("shape mismatch")
        return vec + self.lr*delta
    
class AdamUpdater(UpdaterTemplate):
    """ Adam Updater """
    def __init__(self, size:int, lr:float=0.1, beta1:float = 0.9, beta2:float = 0.999, epsilon:float = 1e-8):
        """ 
        parameters
        ----------
        size : the size of the vector
        lr : the learning rate
        beta1 : the decay rate of the first moment
        beta2 : the decay rate of the second moment
        epsilon : the small value to avoid division by zero

        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        super().__init__(size)

    def reset(self):
        """ Reset the updater """
        self.vt    = np.zeros(self.size)
        self.sigma = np.zeros(self.size)
        self.step = 0

    def update(self, vec:np.ndarray, delta:np.ndarray) -> np.ndarray:
        """ 
        Update the parameters of the model 
        
        usage
        -----
        vec = updater(vec, delta)

        Parameters
        ----------
        vec : the vector to be updated
        delta : the delta of the vector
        
        """
        if vec.shape != delta.shape:
            raise ValueError("shape mismatch")
        
        self.step += 1
        self.vt    = self.beta1*self.vt    + (1.0-self.beta1)*delta
        self.sigma = self.beta2*self.sigma + (1.0-self.beta2)*(delta**2)
        
        vt_hat    =    self.vt / (1.0 - self.beta1**self.step)
        sigma_hat = self.sigma / (1.0 - self.beta2**self.step)
        new_delta = vt_hat / (np.sqrt(sigma_hat) + self.epsilon)
        return vec + self.lr*new_delta
    
class AdagradUpdater(UpdaterTemplate):
    """ Adagrad Updater """
    def __init__(self, size:int, lr:float=0.5, epsilon:float = 1e-8):
        """
        Parameters
        ----------
        size : the size of the vector
        lr : the learning rate
        epsilon : the small value to avoid division by zero

        """
        self.lr = lr
        self.epsilon = epsilon
        super().__init__(size)

    def reset(self):
        """ Reset the updater """
        self.sigma = np.zeros(self.size)
        self.step = 0

    def update(self, vec:np.ndarray, delta:np.ndarray) -> np.ndarray:
        """ 
        Update the parameters of the model 
        
        usage
        -----
        vec = updater(vec, delta)

        Parameters
        ----------
        vec : the vector to be updated
        delta : the delta of the vector
        
        """
        if vec.shape != delta.shape:
            raise ValueError("shape mismatch")
        
        self.step += 1
        self.sigma += delta**2
        new_delta = delta / np.sqrt(self.sigma  + self.epsilon)
        return vec + self.lr*new_delta
    

def construct_updater(size, V_updater:Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None):
    """
    construct_updater is a function to construct the updater

    Parameters
    ----------
    size : the size of the vector
    V_updater : the name of the updater or the updater itself
                [default: None] SimpleUpdater

    Returns
    -------
    V_updater : the updater

    """
    if V_updater is None:
        V_updater = SimpleUpdater(size = size)
    elif callable(V_updater):
        V_updater = V_updater
    elif V_updater.upper() in ('SIMPLE', 'SIMPLEUPDATER'):
        V_updater = SimpleUpdater(size = size)
    elif V_updater.upper() in ('ADAM', 'ADAMUPDATER'):
        V_updater = AdamUpdater(size = size)
    elif V_updater.upper() in ('ADAGRAD', 'ADAGRADUPDATER'):
        V_updater = AdagradUpdater(size = size)
    else:
        V_updater = SimpleUpdater(size = size)
    return V_updater


