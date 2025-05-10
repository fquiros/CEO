from abc import ABC, abstractmethod

class SimBlock(ABC):
    """
    Abstract class that defines the interface of a Simulation Block.
    
    Parameters
    ----------
    T_in : float
        Sampling time of simulation block [in seconds]. Default: TICK_TIME
    T_out : float
        Integration time of simulation block [in seconds]. Default: TICK_TIME
    T_d : float
        Time delay for the simulation block to start operation [in seconds]. Default: 0.0
    """ 
    TICK_TIME = 0.0
    CURRENT_TIME = 0.0

    @classmethod
    def current_iteration(cls):
        """
        Returns the current closed-loop iteration number
        """
        return int(cls.CURRENT_TIME / cls.TICK_TIME)
    
    
    def __init__(self, T_in=None, T_out=None, T_d=0.0):
        super().__init__()        
        if T_in == None:
            self.T_in  = SimBlock.TICK_TIME
        else:
            if T_in < SimBlock.TICK_TIME:
                raise Exception("T_in cannot be smaller than TICK_TIME.")
            self.T_in = T_in
        if T_out == None:
            self.T_out = SimBlock.TICK_TIME
        else:
            self.T_out = T_out
        self.T_d = T_d
        
        self._integration_counter = 0
        self._integration_niter = int(self.T_out / SimBlock.TICK_TIME)
    
    
    @abstractmethod
    def trigger(self):
        pass