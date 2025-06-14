from abc import ABC, abstractmethod

class SimBlock(ABC):
    """
    Abstract class that defines the interface of a Simulation Block.
    
    Parameters
    ----------
    T_se : float
        "Short exposure" time of simulation block [in seconds]. Default: TICK_TIME
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
    
    
    def __init__(self, T_se=None, T_out=None, T_d=0.0):
        super().__init__()
        assert SimBlock.TICK_TIME > 0, "Set TICK_TIME before defining Simulation Blocks!"
        
        if T_se == None:
            self.T_se  = SimBlock.TICK_TIME
        else:
            if T_se < SimBlock.TICK_TIME:
                raise Exception("T_se cannot be smaller than TICK_TIME.")
            self.T_se = T_se
        if T_out == None:
            self.T_out = SimBlock.TICK_TIME
        else:
            if T_out < SimBlock.TICK_TIME:
                raise Exception("T_out cannot be smaller than TICK_TIME.")
            self.T_out = T_out
        self.T_d = T_d
        
        self._se_counter = 0
        self._se_niter = int(self.T_se / SimBlock.TICK_TIME)
        
        self._integration_counter = 0
        self._integration_niter = int(self.T_out / SimBlock.TICK_TIME)
        
        self.telemetry_data = {}
    
    
    def trigger(self):
        if SimBlock.CURRENT_TIME >= self.T_d:
            self._integrate()
            self._integration_counter += 1
            if self._integration_counter == self._integration_niter:
                self._compute_output()
                self._integration_counter = 0
    
    
    def _integrate(self):
        pass
    
    @abstractmethod
    def _compute_output(self):
        pass
    
    @abstractmethod
    def register_input_method(self, *args, **kwargs):
        pass
    
    def reset_telemetry(self):
        """
        Resets telemetry buffers.
        """
        for key in self.telemetry_data.keys():
            self.telemetry_data[key] = []