import copy

class wfpt_state:
    """
    A class that represents a WFPT state vector.

    Parameters
    ----------
    state_template : dict
        Dictionary that defines the state vector. It should have the following structure:
        state_template = {'mirror1': {'dof1': np.array(<shape1>), 'dof2': np.array(<shape2>), ...}, ...}
    """
    def __init__(self, state_template):
        if type(state_template) is not dict:
            raise TypeError('state_template must be a dictionary.')
        self.state = copy.deepcopy(state_template)


    #================= State vector arithmetics ====================
    def __add__(self, other_wfpt_state):
        sum_state = copy.deepcopy(self.state)
        for mirror in sum_state:
            for dof in sum_state[mirror]:
                sum_state[mirror][dof] += other_wfpt_state.state[mirror][dof]
        return wfpt_state(sum_state)

    def __sub__(self, other_wfpt_state):
        sum_state = copy.deepcopy(self.state)
        for mirror in sum_state:
            for dof in sum_state[mirror]:
                sum_state[mirror][dof] -= other_wfpt_state.state[mirror][dof]
        return wfpt_state(sum_state)
    
    def __mul__(self, constant):
        sum_state = copy.deepcopy(self.state)
        for mirror in sum_state:
            for dof in sum_state[mirror]:
                sum_state[mirror][dof] *= constant
        return wfpt_state(sum_state)
    
    def __rmul__(self, constant):
        self.__mul__(constant)

    def __str__(self):
        return str(self.state)

    def __getitem__(self, key):
        return self.state[key]