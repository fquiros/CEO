from ceo.wfpt import wfpt_state, wfpt_utilities
import numpy as np
import os

class wfpt_modes:
    """
    A class that represents a modal basis produced by WFPT M1 and M2 mirrors.
    
    Parameters
    ----------
    M1_m2c_file : string
        Name of .npz file containing the modes-to-commands matrix for M1.
    M2_m2c_file : string
        idem for M2
        Note: these files should be stored in the dedicated WFPT_model_data/M2C repository..
    """
    def __init__(self, M1_m2c_file, M2_m2c_file):
        #--- Load M2C file
        here = os.path.abspath(os.path.dirname(__file__))
        M1_m2c_fullname = os.path.join(here, 'WFPT_model_data', 'M2C', M1_m2c_file)
        self.M1_M2C_data = wfpt_utilities.load_dictionary_from_file(M1_m2c_fullname)
        M2_m2c_fullname = os.path.join(here, 'WFPT_model_data', 'M2C', M2_m2c_file)
        self.M2_M2C_data = wfpt_utilities.load_dictionary_from_file(M2_m2c_fullname)
        
        #--- number of modes
        self._M1_nmodes = self.M1_M2C.shape[1]
        self._M2_nmodes = self.M2_M2C.shape[1]
        
        #--- Create modal states
        self._M1_coeffs = np.zeros((7, self._M1_nmodes))
        self._M2_coeffs = np.zeros((7, self._M2_nmodes))

    
    def set_zonal_state(self, wfpt_state):
        """
        Pass on an empty WFPT state vector to stored internally the structure.
        """
        self._wfpt_state = wfpt_state
    
    @property
    def M1_M2C(self):
        return self.M1_M2C_data['M2C']

    @property
    def M2_M2C(self):
        return self.M2_M2C_data['M2C']
    
    @property
    def M1_nmodes(self):
        return self._M1_nmodes
    
    @property
    def M2_nmodes(self):
        return self._M2_nmodes

    @property
    def modal_state(self):
        d = {'M1': {'modes': self._M1_coeffs}, 
             'M2': {'modes': self._M2_coeffs}
            }
        return wfpt_state(d)
    
    @property
    def zonal_state(self):
        return self._wfpt_state
    
    def reset(self):
        self._M1_coeffs *= 0
        self._M2_coeffs *= 0
        
    
    def update(self, state):
        """
        Updates the WFPT zonal state from an input modal state.
        """
        self._M1_coeffs = np.copy(state['M1']['modes'])
        M1_zonal_commands = np.zeros(self.M1_M2C.shape[0])
        for segid in range(7):
            M1_zonal_commands += self.M1_M2C[:,:,segid] @ self._M1_coeffs[segid,:]
        pistvec, ttvec, dmvec = np.split(M1_zonal_commands, [7, 7+14])
        self._wfpt_state['M1_PTT']['segment piston'][:] = pistvec
        self._wfpt_state['M1_PTT']['segment tip-tilt'][:] = ttvec.reshape((7,2), order='F')
        self._wfpt_state['M1_DM']['actuators'][:] = dmvec
        
        self._M2_coeffs = np.copy(state['M2']['modes'])
        M2_zonal_commands = np.zeros(self.M2_M2C.shape[0])
        for segid in range(7):
            M2_zonal_commands += self.M2_M2C[:,:,segid] @ self._M2_coeffs[segid,:]
        pistvec, ttvec, dmvec = np.split(M2_zonal_commands, [7, 7+14])
        self._wfpt_state['M2_PTT']['segment piston'][:] = pistvec
        self._wfpt_state['M2_PTT']['segment tip-tilt'][:] = ttvec.reshape((7,2), order='F')
        self._wfpt_state['M2_DM']['actuators'][:] = dmvec
    