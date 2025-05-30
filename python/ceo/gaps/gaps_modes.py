import numpy as np
import os
from gaps_utilities import load_dictionary_from_file

class gaps_modes:
    """
    Class encapsulating the GAPS modes used in a simulation.

    Parameters:
    -----------
    m2c_file : str
        Name of file where M2C is stored.
    """
    def __init__(self, m2c_file):
        here = os.path.abspath(os.path.dirname(__file__))
        m2c_fullname = os.path.join(here, 'data', 'M2C', m2c_file)
        self.M2C_DATA = load_dictionary_from_file(m2c_fullname)
        
        #--- number of actuators, number of modes, segment or global?
        sz = self.M2C.shape
        self.n_dofs = sz[0]
        self.n_mode = sz[1]
        self.n_seg = sz[2] if self.M2C.ndim == 3 else 1
    
    
    @property
    def M2C(self):
        return self.M2C_DATA['KLF_M2C']
    
    @property
    def dm_valid_acts(self):
        return self.M2C_DATA['dm_valid_acts']
    
    
    def compute_modal_shapes(self, tel):
        """
        Compute the cube of modal shapes.
        
        Parameters:
        ------------
        tel : telescope_simulator object
            Telescope Simulator object from which to retrieve the influence functions.
        """
        mergedIFmat = tel.get_merged_influence_matrices(self.dm_valid_acts, 
                    get_only_descaled_merged_ifmat = True)

        print("--> Computing modal shapes.....")
        KLFmat = mergedIFmat @ self.M2C.reshape((self.n_dofs,-1), order='F')
        
        mode_wf = np.zeros((tel.pup.nPx,tel.pup.nPx))
        self.KLFcube = np.zeros((tel.pup.nPx,tel.pup.nPx,self.n_mode*self.n_seg))
        
        for jj in range(self.n_mode*self.n_seg):
            mode_wf[tel.pup.GMTmask2D] = KLFmat[:, jj]
            self.KLFcube[:,:,jj] = mode_wf
        
        print("Done! .....")