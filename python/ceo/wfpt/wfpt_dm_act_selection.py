import numpy as np
import os
import errno

class wfpt_dm_act_selection:
    """
    A class that assists in the selection of DM valid actuators.

    Parameters:
    -----------
        dm_response_file : string
            file to load actuator peak values.
        dm_response_type : string
            Either "IFpeaks" (for influence function wavefront peak values) or "SHmeas" for zonal SH slopes.
    """
    def __init__(self, dm_response_file, dm_response_type):
        assert dm_response_type in ['IFpeaks', 'SHmeas'], \
                '"dm_response_type" must be either "IFpeaks" or "SHmeas"'
        if dm_response_type == 'IFpeaks':
            if os.path.isfile(dm_response_file):
                print("Restoring IF peaks from file %s"%dm_response_file)
                with np.load(dm_response_file) as data:
                    self.IFpeaks = data['IF_peak']
                    self.nact = len(self.IFpeaks)
            else:
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), dm_response_file)
        elif dm_response_type == 'SHmeas':
            #--- TO DO: load a zonal IntMat, and use slopes amplitudes as IFpeaks.
            raise Exception("SHmeas type not implemented yet!!")

        self._dm_response_file = dm_response_file
        self._dm_response_type = dm_response_type
        
        self._valid_acts = np.ones(self.nact, dtype='bool')
        self.act_thr = 0.0
        self.slaveMat = np.diag(self._valid_acts).astype('int')
        self.act_masters = []
        self.act_slaves = []


    def selection_with_threshold(self, thr):
        """
        Select actuators with response over the threshold.

        Parameters:
        ----------
            thr : float
                relative threshold value [0 <= thr <= 1].
                Note: the absolute threshold value is computed as thr*max(response).
        """
        if thr < 0 or thr > 1:
            raise ValueError('"thr" value must be 0 <= thr <= 1')
        self.act_thr = thr
        max_peak = np.max(self.IFpeaks)
        self._valid_acts = self.IFpeaks > thr * max_peak
        print("With a threshold of %0.3f you get %d valid actuators."%(thr, self.n_valid_acts))
        self.slaveMat = np.diag(self._valid_acts).astype('int')
        self.act_masters = []
        self.act_slaves = []


    def selection_with_actuator_index(self, remove_act_list=[], add_act_list=[]):
        """
        Add or remove specific actuators using their indexes.

        Parameters:
        -----------
            remove_act_list : list
                index of actuators to remove from selection
            add_act_list : list
                index of actuators to add to selection
        """
        if not type(remove_act_list) is list or not type(add_act_list) is list:
            raise TypeError("'remove_act_list' and 'add_act_list' must be of'list' type.")
        
        self._valid_acts[remove_act_list] = False
        self._valid_acts[add_act_list] = True
        print('Number of valid actuators: %d'%self.n_valid_acts)
        self.slaveMat[:, remove_act_list] = 0
        self.slaveMat[add_act_list, add_act_list] = 1
        
        #--- Check whether removed acts were masters:
        for act in remove_act_list:
            if act in self.act_masters:
                print("removing act #%d from masters list"%act)
                idx = self.act_masters.index(act) 
                del self.act_masters[idx]
                del self.act_slaves[idx]
        
        #--- Check whether added acts were slaves:
        for act in add_act_list:
            if act in self.act_slaves:
                print("removing act #%d from slaves list"%act)
                idx = self.act_slaves.index(act)
                del self.act_masters[idx]
                del self.act_slaves[idx]
        

    @property
    def n_valid_acts(self):
        return np.sum(self._valid_acts)


    @property
    def valid_acts(self):
        return self._valid_acts


    def slaving_selection(self, act_masters=[], act_slaves=[]):
        """
        Slave a set of actuators to a corresponding set of master actuators.

        Parameters:
        -----------
            act_masters : list
                List of master actuators
            act_slaves : list
                List of slaves actuators
            Note: the number of masters and slaves must be the same (one to one correspondence).
        """
        if len(act_masters) != len(act_slaves):
            raise ValueError("The number of master and slave actuators must be equal.")

        #------ Check that master actuators are in valid actuator list
        valid_master_set = True
        for act in act_masters:
            if self._valid_acts[act] == False:
                print("Master actuator #%d not in valid actuator set. Add it first using 'selection_with_actuator_index()'"%act)
                valid_master_set = False
        if not valid_master_set:
            return

        #------ Check that slave actuators are not in valid actuator list
        valid_slave_set = True
        for act in act_slaves:
            if self._valid_acts[act] == True:
                print("Slave actuator #%d is in valid actuator set. Remove it first using 'selection_with_actuator_index()'"%act)
                valid_slave_set = False
        if not valid_slave_set:
            return
        
        #------ Update slaving matrix
        self.act_masters = act_masters
        self.act_slaves = act_slaves
        self.slaveMat = np.diag(self._valid_acts).astype('int')
        self.slaveMat[self.act_slaves, self.act_masters] = 1


    def save_valid_actuators_file(self, fname):
        """
        Save valid actuator data to .npz file.
        Note: the file is saved to dedicated repository in WFPT_model_data/dm_valid_actuators/
        
        Parameters:
        -----------
        fname : str
            Name of file.
        
        """
        here = os.path.abspath(os.path.dirname(__file__))
        fullname = os.path.join(here, 'WFPT_model_data', 'dm_valid_actuators', fname)        

        
        tosave = dict(valid_acts=self.valid_acts, n_valid_acts=self.n_valid_acts, thr=self.act_thr,
                      act_masters=self.act_masters, act_slaves=self.act_slaves, slaveMat=self.slaveMat,
                      dm_response_file=self._dm_response_file, dm_response_type=self._dm_response_type)
        
        np.savez(fullname, **tosave)
        print("Saving to file %s"%fullname)
        
                      
                      