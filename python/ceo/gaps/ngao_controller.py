import numpy as np
import cupy as cp
import sys
from SimBlock import SimBlock
from gaps_modes import gaps_modes

class hdfs_controller(SimBlock):
    """
    Simulates segment piston reconstruction and control based on HDFS measurements.
    
    Parameters:
    ------------
    RecMat : numpy array
        HDFS segment piston reconstruction matrix.
    
    operation_mode : str
        Either 'bootstrap' or 'baseline'.
    
    int_gain : float
        Integrator's gain (when operating in bootstrap mode). Default: 0.5
    
    Psig2spp : numpy vector (12 elements)
        Scaling factor to convert HDFS signal vector to differential segment phase piston between segment pairs (i.e. pairs defined by the HDFS mask).
    
    intmat : numpy array
        HDFS segment piston interaction matrix. (Needed if RecMat needs to be recomputed).
    
    global_pist_reg_factor : float
        global piston penalizing factor for the HDFS reconstructor (Needed if RecMat needs to be recomputed).
    
    eject_thr : float
        Segment ejection detection threshold [in meters] (when operating in baseline mode). Default: 380 nm
    
    capture_thr : float
        Capture range of HDFS [in meters]. Default: 10 um
    
    Parameters passed on to parent class SimBlock:
    ----------------------------------------------
    T_out : float
        Integration time of simulation block [in seconds]. Default: TICK_TIME
        
    T_d : float
        Time delay for the simulation block to start operation [in seconds]. Default: 0.0
    """    
    def __init__(self, RecMat, operation_mode='bootstrap', int_gain=0.5,
                 Psig2spp=None, intmat=None, global_pist_reg_factor=None,
                 eject_thr=380e-9, capture_thr=10e-6,
                 T_out=None, T_d=0.0):
        
        #----- SimBlock timing parameters
        super().__init__(T_out=T_out, T_d=T_d)
        
        #----- Properties
        self.__R = RecMat
        self._Psig2spp = Psig2spp
        self._intmat = intmat
        self._global_pist_reg_factor = global_pist_reg_factor
        
        self.operation_mode = operation_mode
        if self.operation_mode == 'bootstrap':
            self.g_i = int_gain
        elif self.operation_mode == 'baseline':
            self.eject_thr = eject_thr
            self.capture_thr = capture_thr
        
        #------ Buffers
        self.hdfs_command = np.zeros(7)
        
        #------ Telemetry buffers
        self.telemetry_data['hdfs_ctrl_command'] = []
        self.telemetry_data['hdfs_ctrl_time_vec'] = []
        self.telemetry_data['hdfs_meas_quality'] = []
    
    
    def register_input_method(self, hdfs_get_measurement):
        """
        Registers the external method used to retrieve the HDFS measurement.
        
        Parameters:
        -----------
        hdfs_get_measurement : callable
            External method that provides a 14-element HDFS measurement vector.
        """
        assert callable(hdfs_get_measurement), "'hdfs_get_measurement' must be a callable function."        
        self.__meas = hdfs_get_measurement
    
    
    def _integrate(self):
        if self._integration_counter == 0 and self.operation_mode == 'baseline':
            self.reset()
    
    
    def _compute_output(self):
        """
        This function multiplies the HDFS measurement by the HDFS reconstructor and delivers the HDFS command.
        In "bootstrap" operation mode, it delivers an integrated HDFS command.
        In "baseline operation mode", it delivers a thresholded command to recover ejected segments.
        NOTE: This internal function is called by the trigger() method defined in the SimBlock parent class.
        """
        hdfs_meas = self.__meas()
        
        #--> Deal with possible pair of blurred fringes
        self._quality_control(hdfs_meas)
        
        if self.hdfs_meas_quality == 1.0:
            PISTvec = self.__R @ hdfs_meas
        elif self.hdfs_meas_quality == 0.5:
            PISTvec = self.__R_adhoc @ hdfs_meas
        else:
            PISTvec = np.zeros(7)
            
        if self.operation_mode == 'bootstrap':
            self.hdfs_command = self.hdfs_command - self.g_i * PISTvec
        elif self.operation_mode == 'baseline':
            if np.logical_and(np.abs(PISTvec) > self.eject_thr, \
                              np.abs(PISTvec) < self.capture_thr).any():
                self.hdfs_command = -PISTvec
        
        self._updateTelemetry()       
    

    def _quality_control(self, hdfs_meas):
        """
        Identify blurred fringes, sets the "hdfs_meas_quality" metric:
            1.0: good measurement
            0.0: bad measurement (HDFS vector discarded)
            0.5: partially usable measurement. Ad-hoc HDFS reconstructor provided.
        """
        hdfs_meas_nm = hdfs_meas * self._Psig2spp * 1e9 #in nm
        meas1 = hdfs_meas_nm[0:7]
        meas2 = hdfs_meas_nm[7:]
        QualityCheck = np.abs(meas1 + meas2) / np.sqrt(2)
        
        if np.all(QualityCheck < self.eject_thr*1e9):
            self.hdfs_meas_quality = 1.0
        else:
            idx_bad_meas, = np.where(QualityCheck >= self.eject_thr*1e9)
            idx_bad_meas = np.concatenate((idx_bad_meas, idx_bad_meas+7)) #pair of HDFS SAs
            sys.stdout.write(('\nBad HDFS meas pairs: '+np.array_str(idx_bad_meas)))
            
            if idx_bad_meas.size > 2:
                self.hdfs_meas_quality = 0.0
                #sys.stdout.write('\nHDFS Measurement DISCARDED.\n')
            else:
                self.hdfs_meas_quality = 0.5
                #sys.stdout.write("\nHDFS Measurement PARTIALLY USED.\n")
                self._recompute_hdfs_reconstructor(idx_bad_meas)


    def _recompute_hdfs_reconstructor(self, idx_bad_meas):
        """
        Recompute HDFS reconstructor weighing out bad measurements.

        Parameters:
        -----------
        idx_bad_meas : numpy array
            Index of bad measurements
        """
        cnninv0 = np.diag(np.ones(14))
        cnninv0[idx_bad_meas,idx_bad_meas] = 0 # remove the bad measurement(s)
        HDFS_reg_mat = self._intmat.T @ cnninv0 @ self._intmat + \
                       self._global_pist_reg_factor * np.ones((7,7))
        self.__R_adhoc = np.linalg.solve(HDFS_reg_mat, self._intmat.T @ cnninv0)
    
    
    def _updateTelemetry(self):
        """
        Updates telemetry buffer.
        Note: Internal method called by _compute_output()
        """
        self.telemetry_data['hdfs_ctrl_command'] += [self.get_piston_command()]
        self.telemetry_data['hdfs_ctrl_time_vec'] += [SimBlock.CURRENT_TIME + SimBlock.TICK_TIME]
        self.telemetry_data['hdfs_meas_quality'] += [self.hdfs_meas_quality]
    
        
    def get_ptt_command(self):
        """
        Get the PTT command from the HDFS controller.
        Note: the output is delivered as a full 21-element PTT command.
        """
        ptt_command = np.zeros(21)
        ptt_command[0::3] = self.get_piston_command()
        return ptt_command
    
    
    def get_piston_command(self):
        """
        Get the segment piston command from the HDFS controller.
        Note: only a 7-element vector is delivered.
        """
        return np.copy(self.hdfs_command)
    
    
    def reset(self):
        """
        Resets the HDFS command buffer.
        """
        self.hdfs_command *= 0


class ao_controller(SimBlock):
    """
    Simulates AO zonal or modal control based on PWFS measurements.
    
    Parameters:
    ------------
    RecMat : numpy array
        WF Reconstruction matrix. Can be either zonal or modal.
    
    int_gain : float
        Integrator's gain. Default: 0.5
    
    forget_factor : float
        Controller's forgetting factor. Default: 1.0
    
    pure_delay : int
        Pure (discrete) delay, in number of frames. Default: 1
    
    modal_control : bool
        If True, modal control will be simulated. Default: False
    
    modes_obj : 'gaps_modes' object
        If modal_control selected, modes_obj defines the modal basis.
    
    Pp2m : numpy array
        segment piston to modes projection matrix.
    
    Parameters passed on to parent class SimBlock:
    ----------------------------------------------
    T_out : float
        Integration time of simulation block [in seconds]. Default: TICK_TIME
        
    T_d : float
        Time delay for the simulation block to start operation [in seconds]. Default: 0.0
    """    
    def __init__(self, RecMat, int_gain=0.5, forget_factor=1.0, pure_delay=1,
                 modal_control=False, modes_obj=None, Pp2m=None,
                 T_out=None, T_d=0.0):
        
        #----- SimBlock timing parameters
        super().__init__(T_out=T_out, T_d=T_d)
        
        #----- Properties
        self.__modal_control = modal_control
        self.__R = cp.array(RecMat)
        self.g_i = int_gain
        self.g_f = forget_factor
        
        n_dof = RecMat.shape[0]        
        
        #------ Modal Control setup
        if modal_control == True:
            assert modes_obj is not None, "GAPS 'modes_obj' needs to be introduced!..."
            try:
                M2C = modes_obj.M2Cmat
            except:
                raise ValueError("'modes_obj' is not a 'gaps_modes' object...")
            assert M2C.shape[1] == n_dof, "'M2C' and 'RecMat' have incompatible dimensions."
            self.__M2C = M2C
            nvacts = len(modes_obj.dm_valid_acts)
        else:
            nvacts = n_dof
        
        #------ Segment piston to control modes (zonal or modal) projection matrix
        if Pp2m is not None:
            assert Pp2m.shape[0] == n_dof, "'Pp2m' and 'RecMat' have incompatible dimensions."
            self.__Pp2m = Pp2m
        
        #------ Command buffers
        self.ao_integr_command = np.zeros(n_dof) # <-- modal or zonal integrated command       
        self.ptt_command = np.zeros((    21, pure_delay+1))
        self.dm_command  = np.zeros((nvacts, pure_delay+1))
        
        #------ Telemetry buffers
        self.telemetry_data['ao_delta_command'] = []
        self.telemetry_data['ao_integr_command'] = []
        #self.telemetry_data['ao_ctrl_time_vec'] = []
    
    
    @property
    def modal_control(self):
        return self.__modal_control
    
    
    def register_input_method(self, pwfs_get_measurement, hdfs_get_command=None):
        """
        Registers the external method used to retrieve the PWFS measurement.
        
        Parameters:
        -----------
        pwfs_get_measurement : callable
            External method that provides a PWFS measurement vector.
        
        hdfs_get_command : callable
            External method that provides HDFS segment ejection recovery command.
        """
        assert callable(pwfs_get_measurement), "'pwfs_get_measurement' must be a callable function."        
        self.__meas = pwfs_get_measurement
        
        if hdfs_get_command is not None:
            assert callable(hdfs_get_command), "'hdfs_get_command' must be a callable function."
        self.__get_hdfs_command = hdfs_get_command
    
    
    def _compute_output(self):
        """
        This function multiplies the PWFS measurement by the zonal or modal AO reconstructor and delivers the zonal command.
        NOTE: This internal function is called by the trigger() method defined in the SimBlock parent class.
        """
        
        #-- Simple integral control
        self.ao_delta_command = (self.__R @ cp.array(self.__meas())).get()
        self.ao_integr_command = self.g_f * self.ao_integr_command - self.g_i * self.ao_delta_command
        
        #-- Update segment piston coefficients with HDFS command (equivalent to IIR state vector update)
        if callable(self.__get_hdfs_command):
            hdfs_command = self.__get_hdfs_command()
            if np.sum(np.abs(hdfs_command)) > 0:
                #sys.stdout.write("\nHDFS recovering segment piston!\n")
                self.ao_integr_command += self.__Pp2m @ hdfs_command
        
        #-- Update delay buffer
        self.ptt_command = np.roll(self.ptt_command, 1, axis=1)
        self.dm_command  = np.roll(self.dm_command,  1, axis=1)
        
        #-- Update PTT and DM commands
        if self.__modal_control == True:
            ao_command = self.__M2C @ self.ao_integr_command
            self.ptt_command[:,0], self.dm_command[:,0] = np.split(ao_command, [21])
        else:
            self.dm_command[:,0] = self.ao_integr_command[:]
        
        self._updateTelemetry()
    
    
    def _updateTelemetry(self):
        """
        Updates telemetry buffer.
        Note: Internal method called by _compute_output()
        """
        self.telemetry_data['ao_delta_command'] += [self.ao_delta_command]
        self.telemetry_data['ao_integr_command'] += [self.ao_integr_command]        
        #self.telemetry_data['ao_ctrl_time_vec'] += [SimBlock.CURRENT_TIME + SimBlock.TICK_TIME]
    
    
    def get_dm_command(self):
        """
        Get the DM command from the AO controller.
        """
        return self.dm_command[:,-1]
    
    
    def get_ptt_command(self):
        """
        Get the PTT command from the AO controller.
        """
        return self.ptt_command[:,-1]
    
    
    def reset(self):
        """
        Resets the AO command buffer.
        """
        self.ptt_command *= 0
        self.dm_command *= 0
        self.ao_integr_command *= 0


#//////////////////////////// ADDITIONAL FUNCTIONS ////////////////////////////


def interaction_matrix(wfs, phase_cube, act_list, amp_wf):
    """
    Acquire the interaction matrix between a WFS and a DEVICE (DM, PTT, or MODES).
    
    Parameters:
    -----------
    wfs : wfs object (pwf_model or hdfs_model)
    
    phase_cube : array (nPx x nPx x n_dofs)
        Cube containing influence functions or modal shapes of the device being calibrated.
    
    act_list : list
        Index vector of influence functions or modal shapes to take into consideration.
    
    amp_wf : float OR vector
        Command amplitude to apply to Device during calibration [in meters WF].
        NOTE: If amp_wf is a vector, it must have same lenght as 'act_list' parameter.
    """
    def pushpull(act_idx, amp):
        def measure(stroke_sign):
            IF_poke = phase_cube[:,:,act_idx] * amp * stroke_sign
            wfs.reset()
            meas = wfs.measure(IF_poke)
            #print("max abs value: %2.3f"%np.max(np.abs(wfs.get_measurement())))
            return meas
        s_push = measure(+1)
        s_pull = measure(-1)
        return 0.5 * (s_push-s_pull) / amp

    sys.stdout.write("Starting calibration of interaction matrix\n") 
    intmat = np.zeros((wfs.get_measurement_size(), len(act_list)))
    
    if isinstance(amp_wf, float):
        amp_wf = np.full(len(act_list), amp_wf)
    
    for idx, this_act in enumerate(act_list):
        sys.stdout.write("%d "%this_act)
        intmat[:,idx] = np.ravel( pushpull(this_act, amp_wf[idx]) )
    sys.stdout.write("\n")
    return intmat