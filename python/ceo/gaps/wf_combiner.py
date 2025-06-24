import numpy as np
import cupy as cp
from SimBlock import SimBlock

class wf_combiner(SimBlock):
    """
    Combines wavefronts to produce the residual wavefront seen by the wavefront sensors.
    
    Parameters:
    ------------
    pup : GMT pupil object
    
    mergedIFmat : numpy array
        GAPS Merged Influence Functions Matrix (DM + PTT).
        
    Notes:
    ------
    This class inherits timing properties from the abstract class "SimBlock".
    """
    def __init__(self, pup, mergedIFmat):
        
        #------ SimBlock timing parameters
        super().__init__()
        
        #------ Properties
        self._mergedIFmat = cp.array(mergedIFmat)
        self._nvacts = mergedIFmat.shape[1]-21 #number of valid DM actuators
        self._pup = pup
        
        #------ Buffers        
        self.ptt_offset = np.zeros(21)
        self.dm_offset = np.zeros(self._nvacts)
        self.output_wavefront = np.zeros((pup.nPx,pup.nPx))
        
        #------ Telemetry buffers
        self.telemetry_data['wfe'] = []
        self.telemetry_data['seg_wfe'] = []
        self.telemetry_data['spp'] = []
        self.telemetry_data['time_vec'] = []
    
    
    def register_input_method(self, get_ptt_comm, get_dm_comm, get_atmo_wf=None):
        """
        Registers the external method used to retrieve the mirror commands, and disturbance wavefronts.
        
        Parameters:
        -----------
        get_ptt_comm : callable
            External method that provides the PTT array command (x21).
        
        get_dm_comm : callable
            External method that provides the DM command (x nvacts).
        """
        assert callable(get_ptt_comm), "'get_ptt_comm' must be a callable function."
        assert callable(get_ptt_comm), "'get_dm_comm' must be a callable function."
        if get_atmo_wf is not None:
            assert callable(get_atmo_wf), "'get_atmo_wf' must be a callable function."
        
        self.__get_ptt_comm = get_ptt_comm
        self.__get_dm_comm = get_dm_comm
        self.__get_atmo_wf = get_atmo_wf
    
    
    def _compute_output(self):
        """
        Computes the combined (residual) wavefront that the WFS will measure.
        NOTE: This internal function is called by the trigger() method defined in the SimBlock parent class.
        """        
        #--> add DM+PTT WF
        ptt_comm = self.__get_ptt_comm() + self.ptt_offset
        dm_comm = self.__get_dm_comm() + self.dm_offset
        tel_comm = np.hstack((ptt_comm, dm_comm))
        self.output_wavefront[self._pup.GMTmask2D] = (self._mergedIFmat @ cp.array(tel_comm)).get()
        
        #--> add turbulence WF
        if callable(self.__get_atmo_wf):
            self.output_wavefront += self.__get_atmo_wf()
        
        self._updateTelemetry()
    
    
    def _updateTelemetry(self):
        """
        Updates telemetry buffer.
        Note: Internal method called by _compute_output()
        """
        self.telemetry_data['wfe'] += [self.get_wfe()]
        self.telemetry_data['seg_wfe'] += [self.get_segment_wfe()]
        self.telemetry_data['spp'] += [self.get_segment_phase_piston()]
        self.telemetry_data['time_vec'] += [SimBlock.CURRENT_TIME]
    
    
    def get_wavefront(self):
        """
        Get the output (residual) wavefront.
        """
        return self.output_wavefront
    
    
    def get_wfe(self):
        """
        Get the WFE.
        """
        return np.sqrt(np.sum(self.output_wavefront**2) / self._pup.nmask)
    
    
    def get_segment_wfe(self):
        """
        Get the WFE over each GMT segment.
        """
        seg_wfe = np.zeros(7)
        for segId in range(7):
            #seg_wfe[segId] = np.sqrt(self.output_wavefront[self._pup.P[segId,:]]**2 / self._pup.npseg[segId])
            seg_wfe[segId] = np.std(self.output_wavefront.ravel()[self._pup.P[segId,:]])
        return seg_wfe
    
    
    def get_segment_phase_piston(self):
        """
        Get segment phase piston.
        """
        spp = np.zeros(7)
        for segId in range(7):
            spp[segId] = np.sum(self.output_wavefront.ravel()[self._pup.P[segId,:]]) / self._pup.npseg[segId]
        return spp
        
    
    def set_scramble(self, do_piston=False, piston_rms=0.0,
                         do_tiptilt=False, tiptilt_rms=0.0,
                         do_dm_acts=False, dm_acts_rms=0.0):
                         #do_modes=False, modes_rms=0.0, modal_scaling=True):
        """
        Generate an initial offset (a.k.a scramble) of active mirrors.
        
        Parameters:
        -----------
        do_piston : bool
            If True, introduce a random segment piston initial offset. Default: False
        
        piston_rms : float
            segment piston RMS of random initial offset [m WF RMS].
        
        do_tiptilt : bool
            If True, introduce a random segment tip-tilt initial offset. Default: False
        
        tiptilt_rms : float
            segment tip-tilt RMS of random initial offset [m WF RMS].

        do_dm_acts : bool
            If True, introduce a random zonal DM initial offset. Default: False
        
        dm_acts_rms : float
            wavefront RMS of random initial offset [m WF RMS].
        """
        if do_piston == True:
            pistscramble  = np.random.normal(loc=0.0, scale=1, size=7)
            pistscramble *= piston_rms / np.std(pistscramble)
            pistscramble -= np.mean(pistscramble)
            self.ptt_offset[0:7] += pistscramble
            
        if do_tiptilt == True:
            TTscramble = np.random.normal(loc=0.0, scale=1, size=14)
            TTscramble *= tiptilt_rms / np.std(TTscramble)
            self.ptt_offset[7:] += TTscramble
        
        if do_dm_acts == True:
            DMzonalScramble = np.random.normal(loc=0.0, scale=1, size=self._nvacts)
            DMzonalScramble *= dm_acts_rms / np.std(DMzonalScramble)
            DMzonalScramble -= np.mean(DMzonalScramble)
            self.dm_offset += DMzonalScramble
    
    
    def reset(self):
        """
        Resets the WF buffer.
        """
        self.output_wavefront *= 0
    
    
    def reset_offset(self):
        """
        Resets the Offset buffers.
        """
        self.ptt_offset *= 0
        self.dm_offset *= 0