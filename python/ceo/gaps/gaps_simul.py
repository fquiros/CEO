import numpy as np
import sys
import os
from abc import abstractmethod

from ceo import StopWatch
from SimBlock import SimBlock
from ngwsp_model import pwfs_model, hdfs_model 
from telescope_simulator import telescope_simulator
from atmo_disturbance import atmo_disturbance
from gaps_visulib import show_live_loop

class gaps_simul:
    """
    Class representing a GAPS wavefront control and calibration test.
    
    Parameters:
    -----------
    project_truss_onaxis : bool
        If True, truss shadow will be applied to central pupil mask. Default: False
    
    mag : float
        NGS R magnitude. Default: 10
    
    tick_time : float
        Tick time of the simulation [s]. Default: 1 ms
    
    pyr_Ts : float
        Integration time of the NGWS-P PWFS model [s]. Default: 1 ms
    
    pyr_Td : float
        Time delay for the NGWS-P PWFS to start operation [s]. Default: 0 s.
    
    hdfs_Ts : float
        Integration time of the NGWS-P HDFS model [s]. Default: 150 ms
    
    hdfs_Td : float
        Time delay for the NGWS-P HDFS to start operation [s]. Default: 50 ms
    
    pyr_modulation : float
        modulation radius in lambda/D units. Default: 2.0
    
    pyr_RON : float
        Readout noise in e- RMS. Default: 0.5
    
    pyr_emccd_gain : int
        EMCCD gain. Default: 600
    
    pyr_ADU_gain : float
        ADU gain. Default: 1/30
    
    hdfs_fs_shape : string
        Type of field stop: "square", "round", "none". Default: "round"
    
    hdfs_fs_dim_mas : float
        size/diameter of field stop [in mas]. Default: 50
    
    hdfs_RON : float
        Readout noise in e- RMS. Default: 0.5
    
    hdfs_emccd_gain : int
        EMCCD gain. Default: 100
    
    hdfs_ADU_gain : float
        ADU gain. Default: 1/2.76 
        Note: ProEM when readout speed set to 30 MHz (see HDFS design report, GMT-DOC-05337, Figure 9-7).
    
    hdfs_darkCurrent : float
        EMCCD dark current. Default: 0.    
    
    mems_ref_ifunc_fname : string
        npz file containing:
            1) the reference influence function : (nPx,nPx) array.
            2) the actuator pitch value in pixels
            3) the location of the IF peak in pixels
    
     mems_grid_rot_deg : float
        DM grid clocking w.r.t. GMT pupil [deg]. Default: -2.5 deg
    
    turb_type : str
        Type of turbulence simulation. Options for closed-loop tests are:
            1. "LCO typical-typical"
                Simulates the 7-layer turbulence profile.
            2. "Single Layer" (Default)
                Simulates a single turbulence layer shifting according to the Taylor hypothesis.
    
    r0 : float
        Fried parameter [in meters @ 500 nm]. Default: 16.0 cm
    
    L0 : float
        Outer scale [in meters]. Default: 25.0 m
    
    single_layer_wind_speed : float
        Wind speed in m/s when selecting "Single Layer" type. Default: 13.5 m/s
    
    single_layer_wind_direction : float
        Direction of wind flow in degrees when selecting "Single Layer" type. Default: 0 deg (i.e. from left to right).
    
    project_to_mirror_space : bool
        If True, turbulence phase screens will be projected to DM space. Default: True
    """
    def __init__(self, project_truss_onaxis=False, mag=10.0,
                 tick_time=1e-3, pyr_Ts=1e-3, pyr_Td=0.0, hdfs_Ts=150e-3, hdfs_Td=50e-3,
                 pyr_modulation=2.0, pyr_RON=0.5, pyr_emccd_gain=600, pyr_ADU_gain=1/30.,
                 hdfs_fs_shape = 'round', hdfs_fs_dim_mas = 50, hdfs_RON = 0.5,
                 hdfs_emccd_gain = 100, hdfs_ADU_gain = 1/2.76, hdfs_darkCurrent = 0.,
                 mems_ref_ifunc_fname = 'MagAO-X_MEMS2k_ReferenceIF.npz',
                 mems_grid_rot_deg = -2.5,
                 turb_type='Single Layer', L0=25.0, r0=16.0e-2,
                 single_layer_wind_speed = 13.5, single_layer_wind_direction=0.0,
                 project_to_mirror_space=True):

        self.simul_params = {}
        
        #-- Pupil sampling and orientation as seen by the NGWS-P.
        #-- Note: These are not free parameters for the GAPS simulator.
        array_size_pix=460
        array_size_m=25.5
        array_rot_angle=15.0
        self.simul_params['array_params'] = {'array_size_pix': array_size_pix,
                                            'array_size_m': array_size_m,
                                            'array_rot_angle': array_rot_angle}
        
        self.project_truss_onaxis = project_truss_onaxis
        
        #---------------------- Set simulation timing parameters -------------------------
        self.tickTime = tick_time
        self.currentTime = 0.0
        self._tid = StopWatch()
        
        #-------------------------------- Initialize PWFS --------------------------------
        self.pwfs = pwfs_model(array_size_pix = array_size_pix, 
                               array_size_m = array_size_m, 
                               array_rot_angle = array_rot_angle, 
                               mag = mag,
                               pyr_modulation = pyr_modulation,
                               pyr_RON = pyr_RON,
                               pyr_emccd_gain = pyr_emccd_gain,
                               pyr_ADU_gain = pyr_ADU_gain,
                               T_out = pyr_Ts,
                               T_d = pyr_Td)
        
        #-------------------------------- Initialize HDFS --------------------------------
        self.hdfs = hdfs_model(array_size_pix = array_size_pix,
                               array_size_m = array_size_m, 
                               array_rot_angle = array_rot_angle, 
                               mag = mag, 
                               sps_fs_shape = hdfs_fs_shape, 
                               sps_fs_dim_mas = hdfs_fs_dim_mas,
                               sps_RON = hdfs_RON, 
                               sps_emccd_gain = hdfs_emccd_gain, 
                               sps_ADU_gain = hdfs_ADU_gain, 
                               sps_darkCurrent = hdfs_darkCurrent,
                               T_out = hdfs_Ts, 
                               T_d = hdfs_Td)
        
        #------------ Initialize GAPS Telescope Simulator (DM + PTT array) ---------------
        here = os.path.abspath(os.path.dirname(__file__))
        mems_ifunc_fname = os.path.join(here, 'data', 'mems2k', mems_ref_ifunc_fname)
        pupil_size_in_mems_pitches = 48
        
        self.simul_params['mems_params'] = {'mems_ifunc_fname': mems_ifunc_fname, 
                             'pupil_size_in_mems_pitches': pupil_size_in_mems_pitches,
                             'mems_grid_rot_angle': mems_grid_rot_deg}

        self.tel = telescope_simulator(array_size_pix = array_size_pix,
                                   mems_ifunc_fname = mems_ifunc_fname, 
                                   array_size_m = array_size_m, 
                                   array_rot_angle = array_rot_angle,
                                   project_truss_onaxis = project_truss_onaxis, 
                                   pupil_size_in_mems_pitches = pupil_size_in_mems_pitches,
                                   mems_grid_rot_angle = mems_grid_rot_deg)
        
        #----------------------- Atmospheric Turbulence ----------------------------------
        self.atm = atmo_disturbance(self.tel.pup, r0, L0, turb_type=turb_type,
                            single_layer_wind_speed = single_layer_wind_speed,
                            single_layer_wind_direction = single_layer_wind_direction)
        self.atm.project_to_mirror_space = project_to_mirror_space
        
        self.Components = []
        self.calib_repo = {}
        
    #================= Methods related to timing properties ========================
    @property
    def totSimulTime(self):
        return self._totSimulTime

    @totSimulTime.setter
    def totSimulTime(self, _T_):
        self._totSimulTime = _T_
        
    @property
    def tickTime(self):
        return self.__tickTime
    
    @tickTime.setter
    def tickTime(self, _T_):
        SimBlock.TICK_TIME  = _T_
        self.__tickTime = _T_

    @property
    def currentTime(self):
        return self.__currentTime
    
    @currentTime.setter
    def currentTime(self, _T_):
        SimBlock.CURRENT_TIME  = _T_
        self.__currentTime = _T_
    
    #=================== System configurations ======================================
    
    def calibrate_sensors(self, pyr_thr = 0.338, pyr_percent_extra_subaps = 9):
        """
        Calibrate NGWS-P sensors (valid sub-apertures and reference vectors).
        
        Parameters:
        -----------
        pyr_thr : float
            Flux thresholding for pupil registration. Default: 0.338
        
        percent_extra_subaps : int
            Extra sub-apertures for initial circular pupil registration (%). Default: 9% 
        """
        sys.stdout.write('--> Calibrating PWFS....\n')
        self.pwfs.calibrate(pyr_thr = pyr_thr, 
                            percent_extra_subaps = pyr_percent_extra_subaps, 
                            project_truss_onaxis = self.project_truss_onaxis)
        
        sys.stdout.write('--> Calibrating HDFS....\n')
        self.hdfs.calibrate(project_truss_onaxis = self.project_truss_onaxis)
    
    @abstractmethod
    def configure(self, **kwargs):
        pass
    
    
    def simul_noise(self, toggle=True):
        """
        Turns on/off NGWS-P noise simulation.
        
        Parameters:
        -----------
        toggle : bool
            If True, WFS noise will be simulated. Default: True
        """
        assert isinstance(toggle, bool), "toggle value must be either True or False."
        self.pwfs.wfs.simul_noise = toggle
        self.hdfs.wfs.simul_noise = toggle
    
    
    def simul_turb(self, toggle=True):
        """
        Turns on/off atmospheric turbulence simulation.
        
        Parameters:
        -----------
        toggle : bool
            If True, atmospheric turbulence will be simulated. Default: True
        """
        assert isinstance(toggle, bool), "toggle value must be either True or False."
        if toggle == True:
            self.atm.T_d = 0.0
        else:
            self.atm.T_d = np.inf # <-- do not simulate turbulence
    
    
    #===================== Closed-loop simulation ======================================
    def reset_telemetry(self):
        """
        Resets telemetry dictionary from each Component.
        """
        for comp in self.Components:
            comp.reset_telemetry()
    
    def reset_counters(self):
        """
        Resets the inner counters of each Component.
        """
        for comp in self.Components:
            comp.reset_counters()
    
    def reset(self):
        """
        Resets memory buffers from each Component.
        """
        for comp in self.Components:
            comp.reset()
    
    def trigger(self):
        """
        Call the trigger method for each Component.
        """
        for comp in self.Components:
            comp.trigger()
    
    def _collectTelemetry(self):
        """
        Collects all saved telemetry from each Component.
        NOTE: This function is called by runClosedLoop().
        
        Returns:
        --------
        A dictionary with all saved telemetry.
        """
        #TODO: Add simulation parameters to telemetry.
        telemetry = {}
        for comp in self.Components:
            telemetry.update(comp.telemetry_data)
        return telemetry
    
    
    def runClosedLoop(self, totSimulTime, liveShow=False, verbose=True):
        """
        Run a closed-loop simulation.
        
        Parameters:
        -----------
        totSimulTime : float
            Total simulated time [s].

        liveShow : bool
            If True, show telemetry live! (but very slow....). Default: False

        verbose : bool
            Display status of simulation live. Default: True
        
        Returns:
        --------
        The simulations results contained in a telemetry dictionary.
        """
        self.totSimulTime = totSimulTime
        self.totSimulIter = int(self.totSimulTime / self.tickTime) # Total number of iterations
        self.currentTime = 0.0
        self.reset_counters()
        self.reset_telemetry()
        self.reset()
        
        while self.currentTime < self.totSimulTime:
            self._tid.tic()
            self.trigger()
            self.currentTime = self.currentTime + self.tickTime
            if liveShow:
                show_live_loop(self)
            else:
                self._tid.toc()
                if verbose:
                    sys.stdout.write("\r iter: %d/%d, ET: %.3f s, on-axis WF RMS [nm]: %.1f"%(SimBlock.current_iteration(), 
                                    self.totSimulIter, self._tid.elapsedTime*1e-3, self.wf_ctrl.get_wfe()*1e9))
                    sys.stdout.flush()
        return self._collectTelemetry()
