from ceo import GMT_MX, Source, Pyramid, HolographicDFS, cuFloatArray
from SimBlock import SimBlock
import numpy as np

PupilArea = 356.0                       # [m^2] Takes into account baffled segment borders
tel_throughput = 0.9**4                 # M1 + M2 + M3 + GMTIFS dichroic = 0.9^4
ngws_throughput = tel_throughput * 0.4  # NGWS board: 0.4
e0_RIband = 9e12                        # zeropoint in photons/s in R+I band over the GMT pupil
delta_wl_RIband = 300e-9                # from official photometry table

class pwfs_model(SimBlock):
    """
    Class that encapsulates the PWFS model configured to match the NGWS-P parameters.
    
    Parameters:
    --------------------
    array_size_pix : int
        Size in pixels of simulated array containing the GMT pupil. Default: 460

    array_size_m : float
        Size in meters of simulated array containing the GMT pupil (should be slightly larger than GMT diameter). Default: 25.5 m
    
    array_rot_angle : float
        Angle between pyramid facets and GMT pupil [deg]. Default: 15.0

    mag : float
        NGS R magnitude. Default: 10
    
    pyr_modulation : float
        modulation radius in lambda/D units. Default: 2.0
    
    pyr_RON : float
        Readout noise in e- RMS. Default: 0.5
    
    pyr_emccd_gain : int
        EMCCD gain. Default: 600
    
    pyr_ADU_gain : float
        ADU gain. Default: 1/30
    
    Parameters passed on to parent class SimBlock:
    ----------------------------------------------
    T_out : float
        Integration time of simulation block [in seconds]. Default: TICK_TIME
        
    T_d : float
        Time delay for the simulation block to start operation [in seconds]. Default: 0.0
    """
    def __init__(self, array_size_pix = 460, array_size_m = 25.5, array_rot_angle = 15.0, 
                 mag = 10, pyr_modulation = 2.0, pyr_RON = 0.5, pyr_emccd_gain = 600, 
                 pyr_ADU_gain = 1/30, T_out = None, T_d = 0.0):
        
        #--> SimBlock timing parameters
        #==================================================
        super().__init__(T_out=T_out, T_d=T_d)

        #--> Pyramid WFS initialization
        #==================================================
        print("---> Initializing PWFS...")
        pyr_binning = 1
        nLenslet = 92 // pyr_binning  # sub-apertures across the pupil
        
        nPx = 92*10//2 # Sampling of input WF is not a free parameter!
        if nPx != array_size_pix:
            raise ValueError("Sampling is not a free parameter... Set array_size_pix to 460")
        
        pyr_separation = 132 // pyr_binning # separation between centers of adjacent sub-pupil images [pix]
        pyr_throughput = 0.9 * ngws_throughput
        
        pwfs = Pyramid(nLenslet, nPx, 
                       modulation=pyr_modulation, 
                       throughput=pyr_throughput, 
                       separation=pyr_separation/nLenslet)
        if pyr_modulation == 1.0: pwfs.modulation_sampling = 16

        #-- Add parameters to pwfs object for traceability
        pwfs.binning = pyr_binning
        
        #--> Pyramid NGS and sensing band initialization
        #==================================================
        #--> PWFS sensing band
        wl1st = 760e-9
        delta_wl1st = 320e-9
        wvl1st_band = [wl1st-delta_wl1st/2., wl1st+delta_wl1st/2.]
        
        #---> Zero point for the desired bandwidth
        e0_wl1st = e0_RIband * (delta_wl1st/delta_wl_RIband) / PupilArea    #in ph/m^2/s in the desired bandwidth
        #print("Zero point [ph/m^2/s]: %.3e"%e0_wl1st)
        pyr_band = [wl1st, delta_wl1st, e0_wl1st]

        gs = Source(pyr_band, magnitude=mag, zenith=0.,azimuth=0., 
                    rays_box_size=array_size_m, 
                    rays_box_sampling=nPx, 
                    rays_origin=[0.0,0.0,25])
        gs.rays.rot_angle = array_rot_angle * np.pi/180
        
        print('Number of simulated PWFS GS photons [ph/s/m^2]: %.1f'%(gs.nPhoton))
        print(u"Number of pixels across %1.1f-m array: %d"%(array_size_m,nPx))
        
        #--> PWFS camera parameters initialization
        #==================================================
        pyr_emccd_nbits = 14
        self.readout_params = dict(RON=pyr_RON, emccd_gain=pyr_emccd_gain,
                                 ADU_gain=pyr_ADU_gain, emccd_nbits=pyr_emccd_nbits)
        pwfs.ccd_size = 240
        pwfs.simul_noise = True
        
        #--> Store objects and parameters
        #==================================================
        self.__gs = gs
        self.wfs = pwfs
        
        #--> Telemetry buffers
        #==================================================        
        self.telemetry_data['pwfs_meas'] = []
        #self.telemetry_data['pwfs_time_vec'] = []
        self.telemetry_data['pwfs_meas_rms'] = []
    
    
    def __getattr__(self, item):
        return getattr(self.wfs,item)
    
    
    def calibrate(self, pyr_thr = 0.338, percent_extra_subaps = 9, project_truss_onaxis=False):
        """
        PWFS calibrations (pupil registration and slope-null vector).
        
        Parameters:
        ---------------
        pyr_thr : float
            Flux thresholding for pupil registration. Default: 0.338
        
        percent_extra_subaps : int
            Extra sub-apertures for initial circular pupil registration (%). Default: 9%
        
        project_truss_onaxis : bool
            If True, simulates the truss shadows over central segment. Default: False
        """
        gmt = GMT_MX()
        gmt.project_truss_onaxis = project_truss_onaxis
        gmt.reset()
        self.__gs.reset()
        gmt.propagate(self.__gs) # <-- sets reference complex amplitude in "gs" object.
        self.__gs.wavefront.reset_phase()

        #-- Calibrate PYR (Pupil registration, and slope null vector)
        self.wfs.calibrate(self.__gs, 
                            percent_extra_subaps=percent_extra_subaps, 
                            cen_thr=0.2, 
                            thr = pyr_thr)
        self.wfs._extr = (self.wfs.ccd_frame.shape[0]-self.ccd_size//self.binning)//2
    
    
    @property
    def ccd_frame(self):
        extr = self.wfs._extr
        return self.wfs.ccd_frame[extr:-extr, extr:-extr]
    
    @property
    def mag(self):
        return self.__gs.magnitude[0]
    
    @mag.setter
    def mag(self, _mag_):
        self.__gs.magnitude = [_mag_]
    
    
    def measure(self, wf_in):
        """
        Full propagation of input wavevfront onto the detector's plane, and computation of WFS measurement vector.
        Note: loop timing parameters do not apply.
        
        Parameters:
        ---------------
        wf_in : numpy array
            Input wavefront phase [m WF].
        """        
        self.__gs.wavefront.reset_phase()
        self.__gs.wavefront.addPhase(cuFloatArray(host_data=wf_in))
        self.wfs.propagate(self.__gs)
        if self.wfs.simul_noise == True:
            self.wfs.readOut(self.T_out, **self.readout_params)
        else:
            self.wfs.camera.noiselessReadOut(self.T_out)
        self.wfs.process()
        return self.wfs.get_measurement()
    
    
    def register_input_method(self, get_wavefront):
        """
        Registers the external method used to retrieve the input wavefront.
        
        Parameters:
        -----------
        get_wavefront : callable
            External method that provides a wavefront in nPx x nPx format.
        """
        assert callable(get_wavefront), "'get_wavefront' must be a callable function."
        assert get_wavefront().shape == (self.__gs.n, self.__gs.n), "'get_wavefront()' does not provide an output with the expected format."
        self.__get_wf = get_wavefront
    
    
    def _integrate(self):
        """
        Propagate WF to detector plane integrating for a total of T_out seconds.
        NOTE: This internal function is called by the trigger() method defined in the SimBlock parent class.
        """
        #--> Load input wavefront to Source object:
        wf_in = self.__get_wf()
        self.__gs.wavefront.reset_phase()
        self.__gs.wavefront.addPhase(cuFloatArray(host_data=wf_in))
        #--> Propagate WF through WFS model:
        self.wfs.propagate(self.__gs)
    
    
    def _compute_output(self):
        """
        Processes the integrated PWFS frame and computes the PWFS measurement.
        NOTE: This internal function is called by the trigger() method defined in the SimBlock parent class.
        """
        if self.wfs.simul_noise == True:
            self.wfs.readOut(self.T_out, **self.readout_params)
        else:
            self.wfs.camera.noiselessReadOut(self.T_out)
        self.wfs.process()
        self._updateTelemetry()
        self.wfs.reset() #resets camera frame, but measurement buffer remains unmodified.
    
    
    def _updateTelemetry(self):
        """
        Updates telemetry buffer.
        Note: Internal method called by _compute_output()
        """
        self.telemetry_data['pwfs_meas'] += [self.get_measurement()]
        #self.telemetry_data['pwfs_time_vec'] += [SimBlock.CURRENT_TIME + SimBlock.TICK_TIME]
        self.telemetry_data['pwfs_meas_rms'] += [self.measurement_rms()]


class hdfs_model(SimBlock):
    """
    Class that encapsulates the HDFS model configured to match the NGWS-P parameters.
    
    Parameters:
    --------------------
    array_size_pix : int
        Size in pixels of simulated array containing the GMT pupil. Default: 460

    array_size_m : float
        Size in meters of simulated array containing the GMT pupil (should be slightly larger than GMT diameter). Default: 25.5 m
    
    array_rot_angle : float
        Angle between pyramid facets and GMT pupil [deg]. Default: 15.0

    mag : float
        NGS R magnitude. Default: 10
    
    sps_fs_shape : string
        Type of field stop: "square", "round", "none". Default: "round"
    
    sps_fs_dim_mas : float
        size/diameter of field stop [in mas]. Default: 50
    
    sps_RON : float
        Readout noise in e- RMS. Default: 0.5
    
    sps_emccd_gain : int
        EMCCD gain. Default: 100
    
    sps_ADU_gain : float
        ADU gain. Default: 1/2.76 
        Note: ProEM when readout speed set to 30 MHz (see HDFS design report, GMT-DOC-05337, Figure 9-7).
    
    sps_darkCurrent : float
        EMCCD dark current. Default: 0.
    
    Parameters passed on to parent class SimBlock:
    ----------------------------------------------
    T_out : float
        Integration time of simulation block [in seconds]. Default: TICK_TIME
        
    T_d : float
        Time delay for the simulation block to start operation [in seconds]. Default: 0.0        
    """
    def __init__(self, array_size_pix = 460, array_size_m = 25.5, array_rot_angle = 15.0,
                 mag = 10, sps_fs_shape = 'round', sps_fs_dim_mas = 50,
                 sps_RON = 0.5, sps_emccd_gain = 100, sps_ADU_gain = 1/2.76,
                 sps_darkCurrent = 0., T_out=None, T_d=0.0):
        
        assert array_size_pix == 460, "Sampling is not a free parameter... Set array_size_pix to 460"
        assert array_rot_angle == 15.0, "Mask rotation angle is not a free parameter... Set array_rot_angle to 15.0 deg"
        
        #--> SimBlock timing parameters
        #==================================================
        super().__init__(T_out=T_out, T_d=T_d)
        
        #--> HDFS NGS and sensing band initialization
        #==================================================
        #--> HDFS sensing band
        wl2nd = 810e-9
        delta_wl2nd = 220e-9
        wvl_band = [wl2nd-delta_wl2nd/2.,wl2nd+delta_wl2nd/2.]
        
        #---- Scale zeropoint for the desired bandwidth:
        e0_wl2nd = e0_RIband * (delta_wl2nd/delta_wl_RIband) / PupilArea   #in ph/m^2/s in the desired bandwidth
        #print("Zero point [ph/m^2/s]: %.3e"%e0_wl2nd)
        sps_band = [wl2nd, delta_wl2nd, e0_wl2nd]
        
        gs = Source(sps_band, magnitude=mag, zenith=0.,azimuth=0.,
                      rays_box_size=array_size_m,
                      rays_box_sampling=array_size_pix,
                      rays_origin=[0.0,0.0,25])
        gs.rays.rot_angle = array_rot_angle * np.pi/180
        
        #--> HDFS initialization
        #==================================================
        print("---> Initializing HDFS...")
        sps_hdfs_design = 'v2a_gaps' # This is only valid for a pupil rotation angle of 15 deg!!
        sps_achromatic_mask =  False
        sps_lobe_detection = 'peak_value'      
        sps_fov = 1.0  #1.4 #arcsec diameter
        sps_fringe_window_size_mas = 140
        sps_fp_pxscl_mas = 2.4  #mas
        sps_spectral_type = 'tophat'  #-------to be updated with a measured HDFS spectra.
        sps_apodization_window_type = 'Tukey'
        sps_processing_method = 'DFS'
        sps_qe_model = 'ProEM'
        sps_sky_bkgd_model = 'none'
        sps_throughput = 0.1 * ngws_throughput
        
        hdfs = HolographicDFS(hdfs_design=sps_hdfs_design, wvl_band=wvl_band, wvl_res=10e-9,
                              D = array_size_m,
                              fov_mas = sps_fov*1e3, 
                              fp_pxscl_mas = sps_fp_pxscl_mas,
                              fs_shape = sps_fs_shape, 
                              fs_dim_mas = sps_fs_dim_mas, 
                              spectral_type = sps_spectral_type,
                              fringe_window_size_mas = sps_fringe_window_size_mas,
                              apodization_window_type = sps_apodization_window_type,
                              processing_method = sps_processing_method, 
                              throughput = sps_throughput,
                              qe_model = sps_qe_model, 
                              sky_bkgd_model = sps_sky_bkgd_model, 
                              achromatic_mask = sps_achromatic_mask)
        
        #--> HDFS camera parameters initialization
        #==================================================
        sps_emccd_nbits = 14
        hdfs.camera.readoutNoiseRms = sps_RON
        hdfs.camera.EM_gain = sps_emccd_gain
        hdfs.camera.ADU_gain = sps_ADU_gain
        hdfs.camera.nbits = sps_emccd_nbits
        hdfs.camera.darkCurrent = sps_darkCurrent
        hdfs.simul_noise = True
        
        self.readout_params = dict(RON=sps_RON, emccd_gain=sps_emccd_gain,
                                 ADU_gain=sps_ADU_gain, emccd_nbits=sps_emccd_nbits,
                                 darCurrent=sps_darkCurrent)
        
        #--> Store objects and parameters
        #==================================================        
        self.__gs = gs
        self.wfs = hdfs
        
        #--> Telemetry buffers
        #==================================================
        self.telemetry_data['hdfs_meas'] = []
        self.telemetry_data['hdfs_fringes'] = []
        self.telemetry_data['hdfs_time_vec'] = []
    
    
    def __getattr__(self, item):
        return getattr(self.wfs,item)
    
    
    def calibrate(self, project_truss_onaxis=False):
        """
        HDFS calibrations (reference vector).
        
        Parameters:
        -----------
        project_truss_onaxis : bool
            If True, simulates the truss shadows over central segment. Default: False        
        """
        gmt = GMT_MX()
        gmt.project_truss_onaxis = project_truss_onaxis
        gmt.reset()
        self.__gs.reset()
        gmt.propagate(self.__gs) # <-- sets reference complex amplitude in "gs" object.
        self.__gs.wavefront.reset_phase()
        
        self.wfs.calibrate(self.__gs)
        sps_flux_unitary_norm = np.sum(self.wfs._image.get())
        print('HDFS total flux should be equal to one, but we get: %0.4f'%sps_flux_unitary_norm)
    
    
    @property
    def ccd_frame(self):
        return self.wfs._image.get()
    
    @property
    def mag(self):
        return self.__gs.magnitude[0]
    
    @mag.setter
    def mag(self, _mag_):
        self.__gs.magnitude = [_mag_]
    
    
    def measure(self, wf_in):
        """
        Full propagation of input wavevfront onto the detector's plane, and computation of WFS measurement vector.
        Note: loop timing parameters do not apply.
        
        Parameters:
        ---------------
        wf_in : numpy array
            Input wavefront phase [m WF].
        """         
        self.__gs.wavefront.reset_phase()
        self.__gs.wavefront.addPhase(cuFloatArray(host_data=wf_in))
        self.wfs.propagate(self.__gs)
        if self.wfs.simul_noise == True:
            self.wfs.readOut(self.T_out)
        else:
            self.wfs.noiselessReadOut(self.T_out)
        self.wfs.process()
        return self.wfs.get_measurement()
    
    
    def register_input_method(self, get_wavefront):
        """
        Registers the external method used to retrieve the input wavefront.
        
        Parameters:
        -----------
        get_wavefront : callable
            External method that provides a wavefront in nPx x nPx format.
        """
        assert callable(get_wavefront), "'get_wavefront' must be a callable function."
        assert get_wavefront().shape == (self.__gs.n, self.__gs.n), "'get_wavefront()' does not provide an output with the expected format."
        self.__get_wf = get_wavefront
    
    
    def _integrate(self):
        """
        Propagate WF to detector plane integrating for a total of T_out seconds.
        NOTE: This internal function is called by the trigger() method defined in the SimBlock parent class.
        
        """
        #--> Load input wavefront to Source object:
        wf_in = self.__get_wf()
        self.__gs.wavefront.reset_phase()
        self.__gs.wavefront.addPhase(cuFloatArray(host_data=wf_in))
        #--> Propagate WF through WFS model:
        self.wfs.propagate(self.__gs)
    
    
    def _compute_output(self):
        """
        Processes the integrated HDFS frame and computes the HDFS measurement.
        NOTE: This internal function is called by the trigger() method defined in the SimBlock parent class.
        """
        if self.wfs.simul_noise == True:
            self.wfs.readOut(self.T_out)
        else:
            self.wfs.noiselessReadOut(self.T_out)
        self.wfs.process()
        self._updateTelemetry()
        self.wfs.reset() #resets camera frame, but measurement buffer remains unmodified.
    
    
    def _updateTelemetry(self):
        """
        Updates telemetry buffer.
        Note: Internal method called by _compute_output()
        """
        self.telemetry_data['hdfs_meas'] += [self.get_measurement()]
        self.telemetry_data['hdfs_fringes'] += \
            [self.extract_fringes(apodize=True, normalize=False, derotate=True).get()]
        self.telemetry_data['hdfs_time_vec'] += [SimBlock.CURRENT_TIME + SimBlock.TICK_TIME]
    
    


    