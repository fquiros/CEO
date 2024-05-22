from ceo import constants
from ceo.wfpt import wfpt_testbed
from ceo.wfpt import wfpt_source
from ceo.wfpt import wfpt_sh48
from ceo.wfpt import wfpt_dfs
from ceo.wfpt import wfpt_visulib as visu
from ceo.wfpt import wfpt_utilities
import numpy as np
import os

class wfpt_simul:
    """
    A class that simulation the WFPT + Probe Zero operation.
    """
    def __init__(self, src_zen=0.0, src_azi=0.0, M2_baffle_diam=3.6, project_truss_onaxis=True,
                shs_mag=0, dfs_mag=0, keep_segments=[1,2,3,4,5,6,7]):

        # ----------------- SH48 init ----------------------------
        self.shs_path = wfpt_testbed(M2_baffle_diam=M2_baffle_diam, project_truss_onaxis=project_truss_onaxis,
                        path_to_sensor='SH', keep_segments=keep_segments)

        #-- GMT AGWS SH48 design parameters
        #D_GMT = 25.4  #  GMT pupil diameter [m]
        #SH48_N_LENSLET = 48 # linear number of AGWS SH48 sub-apertures across GMT diameter
        #SH48_LENSLET_SIZE = D_GMT/SH48_N_LENSLET  # size of SH48 sub-aperture [m]
        SH48_LENSLET_SIZE = 0.525 # size of SH48 as built on the WFPT [m]
        
        #-- WFPT model parameters to ensure proper SH sampling of GMT pupil.
        self.shs_L = 27.31 # rays box size [m]. The GMT pupil is contained in this box. Value provided by R. Conan.
        N_SIDE_LENSLET = round(self.shs_L / SH48_LENSLET_SIZE)  # number of SAs across the box.
        LENSLET_SIZE = self.shs_L / N_SIDE_LENSLET

        #-- SH simulation parameters to realize proper pixel scale.
        N_PX_LENSLET = 16
        DFT_osf = 2.625
        N_PX_IMAGE = 24
        BIN_IMAGE = 3
        self.shs_nPx = N_SIDE_LENSLET * N_PX_LENSLET + 1 # pixels across sh_L

        # ------ rays CS rotation required to render the correct orientation of the pupil w.r.t. SH grid
        RAYS_ROT_ANGLE = -90-49 #deg

        self.shs_band = 'R+I'
        self.shs_mag = shs_mag
        fudged_shs_mag = shs_mag - 6.26 # needed to simulate the right amount of detected photons
        self.shs_src =  wfpt_source(self.shs_band, self.shs_nPx, self.shs_L, rays_rot_angle = RAYS_ROT_ANGLE,
                                mag=fudged_shs_mag, zenith=src_zen, azimuth=src_azi)
        self.shs = wfpt_sh48(N_SIDE_LENSLET=N_SIDE_LENSLET, N_PX_LENSLET=N_PX_LENSLET, d=LENSLET_SIZE,
               DFT_osf=DFT_osf, N_PX_IMAGE=N_PX_IMAGE, BIN_IMAGE=BIN_IMAGE)

        wfs_pxscl = self.shs_src.wavelength * BIN_IMAGE / (DFT_osf * LENSLET_SIZE) * constants.RAD2ARCSEC
        print("SH48 pixel scale: %0.3f arcsec"%wfs_pxscl)
        
        self.shs_src.fwhm = 2 * BIN_IMAGE #To sample at Shannon the sub-aperture DL spot.
        self.shs_thr = 0.65 #illumination threshold for valid sub-aperture identification.

        # ----------------- DFS init  ----------------------------
        self.dfs_path = wfpt_testbed(M2_baffle_diam=M2_baffle_diam, project_truss_onaxis=project_truss_onaxis,
                        path_to_sensor='DFS', keep_segments=keep_segments)
        
        # ------ rays CS rotation required to render the correct orientation of the pupil w.r.t. DFS subaps
        RAYS_ROT_ANGLE = 90 #deg

        self.dfs_band = 'J'
        #self.dfs_L = 27.41 #m
        self.dfs_L = 27.1989 #m (after fine-tuning pupil. See WFPT_finetune_pixelscale.ipynb)
        self.dfs_nPx = 481 #pixels across L
        self.dfs_mag = dfs_mag
        fudged_dfs_mag = dfs_mag - 6.26 # needed to simulate the right amount of detected photons

        #J_bkgd_mag = 16.2 # J-band sky bkgd (mag/arcsec^2); Tech Note GMT-02274, rev 2.4
        #J_e0 = 1.88e12    # ph/s in J band over the GMT pupil
        self.dfs_src = wfpt_source(self.dfs_band, self.dfs_nPx, self.dfs_L, rays_rot_angle = RAYS_ROT_ANGLE,
                                mag=fudged_dfs_mag, zenith=src_zen, azimuth=src_azi)
        self.dfs = wfpt_dfs(self.dfs_src)

        # --------------- DM grid fine alignment --------------------
        self.dm_default_alignment = {"M1" : { "dm_x_shift" : -7.64136614671678e-05, 
                                              "dm_y_shift" : -0.0002069969227914712, 
                                              "dm_z_rot" : -0.000196500594940761}, 
                                     "M2" : { "dm_x_shift" : -7.332257279909798e-05,
                                              "dm_y_shift" : -8.955094150021824e-05,
                                              "dm_z_rot" : 0.013236816020918134}}
        self.dm_grid_alignment(mirror='M1')
        self.dm_grid_alignment(mirror='M2')

 
    def calibrate_sensors(self, keep_rays_for_plot=True):
        """
        Calibrate SH and DFS sensors (including reference vectors)
        """
        self.reset()
        self.propagate(keep_rays_for_plot=keep_rays_for_plot)
        
        #------------ Calibrate SH48
        self.shs_src.set_reference_wavefront()
        self.shs.calibrate(self.shs_src, self.shs_thr)
        print('SH48 calibration completed.')
        
        #------------ Calibrate DFS
        self.dfs_src.set_reference_wavefront()
        self.dfs.calibrate(self.dfs_src)
        print('DFS calibration completed.')


    def interaction_matrix(self, mirror='M1', modes='zonal'):
        """
        Calibrate interaction matrices between mirrors and sensors in probe zero.
        """
        IntMats = {}
        if modes == 'zonal':
            IntMats[mirror] = \
                {'sh-dm': self.shs_path.calibrate(self.shs, self.shs_src, mirror=mirror, 
                                                   mode='actuators', stroke=0.1),
                'sh-stt': self.shs_path.calibrate(self.shs, self.shs_src, mirror=mirror, 
                                                   mode='segment tip-tilt', stroke=1e-5),
                'dfs-dm': self.dfs_path.calibrate(self.dfs, self.dfs_src, mirror=mirror, 
                                                   mode='actuators', stroke=0.1),
                'dfs-stt': self.dfs_path.calibrate(self.dfs, self.dfs_src, mirror=mirror,
                                                   mode='segment tip-tilt', stroke=1e-5),
                'dfs-spp': self.dfs_path.calibrate(self.dfs, self.dfs_src, mirror=mirror, 
                                                   mode='segment piston', stroke=50e-9)}
            
            #-- DFS signal to SPP convertion.
            IntMats[mirror].update({'dfs-sig2pist': np.diag( 1 / np.max(np.abs(IntMats[mirror]['dfs-spp']), axis=1))})
        
        elif modes == 'modal':
            IntMats[mirror] = \
                {'sh-modes': self.shs_path.calibrate(self.shs, self.shs_src, mirror=mirror,
                                mode='segment zernikes', stroke=100e-9, wfpt_modes=self.modal_control),
                 'dfs-modes': self.dfs_path.calibrate(self.dfs, self.dfs_src, mirror=mirror,
                                mode='segment zernikes', stroke=100e-9, wfpt_modes=self.modal_control)}
            
            #-- DFS signal to SPP convertion.
            D_SPP_DFS = self.dfs_path.calibrate(self.dfs, self.dfs_src, mirror=mirror, silent=True,
                                                   mode='segment piston', stroke=50e-9)
            IntMats[mirror].update({'dfs-sig2pist': np.diag( 1 / np.max(np.abs(D_SPP_DFS), axis=1))})
            
        return IntMats
    
    
    @property
    def state(self):
        return self.shs_path.state # Note: state on dfs_path is the same
    
    
    def update(self, state):
        """
        Updates the position of all active mirrors using a state vector.
        """
        self.shs_path.update(state)
        self.dfs_path.update(state)
    
    
    def reset_mirrors(self):
        """
        Resets all active mirrors to nominal positions.
        """
        self.shs_path.reset()
        self.dfs_path.reset()
    
    
    def reset_sensors(self):
        """
        Resets SH and DFS
        """
        self.shs.reset()
        self.dfs.reset()
    
    
    def reset_sources(self):
        """
        Reset WF in source objects
        """
        self.shs_src.reset()
        self.dfs_src.reset()
    
    
    def reset(self):
        """
        Reset all WFPT simulation components (mirrors, sensors, and sources).
        """
        self.reset_mirrors()
        self.reset_sensors()
        self.reset_sources()
    
    
    def propagate(self, keep_rays_for_plot=False):
        """
        Propagates WF from source down to SH+DFS detectors.
        """
        #--- SH propagation
        self.shs_path.propagate(self.shs_src, keep_rays_for_plot=keep_rays_for_plot)
        self.shs.propagate(self.shs_src)
        #--- DFS propagation
        self.dfs_path.propagate(self.dfs_src, keep_rays_for_plot=keep_rays_for_plot)
        self.dfs.propagate(self.dfs_src)
    
    
    def opd(self, path='SH', subtract_ref_wf=True):
        """
        get OPD at exit pupil for selected path.
        
        Parameters:
        -----------
        path : string, optional
            selected path. Either {'SH', 'DFS'}. Default: 'SH'
        subtract_ref_wf: bool, optional
            remove reference WF from OPD map. Default: False
        """        
        if path=='SH':
            src = self.shs_src
        elif path=='DFS':
            src = self.dfs_src
        
        opd = src._gs.phase.host()
        mask = src._gs.amplitude.host()
        if subtract_ref_wf==True:
            opd -= src.reference_wavefront        
        return opd*mask


    def influence_matrix(self, path='SH', mirror=None, mode=None, stroke=None):
        """
        Get the influence matrix for a selected set of DoFs.

        Parameters:
        -----------
        path : string, optional
            selected path. Either {'SH', 'DFS'}. Default: 'SH'
        mirror : string
            The mirror label: eiher "M1" or "M2"
        mode : string
            The degrees of freedom: either "segment piston", "segment tip-tilt", or "actuators".
        stroke : float
            The amplitude of the motion
        """
        assert mirror in ['M1', 'M2'], '"mirror" must be either "M1" or "M2"'

        #---- Select device to poke
        if mode == "actuators":
            device = mirror+'_DM'
        else:
            device = mirror+'_PTT'

        #---- Select path to use to get exit pupil wavefront.
        if path=='SH':
            _path_ = self.shs_path
            _src_ = self.shs_src
        elif path=='DFS':
            _path_ = self.dfs_path
            _src_ = self.dfs_src

        #---- Retrieve GMT mask
        self.reset()
        _path_.propagate(_src_)
        GMTmask = _src_._gs.amplitude.host().astype('bool')
        nmask = np.sum(GMTmask)
        masktemp = GMTmask.copy() #--> for vignetting check
        segmask = np.array([vec.get() for vec in _src_.piston_mask])

        #---- Retrieve IFmat from file if available
        prefix = device
        if mode == 'segment piston':
            prefix += '_SPP'
        elif mode == 'segment tip-tilt':
            prefix += '_sTT'
        IFmat_fullname = wfpt_utilities.influence_matrix_filename(prefix, _src_._nPx,
                                        _path_._M2_baffle_diam, _path_.project_truss_onaxis)
        if os.path.isfile(IFmat_fullname):
            print("Restoring %s IFmat from file: %s"%(device, os.path.basename(IFmat_fullname)))
            with np.load(IFmat_fullname, allow_pickle=True) as data:
                IFmat = data['IFmat']
                saved_GMTmask = data['GMTmask']
                
                if mode == "actuators":
                    if data['dm_default_alignment'] != self.dm_default_alignment[mirror]:
                        raise Exception("DM alignment in saved IFmat is different from current one.")
                    
            if np.sum(np.logical_xor(GMTmask, saved_GMTmask)) > 0:
                raise Exception("Mask used in saved IFmat is different from current mask!")
            

        #--- Compute and save the IFmat
        else:
            if mode == "segment piston":
                print("Computing %s sPP influence matrix...."%device)
                n_mode = 7
                IFmat = np.zeros((nmask, n_mode))
                for jj in range(n_mode):
                    self.reset()
                    state = _path_.state
                    state[device]['segment piston'][jj] = stroke
                    _path_.update(state)
                    _path_.propagate(_src_)
                    masktemp *= _src_._gs.amplitude.host().astype('bool')
                    IFmat[:,jj] = (self.opd(path=path))[GMTmask] / stroke

            elif mode == "segment tip-tilt":
                #---> order: Rx_IFmat, Ry_IFmat
                print("Computing %s sTT influence matrix...."%device)
                n_mode = 14
                IFmat = np.zeros((nmask, n_mode))
                for jj in range(n_mode):
                    self.reset()
                    state = _path_.state
                    state[device]['segment tip-tilt'][np.mod(jj,7), jj//7] = stroke
                    _path_.update(state)
                    _path_.propagate(_src_)
                    masktemp *= _src_._gs.amplitude.host().astype('bool')
                    IFmat[:,jj] = (self.opd(path=path))[GMTmask] / stroke

            elif mode == "actuators":
                print("Computing %s influence matrix...."%device)
                n_mode = 292
                IFmat = np.zeros((nmask, n_mode))
                IF_peak = np.zeros(n_mode)
                for jj in range(n_mode):
                    self.reset()
                    state = _path_.state
                    state[device]['actuators'][jj] = stroke
                    _path_.update(state)
                    _path_.propagate(_src_)
                    masktemp *= _src_._gs.amplitude.host().astype('bool')
                    IFmat[:,jj] = (self.opd(path=path))[GMTmask] / stroke
                    IF_peak[jj] = np.max(np.abs(IFmat[:,jj]))

            if np.sum(np.logical_xor(GMTmask, masktemp)) > 0:
                raise Exception("Vignetting occurred during IFmat acquisition. Reduce stroke amplitude.")

            tosave = dict(IFmat=IFmat, GMTmask=GMTmask, segmask=segmask)
            if mode == "actuators":
                tosave['IF_peak'] = IF_peak
                tosave['dm_default_alignment'] = self.dm_default_alignment[mirror]
            np.savez(IFmat_fullname, **tosave)
            print("IFmat saved to file %s"%IFmat_fullname)

        return IFmat


    def dm_grid_alignment(self, mirror=None, dm_x_shift=0.0, dm_y_shift=0.0, dm_z_rot=0.0):
        """
        Adjust the alignment of the DM grid. All adjustments defined in the DM grid coordinate system.
        As reference: the DM pitch by design is 1.5 mm, and the GMT pupil diameter in the DM plane is 24.5 mm.
        
        Parameters:
        -----------
        mirror : string
            Either "M1" or "M2"
        dm_x_shift : float
            Shift in the x-axis (meters in the DM plane). Default: 0.0
        dm_y_shift : float
            Shift in the y-axis (meters in the DM plane). Default: 0.0
        dm_z_rot : float
            Rotation about the z-axis (radians). Default: 0.0
        """
        #-- Apply a shift to the DM grid
        assert mirror in ['M1', 'M2'], '"mirror" must be either "M1" or "M2"'
        if mirror == 'M1':
            DMs = [self.shs_path.M1_DM, self.dfs_path.M1_DM]
        elif mirror == 'M2':
            DMs = [self.shs_path.M2_DM, self.dfs_path.M2_DM]
        
        for DM in DMs:
            DM.motion_CS.origin[-1,0] = dm_x_shift + self.dm_default_alignment[mirror]['dm_x_shift']
            DM.motion_CS.origin[-1,1] = dm_y_shift + self.dm_default_alignment[mirror]['dm_y_shift']
            DM.motion_CS.euler_angles[-1,2] = dm_z_rot + self.dm_default_alignment[mirror]['dm_z_rot']
            DM.motion_CS.update()
    
    #========================= MODAL CONTROL ==========================
    def set_modal_control(self,  M1_m2c_file, M2_m2c_file):
        """
        Set up for WFPT modal control.
        
        Parameters
        ----------
        M1_m2c_file : string
            Name of .npz file containing the modes-to-commands matrix for M1.
        M2_m2c_file : string
            idem for M2
            Note: these files should be stored in the dedicated WFPT_model_data/M2C repository.
        """
        from ceo.wfpt import wfpt_modes
        self.modal_control = wfpt_modes(M1_m2c_file, M2_m2c_file)


    @property
    def modal_state(self):
        if hasattr(self, 'modal_control'):
            return self.modal_control.modal_state


    def modal_update(self, modal_state):
        """
        Updates the position of all active mirrors using a modal state vector.
        """
        if hasattr(self, 'modal_control'):
            self.modal_control.update(modal_state)
            self.update(self.modal_control.zonal_state)

    
    #======================= VISUALIZATION METHODS ========================
    
    def show_wavefront(self, path='SH', subtract_ref_wf=True, fig=None, ax=None,
                      title=None, clb_label=None):
        """
        Show wavefront map in exit pupil.
        
        Parameters:
        -----------
        path : string, optional
            path to show. Either {'SH', 'DFS'}. Default: 'SH'
        subtract_ref_wf: bool, optional
            remove reference WF from OPD map. Default: True
        """
        
        if path=='SH':
            opd = self.opd(path='SH', subtract_ref_wf=subtract_ref_wf)
            src = self.shs_src
        elif path=='DFS':
            opd = self.opd(path='DFS', subtract_ref_wf=subtract_ref_wf)
            src = self.dfs_src
        
        if subtract_ref_wf==True:
            wfe = src.phaseRms()
        else:
            wfe = src._gs.phaseRms()
        
        if title is None:
            title = path+" (%0.1f nm RMS)"%(wfe*1e9)
        visu.show_wavefront(opd*1e9, fig=fig, ax=ax, clb_label='nm WF', title=title)
    
    
    def show_rays(self, path='SH', rays_color='gray', label_surf_from=0, fig=None, ax=None):
        """
        Show rays diagram.
        """
        
        if path=='SH':
            path = self.shs_path
            src = self.shs_src
        elif path=='DFS':
            path = self.dfs_path
            src = self.dfs_src
        
        xyz,klm,sid = path.rays_data
        vig = src.rays.vignetting.host().ravel()>0
        
        visu.show_rays(xyz, klm, sid, vig, fig=fig, ax=ax, rays_color=rays_color,
                       label_surf_from=label_surf_from)
