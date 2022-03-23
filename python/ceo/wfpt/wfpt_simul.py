from ceo import constants
from ceo.wfpt import wfpt_testbed
from ceo.wfpt import wfpt_source
from ceo.wfpt import wfpt_sh48
from ceo.wfpt import wfpt_dfs
from ceo.wfpt import wfpt_visulib as visu
import numpy as np

class wfpt_simul:
    """
    A class that simulation the WFPT + Probe Zero operation.
    """
    def __init__(self, src_zen=0.0, src_azi=0.0, M2_baffle_diam=3.6, project_truss_onaxis=True,
                shs_mag=0, dfs_mag=0):

        # ----------------- SH48 init ----------------------------
        self.shs_path = wfpt_testbed(M2_baffle_diam=M2_baffle_diam, project_truss_onaxis=project_truss_onaxis,
                        path_to_sensor='SH')

        #-- GMT AGWS SH48 design parameters
        D_GMT = 25.4  #  GMT pupil diameter [m]
        SH48_N_LENSLET = 48 # linear number of AGWS SH48 sub-apertures across GMT diameter
        SH48_LENSLET_SIZE = D_GMT/SH48_N_LENSLET  # size of SH48 sub-aperture [m]
        
        #-- WFPT model parameters to ensure proper SH sampling of GMT pupil.
        self.shs_L = 27.31 # rays box size [m]. The GMT pupil is contained in this box. Value provided by R. Conan.
        N_SIDE_LENSLET = round(self.shs_L / SH48_LENSLET_SIZE)  # number of SAs across the box.
        LENSLET_SIZE = self.shs_L / N_SIDE_LENSLET

        #-- SH simulation parameters to realize proper pixel scale.
        N_PX_LENSLET = 16
        DFT_osf = 2
        N_PX_IMAGE = 24
        BIN_IMAGE = 3
        self.shs_nPx = N_SIDE_LENSLET * N_PX_LENSLET + 1 # pixels across sh_L

        self.shs_band = 'R+I'
        self.shs_mag = shs_mag
        self.shs_src =  wfpt_source(self.shs_band, self.shs_nPx, self.shs_L, 
                                mag=self.shs_mag, zenith=src_zen, azimuth=src_azi)
        self.shs = wfpt_sh48(N_SIDE_LENSLET=N_SIDE_LENSLET, N_PX_LENSLET=N_PX_LENSLET, d=LENSLET_SIZE,
               DFT_osf=DFT_osf, N_PX_IMAGE=N_PX_IMAGE, BIN_IMAGE=BIN_IMAGE)

        wfs_pxscl = self.shs_src.wavelength * BIN_IMAGE / (DFT_osf * LENSLET_SIZE) * constants.RAD2ARCSEC
        print("SH48 pixel scale: %0.3f arcsec"%wfs_pxscl)
        
        self.shs_src.fwhm = DFT_osf * BIN_IMAGE #To sample at Shannon the sub-aperture DL spot.
        self.shs_thr = 0.65 #illumination threshold for valid sub-aperture identification.

        # ----------------- DFS init  ----------------------------
        self.dfs_path = wfpt_testbed(M2_baffle_diam=M2_baffle_diam, project_truss_onaxis=project_truss_onaxis,
                        path_to_sensor='DFS')
        
        self.dfs_band = 'J'
        self.dfs_L = 27.41 #m
        self.dfs_nPx = 481 #pixels across L
        self.dfs_mag = dfs_mag
        #J_bkgd_mag = 16.2 # J-band sky bkgd (mag/arcsec^2); Tech Note GMT-02274, rev 2.4
        #J_e0 = 1.88e12    # ph/s in J band over the GMT pupil
        self.dfs_src = wfpt_source(self.dfs_band, self.dfs_nPx, self.dfs_L, 
                                mag=self.dfs_mag, zenith=src_zen, azimuth=src_azi)
        self.dfs = wfpt_dfs(self.dfs_src)

 
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
            #-- Additional transformation matrices:
            #IntMats[mirror].append('dfs-sig2pist': np.diag( 1 / np.max(np.abs(IntMats[mirror]['dfs-spp']), axis=1)))
        else:
            print("modes not recognized. sorry!")
        
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
    
       
        