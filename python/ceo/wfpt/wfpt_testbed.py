from ceo import ZMX, raytrace, WFPT_MX, Transform_to_R, Transform_to_S
import numpy as np
import os
import sys
import copy


class wfpt_testbed:
    """
    A class to simulate ray tracing on the WFPT optical train.
    
    Parameters
    ----------
    M2_baffle_diam : float, optional
        Physical diameter of the M2 baffle [m]. Default: 3.6
    project_truss_onaxis: bool, optional
        Simulate truss shadows as when the GMT mask is installed on the WFPT. Default: True
    path_to_sensor: string, optional. Either: "SH" or "DFS"
        Ray trace over path to selected sensor. Default: "SH"
    keep_segments: list, optional.
        List of segments installed (incomplete pupil simulation). Default: [1,2,3,4,5,6,7]
    """
    def __init__(self, M2_baffle_diam=3.6, project_truss_onaxis=True, path_to_sensor='SH', keep_segments=[1,2,3,4,5,6,7]):
                
        # ---------------------------------------------
        # Load WFPT Zemax Model (SH48 path)
        here = os.path.abspath(os.path.dirname(__file__))
        if path_to_sensor == 'SH':
            zemax_file = "wfpt-nolexitek-noadc-sh48collimator-2023-01-23-pttrot.zmx"
        elif path_to_sensor == 'DFS':
            zemax_file = "wfpt-nolexitek-noadc-dfscollimator-2023-01-23-pttrot.zmx"
        else:
            raise ValueError("'path_to_sensor' should be either 'SH' or 'DFS'.")

        zemax_path = os.path.join(here, 'WFPT_model_data', 'zemax_models', zemax_file)
        ZmxModel = ZMX.ZemaxModel(zemax_path)
        S = ZmxModel.surfaces[1:]  #remove source surface
        GlassIndex = ZmxModel.GlassIndex
        for k in range(len(S)):
            s = S[k]
            try:
                ZMX.update_material(s, GlassIndex)
            except:
                print(k,s.material)
        [ZMX.update_material(s, GlassIndex) for s in S];
        self._S = S
        self._PTT_DM_IDX = [26, 37, 47, 56]  # surface index of PTT and DMs (minus 1)
        
        #----------------------------------------------
        # Initialize Active Elements (PTT and ALPAO DMs)
        
        #---- PTT arrays:
        m12_ptt = {"segment_diameter": 16e-3, "segment_distance":17.125e-3}
        self.M1_PTT = WFPT_MX(**m12_ptt)
        self.M1_PTT.keep(keep_segments)
        self.M2_PTT = WFPT_MX(**m12_ptt)
        self.M2_PTT.keep(keep_segments)
        
        #---- ALPAO DMs:
        
        #- Create soft links in CEO/gmtMirrors/ directory to ALPAO influence function files.
        gmtMirrors_path = here
        tail = ''
        while tail != 'python':
            gmtMirrors_path, tail = os.path.split(gmtMirrors_path)
        gmtMirrors_path = os.path.join(gmtMirrors_path, 'gmtMirrors')
        assert os.path.isdir(gmtMirrors_path), 'Unable to locate the CEO/gmtMirrors folder...'
        
        ALPAO_IFs_path = os.path.join(here, 'WFPT_model_data', 'alpao_dm_ifs')
        
        if not os.path.islink(os.path.join(gmtMirrors_path, 'ALPAO_BAX450.ceo')):
            os.symlink(os.path.join(ALPAO_IFs_path,'ALPAO_BAX450.ceo'), 
                            os.path.join(gmtMirrors_path, 'ALPAO_BAX450.ceo'))
            
        if not os.path.islink(os.path.join(gmtMirrors_path, 'ALPAO_BAX449.ceo')):
            os.symlink(os.path.join(ALPAO_IFs_path,'ALPAO_BAX449.ceo'), 
                            os.path.join(gmtMirrors_path, 'ALPAO_BAX449.ceo'))    
        
        m1_dm = {"segment_diameter": 26.5e-3, "segment_id":7, "mirror_modes":"ALPAO_BAX449", "N_MODE":292}
        self.M1_DM = WFPT_MX(**m1_dm)
        m2_dm = {"segment_diameter": 26.5e-3, "segment_id":7, "mirror_modes":"ALPAO_BAX450", "N_MODE":292}
        self.M2_DM = WFPT_MX(**m2_dm)
       
        #----------------------------------------------
        # Miscelaneous 
        self._scale = 8.365/m12_ptt['segment_diameter'] # scale factor between the WFPT and the GMT
        self._M2_baffle_diam = M2_baffle_diam
        self.project_truss_onaxis = project_truss_onaxis


    # ================================ MAGIC METHODS =========================================
    def __getitem__(self,key):
        if key=="M1_PTT":
            return self.M1_PTT
        elif key=="M1_DM":
            return self.M1_DM
        elif key=="M2_PTT":
            return self.M2_PTT
        elif key=="M2_DM":
            return self.M2_DM
        else:
            raise KeyError("Available keys are: M1_PTT, M1_DM, M2_PTT or M2_DM.")

    def __ixor__(self, state):
        self.update(state)
        return self

    def __invert__(self):
        self.reset()
    
    # ================================ PUBLIC METHODS =========================================
    def propagate(self, src, keep_rays_for_plot=False):
        """
        Propagates rays from illumination source to the exit pupil of the WFPT.
        
        Parameters
        ----------
        src : wfpt_source
            The illumination source (aka guide star)
        keep_rays_for_plot: bool
            If True, it will store rays data required for plotting rays (Careful: time consuming). Default: False
        """
        
        src.reset_rays()
        if keep_rays_for_plot == True:
            xyz = [src.rays.coordinates.host()]
            klm = [src.rays.directions.host()]
            sid = [] # surface index
        else:
            xyz = None
            klm = None
            sid = None
        
        #-- 1. Ray tracing to the last surface before M1 PTT
        [raytrace.raytrace(src.rays, src.wavelength, self._S, k+1, xyz, klm, sid) for k in range(self._PTT_DM_IDX[0])];
        
        #-- 2. Ray tracing on M1 PTT (and apply GMT mask)
        self.M1_PTT.trace(src.rays)
        if self.project_truss_onaxis == True:
            src.rays.gmt_truss_onaxis(self._scale)
        src.rays.gmt_m2_baffle(self._M2_baffle_diam, self._scale)
        idx = self._PTT_DM_IDX[0]
        raytrace.bounceoff(src.rays, self._S, idx, xyz, klm, sid)

        #-- 3. Ray tracing to the last surface before M2 PTT
        [raytrace.raytrace(src.rays, src.wavelength, self._S, k+1, xyz, klm, sid) for k in range(self._PTT_DM_IDX[0]+1,self._PTT_DM_IDX[1])];

        #-- 4. Ray tracing on M2 PTT
        self.M2_PTT.trace(src.rays)
        idx = self._PTT_DM_IDX[1]
        raytrace.bounceoff(src.rays, self._S, idx, xyz, klm, sid)

        #-- 5. Ray tracing to the last surface before M1 DM
        [raytrace.raytrace(src.rays, src.wavelength, self._S, k+1, xyz, klm, sid) for k in range(self._PTT_DM_IDX[1]+1,self._PTT_DM_IDX[2])];
        
        #-- 6. Ray tracing on M1 DM
        self.M1_DM.trace(src.rays)
        idx = self._PTT_DM_IDX[2]
        raytrace.bounceoff(src.rays, self._S, idx, xyz, klm, sid)       

        #-- 7. Ray tracing to the last surface before M2 DM
        [raytrace.raytrace(src.rays, src.wavelength, self._S, k+1, xyz, klm, sid) for k in range(self._PTT_DM_IDX[2]+1,self._PTT_DM_IDX[3])];

        #-- 8. Ray tracing on M2 DM
        self.M2_DM.trace(src.rays)
        idx = self._PTT_DM_IDX[3]
        raytrace.bounceoff(src.rays, self._S, idx, xyz, klm, sid)
            
        #-- 9. Ray tracing to the very last surface
        [raytrace.raytrace(src.rays, src.wavelength, self._S, k+1, xyz, klm, sid) for k in range(self._PTT_DM_IDX[3]+1,len(self._S))];
        
        if keep_rays_for_plot == True:
            self.rays_data=[xyz,klm,sid]
            
        src.applyOPD()


    def dm_grid_alignment(self, mirror=None, dm_x_shift=0.0, dm_y_shift=0.0, dm_z_rot=0.0):
        """
        Adjust the alignment of the DM grid. All adjustments defined in the DM grid coordinate system.
        
        Parameters:
        -----------
        mirror : string
            Either "M1" or "M2"
        dm_x_shift : float
            Shift in the x-axis (expressed as fraction of pupil size). Default: 0.0
        dm_y_shift : float
            Shift in the y-axis (expressed as fraction of pupil size). Default: 0.0
        dm_z_rot : float
            Rotation about the z-axis (radians). Default: 0.0
        """
        #-- Apply a shift to the DM grid
        assert mirror in ['M1', 'M2'], '"mirror" must be either "M1" or "M2"'
        if mirror == 'M1':
            DM = self.M1_DM
        elif mirror == 'M2':
            DM = self.M2_DM
            
        DM.motion_CS.origin[-1,0] = dm_x_shift * DM.D_clear
        DM.motion_CS.origin[-1,1] = dm_y_shift * DM.D_clear
        DM.motion_CS.euler_angles[-1,2] = dm_z_rot
        DM.motion_CS.update()


    @property
    def state(self):
        d = {'M1_PTT': {'segment piston': (-1)*self.M1_PTT.motion_CS.origin[:,2],
                        'segment tip-tilt': (-1)*self.M1_PTT.motion_CS.euler_angles[:,0:2]},
             'M2_PTT': {'segment piston': self.M2_PTT.motion_CS.origin[:,2],
                        'segment tip-tilt': self.M2_PTT.motion_CS.euler_angles[:,0:2]},
             'M1_DM': {'actuators': (-1)*self.M1_DM.modes.a[-1,:]},
             'M2_DM': {'actuators': self.M2_DM.modes.a[-1,:]},
            }
        return wfpt_state(d)


    def update(self, state):
        """
        Updates the position of all active mirrors using a state vector.
        """
        for mirror in state.state:
            if 'M1' in mirror:
                sign = -1
            else:
                sign = +1
            if 'PTT' in mirror:
                for dof in state[mirror]:
                    if dof == 'segment piston':
                        self[mirror].motion_CS.origin[:,2] = sign * state[mirror][dof][:]
                    elif dof == 'segment tip-tilt':
                        self[mirror].motion_CS.euler_angles[:,0:2] = sign * state[mirror][dof][:]
                self[mirror].motion_CS.update()
            if 'DM' in mirror:
                self[mirror].modes.a[-1,:] = sign * state[mirror]['actuators'][:]
                self[mirror].modes.update()
    
    
    def reset(self):
        """
        Resets all active mirrors to nominal positions and clears the WFPT state vector.
        """
        self.M1_PTT.motion_CS.reset()
        self.M2_PTT.motion_CS.reset()
        self.M1_DM.modes.reset()
        self.M2_DM.modes.reset()
        
    
    def calibrate(self, wfs, src, mirror=None, mode=None, stroke=None, mode_list=None):
        """
        Calibrate the different degrees of freedom of the  mirrors

        Parameters
        ----------
        wfs : wfpt_sh48, wfpt_dfs, etc
            The wavefront sensor
        src : wfpt_source
            The illumination source (aka guide star)
        mirror : string
            The mirror label: eiher "M1" or "M2" ("MOUNT" is also accepted and will emulate a telescope pointing error)
        mode : string
            The degrees of freedom label
            for M1: "global tip-tilt", "zernike", "bending modes", "Txyz", "Rxyz", "segment tip-tilt", "segment piston", actuators"
            for M2: "global tip-tilt", "zernike", "Txyz", "Rxyz", "segment tip-tilt", "segment piston", "actuators"
            for MOUNT: "pointing"
        stroke : float
            The amplitude of the motion
        mode_list : list
            Subset of actuators specified in a list to calibrate (applicable to DMs only).
        """
        
        def M1_DM_zonal_update(_stroke_):
            self.M1_DM.modes.a[-1,kAct] = (-1) * _stroke_
            self.M1_DM.modes.update()
            
        def M2_DM_zonal_update(_stroke_):
            self.M2_DM.modes.a[-1,kAct] = _stroke_
            self.M2_DM.modes.update()
            
        def pushpull(action):
            def get_slopes(stroke_sign):
                self.reset()
                action(stroke_sign*stroke)
                src.reset()
                self.propagate(src)
                wfs.reset()
                wfs.analyze(src)
                #print("max abs value: %2.3f"%np.max(np.abs(wfs.get_measurement())))
                #print("slope rms: %2.3f, %2.3f"%wfs.measurement_rms())
                return wfs.get_measurement()
            s_push = get_slopes(+1)
            s_pull = get_slopes(-1)
            return 0.5*(s_push-s_pull)/stroke        
        
        #-------- Start of main program -------------------
        sys.stdout.write("___ %s ___ (%s)\n"%(mirror,mode))
        
        if mirror=="M1":
            if mode=="actuators":
                if mode_list is None:
                    n_mode = self.M1_DM.modes.n_mode
                    mode_list = range(n_mode)
                else:
                    n_mode = len(mode_list)
                D = np.zeros((wfs.get_measurement_size(),n_mode))
                idx = 0
                for kAct in mode_list:
                    sys.stdout.write("%d "%(kAct))
                    D[:,idx] = np.ravel( pushpull( M1_DM_zonal_update ) )
                    idx += 1
                sys.stdout.write("\n")
                
            if mode=="segment tip-tilt":
                D = np.zeros((wfs.get_measurement_size(),2*7))
                idx = 0
                Rx = lambda x : self.M1_PTT.update(origin=[0,0,0],euler_angles=[-x,0,0],idx=kSeg)
                Ry = lambda x : self.M1_PTT.update(origin=[0,0,0],euler_angles=[0,-x,0],idx=kSeg)
                sys.stdout.write("Segment #:")
                for kSeg in range(1,8):
                    sys.stdout.write("%d "%kSeg)
                    D[:,idx] = pushpull( Rx )
                    idx += 1
                    D[:,idx] = pushpull( Ry )
                    idx += 1
                sys.stdout.write("\n")                

            if mode=="segment piston":
                n_mode = 7
                D = np.zeros((wfs.get_measurement_size(),n_mode))
                idx = 0
                Tz = lambda x : self.M1_PTT.update(origin=[0,0,-x],euler_angles=[0,0,0],idx=kSeg)
                sys.stdout.write("Segment #:")
                for kSeg in range(1,8):
                    sys.stdout.write("%d "%kSeg)
                    D[:,idx] = pushpull( Tz )
                    idx += 1
                sys.stdout.write("\n")

        if mirror=="M2":
            if mode=="actuators":
                if mode_list is None:
                    n_mode = self.M2_DM.modes.n_mode
                    mode_list = range(n_mode)
                else:
                    n_mode = len(mode_list)
                D = np.zeros((wfs.get_measurement_size(),n_mode))
                idx = 0
                for kAct in mode_list:
                    sys.stdout.write("%d "%(kAct))
                    D[:,idx] = np.ravel( pushpull( M2_DM_zonal_update ) )
                    idx += 1
                sys.stdout.write("\n")

            if mode=="segment tip-tilt":
                D = np.zeros((wfs.get_measurement_size(),2*7))
                idx = 0
                Rx = lambda x : self.M2_PTT.update(origin=[0,0,0],euler_angles=[x,0,0],idx=kSeg)
                Ry = lambda x : self.M2_PTT.update(origin=[0,0,0],euler_angles=[0,x,0],idx=kSeg)
                sys.stdout.write("Segment #:")
                for kSeg in range(1,8):
                    sys.stdout.write("%d "%kSeg)
                    D[:,idx] = pushpull( Rx )
                    idx += 1
                    D[:,idx] = pushpull( Ry )
                    idx += 1
                sys.stdout.write("\n")

            if mode=="segment piston":
                n_mode = 7
                D = np.zeros((wfs.get_measurement_size(),n_mode))
                idx = 0
                Tz = lambda x : self.M2_PTT.update(origin=[0,0,x],euler_angles=[0,0,0],idx=kSeg)
                sys.stdout.write("Segment #:")
                for kSeg in range(1,8):
                    sys.stdout.write("%d "%kSeg)
                    D[:,idx] = pushpull( Tz )
                    idx += 1
                sys.stdout.write("\n")
        sys.stdout.write("------------\n")
        return D


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

    def __str__(self):
        return str(self.state)

    def __getitem__(self, key):
        return self.state[key]
