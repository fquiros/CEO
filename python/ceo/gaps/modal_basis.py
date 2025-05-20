import numpy as np
from ceo import GMT_MX, Source, PhaseProjectionSensor

class segment_modal_basis:
    """
    Class to generate GMT segment modal basis (KLs or Zernike modes available). 
    
    Parameters:
    -----------
    array_size_pix : int
        Size in pixels of simulated array containing the GMT pupil.
    
    array_size_m : float
        Size in meters of simulated array containing the GMT pupil (should be slightly larger than GMT diameter). Default: 25.5 m
    
    array_rot_angle : float
        Angle of rotation in degrees of the GMT pupil. Default: 0 deg
        
    Usage:
    -------
    After instantiation, call the method to generate the desired modes:
        - For segment KL modes call "do_segment_kls()" method.
        - For segment Zernike modes call "do_segment_zernike()" method.
    You can then retrieve the wavefront of a given mode using the methods:
        - get_segkl_wf() for segment KL modes.
        - get_segzern_wf() for segment Zernike modes.
    """
    def __init__(self, array_size_pix, array_size_m = 25.5, array_rot_angle = 0.0):
        
        #-- Initialize CEO objects used to generate the theoretical modes
        gmt = GMT_MX(M2_mirror_modes="M2_OC36p_OrthoNorm_EP_KarhunenLoeveModes", M2_N_MODE=300)
        gmt.project_truss_onaxis = False
        gs = Source('R+I', rays_box_size = array_size_m, 
                        rays_box_sampling = array_size_pix, 
                        rays_rot_angle = array_rot_angle * np.pi / 180)
        
        #-- Extract the GMT segment masks
        gmt.reset()
        gs.reset()
        gmt.propagate(gs)
        self._phase_ref = gs.wavefront.phase.host()
        self._P = np.squeeze(np.array(gs.rays.piston_mask))
        self._GMTmask = np.sum(self._P,axis=0).astype('bool')
        self._GMTmask2D = self._GMTmask.reshape((array_size_pix,array_size_pix))
        self._nPx = array_size_pix
        self._npseg = np.sum(self._P, axis=1)
        self._nmask = np.sum(self._P)
        
        #-- Store CEO objects
        self.__gmt = gmt
        self.__gs = gs
    
    
    def do_segment_kls(self, KLamp=100e-9):
        """
        Generate GMT segment KL modes. The first three modes are pure segment piston, tip, and tilt.
        
        Parameters:
        -----------
        KL_amp : float
            Amplitude to apply to M2 segment KL modes [m RMS]. Default: 100e-9
        
        Notes:
        ------
        The list containing the segment KL modes is stored in the "segKLmat" property.
        """
        self.segKLmat = [np.zeros((self._npseg[segId], self.__gmt.M2.modes.n_mode)) for segId in range(7) ]
        masktemp = self._GMTmask.copy()

        for jj in range(self.__gmt.M2.modes.n_mode):
            self.__gmt.reset()
            self.__gmt.M2.modes.a[:, jj] = KLamp
            self.__gmt.M2.modes.update()
            self.__gs.reset()
            self.__gmt.propagate(self.__gs)
            masktemp *= self.__gs.amplitude.host().astype('bool').ravel()
            for segId in range(7):
                self.segKLmat[segId][:,jj] = np.squeeze(self.__gs.wavefront.phase.host() - \
                                                        self._phase_ref)[self._P[segId,:]] / KLamp #normalize to unitary RMS WF

        if np.sum(np.logical_xor(self._GMTmask, masktemp)) > 0:
            raise Exception("Vignetting occurred during IFmat acquisition. Reduce stroke amplitude.")
        
        self.n_modes_per_segment = self.__gmt.M2.modes.n_mode
        print("--> Segment KL modes computed successfully.")
        
    
    def do_segment_zernikes(self, zernikes_radial_order=4):
        """
        Generate GMT segment Zernike modes.

        Parameters:
        -----------
        zernike_radial_order : int
            Last radial order of Zernikes to generate. Default: 4 (i.e. 15 Zernike modes)
        
        Notes:
        ------
        The list containing the segment Zernike modes is stored in the "segZmat" property.
        """
        zsensor = PhaseProjectionSensor(zernikes_radial_order)
        
        #---- Generate theoretical Zernike modes
        zsensor.calibrate(self.__gs, CS_rotation=True)
        
        #---- Re-orthonormalize zernike modes over segment pupils.
        self.segZmat = []

        for segid in range(7):
            Zmat1 = np.squeeze(zsensor.Zmat[:,:,segid])
            Dmat1 = np.matmul(np.transpose(Zmat1), Zmat1) / self._npseg[segid]
            Lmat1 = np.linalg.cholesky(Dmat1)
            inv_Lmat1 = np.linalg.pinv(Lmat1)
            Zmato = np.matmul(Zmat1, np.transpose(inv_Lmat1))
            self.segZmat.append( Zmato[self._P[segid,:],:])
        
        self.n_modes_per_segment = (zernikes_radial_order+1) * (zernikes_radial_order+2) // 2
        print("--> Theoretical segment Zernike modes computed successfully.")
    
    
    def get_segkl_wf(self, segId, modenum):
        """
        Get the wavefront corresponding to a given segment KL mode.
        
        Parameters:
        ------------
        segId : int
            segment index number [0 to 6], where 6 is the central segment.
        
        modenum : int
            segment mode number
        """
        if not hasattr(self, 'segKLmat'):
            raise Exception("Compute segment KL modes first using 'do_segment_kls()'...")
        
        wf = np.zeros((self._nPx**2))
        wf[self._P[segId,:]] = self.segKLmat[segId][:,modenum]
        return wf.reshape((self._nPx, self._nPx))
    
    
    def get_segzern_wf(self, segId, modenum):
        """
        Get the wavefront corresponding to a given segment Zernike mode.
        
        Parameters:
        ------------
        segId : int
            segment index number [0 to 6], where 6 is the central segment.
        
        modenum : int
            segment mode number
        """
        if not hasattr(self, 'segZmat'):
            raise Exception("Compute segment Zernikes first using 'do_segment_zernikes()'...")
        
        wf = np.zeros((self._nPx**2))
        wf[self._P[segId,:]] = self.segZmat[segId][:,modenum]
        return wf.reshape((self._nPx, self._nPx))


class fitted_segment_modal_basis:
    """
    Class to generate GAPS segment modal basis (i.e. GMT segment modes fitted by the DM+PTT array).
    
    Parameters:
    -----------
    smb_obj : segment_modal_basis object
        Object representing theoretical segment modes.
    
    gaps_obj : telescope_simulator object
        Object representing the GAPS telescope simulator (i.e. DM and PTT array).
    
    dm_validacts : numpy vector
        Index vector of valid DM actuators.
    """
    def __init__(self, smb_obj, gaps_obj, dm_validacts):
        assert isinstance(smb_obj, segment_modal_basis), \
          '"smb_obj" must be an instance of "segment_modal_basis".'
        assert 'telescope_simulator' in str(type(gaps_obj)), \
          '"gaps_obj" must be an instance of "telescope_simulator".'
        self._smb = smb_obj
        self._gaps = gaps_obj
        self._validacts = dm_validacts
        self._nvacts = len(dm_validacts)
    
    
    def __get_dm_influence_matrices(self):
        DMmat, DMmat_norm, inv_DMmat = self._gaps.get_dm_influence_matrices(self._validacts)
        self._DMmat = DMmat
        self._DMmat_norm = DMmat_norm
        self._inv_DMmat = inv_DMmat
    
    
    def __get_ptt_influence_matrices(self):
        PTTmat, PTTmat_norm, inv_PTTmat = self._gaps.get_ptt_influence_matrices()
        self._PTTmat = PTTmat
        self._PTTmat_norm = PTTmat_norm
        self._inv_PTTmat = inv_PTTmat
    
    
    def __get_merged_influence_matrices(self, regularization_factor):
        MergedIFmat, inv_MergedIFmat, DMmat_norm, PTTmat_norm = \
           self._gaps.get_merged_influence_matrices(self._validacts, \
                            regularization_factor=regularization_factor)
        self._MergedIFmat = MergedIFmat
        self._inv_MergedIFmat = inv_MergedIFmat
        self._DMmat_norm = DMmat_norm
        self._PTTmat_norm = PTTmat_norm
    
    
    def fit_with_dm_only(self, nmode):
        """
        Fit the segment modes with the DM only.
        
        Parameters:
        -----------
        nmode : int
            Number of modes per segment to fit with influence functions.
        """
        assert nmode <= self._smb.n_modes_per_segment, "max number of modes: %d"%self._smb.n_modes_per_segment
        
        if not hasattr(self, "_inv_DMmat"):
            self.__get_dm_influence_matrices()
        
        self.KLF_M2C_DMonly = np.zeros((self._nvacts, nmode, 7))
        self.KLFmat_DMonly  = np.zeros((self._smb._nmask, nmode, 7))

        for segId in range(7):
            for jj in range(nmode):
                mymode = self._smb.get_segkl_wf(segId, jj)
                self.KLF_M2C_DMonly[:,jj,segId] = self._inv_DMmat @ mymode[self._smb._GMTmask2D]
                self.KLFmat_DMonly[:,jj,segId] = self._DMmat @ self.KLF_M2C_DMonly[:,jj,segId]
                
        print("--> Segment modes fitted by the DM computed successfully.")
    
    
    def fit_with_dm_and_ptt(self, nmode, regularization_factor = 5e-4):
        """
        Fit segment modes with DM + PTT array influence functions.
        
        Parameters:
        -----------
        nmode : int
            Number of modes per segment to fit with influence functions.
        
        regularization_factor : float
            regularization factor to penalize segment PTT modes fitted by DM. Default: 5e-4
        """
        assert nmode <= self._smb.n_modes_per_segment, "max number of modes: %d"%self._smb.n_modes_per_segment
        
        if not hasattr(self, "_inv_MergedIFmat"):
            self.__get_merged_influence_matrices(regularization_factor)
        
        merged_dofs = self._nvacts + 21
        self.KLF_M2C_merged = np.zeros((merged_dofs, nmode, 7))
        self.KLFmat_merged = np.zeros((self._smb._nmask, nmode, 7))

        for segId in range(7):
            for jj in range(nmode):
                mymode = self._smb.get_segkl_wf(segId, jj)
                self.KLF_M2C_merged[:,jj,segId] = self._inv_MergedIFmat @ mymode[self._smb._GMTmask2D]
                self.KLFmat_merged[:,jj,segId] = self._MergedIFmat @ self.KLF_M2C_merged[:,jj,segId]
        
        print("--> Segment modes fitted by the DM + PTT array computed successfully.")
    
    
    def fit_segptt_with_ptt(self):
        """
        Fit segment piston, tip and tilt (first three modes of basis) with the PTT array only.
        """
        if not hasattr(self, "_inv_PTTmat"):
            self.__get_ptt_influence_matrices()
        
        self.Z123_M2C = np.zeros((21, 3, 7))
        self.Z123_PTTmat = np.zeros((self._smb._nmask, 3, 7))
        for segId in range(7):
            for jj in range(3):
                mymode = self._smb.get_segkl_wf(segId, jj)
                self.Z123_M2C[:,jj,segId] = self._inv_PTTmat @ mymode[self._smb._GMTmask2D]
                self.Z123_PTTmat[:,jj,segId] = self._PTTmat @ self.Z123_M2C[:,jj,segId]
        
        print("--> Segment piston, tip, and tilt fitted by the PTT array only computed successfully.")
    
    
    def gaps_segment_modes(self, nmode, lo_mode = 66, regularization_factor = 5e-4, 
                          orthonormalize = True, descaled = False):
        """
        Generate the GAPS segment modes fitted in the following way:
          1. First three segment modes (Z1, Z2, Z3) fitted by the PTT array.
          2. Modes #4 to #<lo_mode> fitted by merged projector (using both DM and PTT array).
          3. Modes #<lo_mode> to #<nmode> fitted only by the DM.
        
        Parameters:
        -----------
        nmode : int
            Number of modes per segment to fit with influence functions.
        
        lo_mode : int
            Index of last mode to be fitted by DM+PTT array. Default: 66.
        
        regularization_factor : float
            regularization factor to penalize segment PTT modes fitted by DM.
        
        orthonormalize : bool
            if True, modes will be re-orthonormalized.
        
        descaled : bool
            if True, PTT and DM portions of the M2C matrix will be scaled back to physical units.
        """
        assert lo_mode <= nmode, "<lo_mode> must be smaller than <nmode>."
        
        if not hasattr(self, 'KLF_M2C_DMonly') and (lo_mode < nmode):
            self.fit_with_dm_only(nmode)
        if not hasattr(self, 'Z123_M2C'):
            self.fit_segptt_with_ptt()
        if not hasattr(self, 'KLF_M2C_merged'):
            self.fit_with_dm_and_ptt(nmode, regularization_factor = regularization_factor)
        
        self.KLF_M2C = np.copy(self.KLF_M2C_merged)
        self.KLFmat = np.copy(self.KLFmat_merged)
        
        #---> Z1, Z2, Z3 with PTT array only:
        self.KLF_M2C[:,0:3,:] *= 0
        self.KLF_M2C[0:21,0:3,:] = np.copy(self.Z123_M2C) * self._PTTmat_norm
        self.KLFmat[:,0:3,:] = np.copy(self.Z123_PTTmat)
        
        #---> highest-order modes with DM only:
        if lo_mode < nmode:
            self.KLF_M2C[  :, lo_mode:, :] *= 0
            self.KLF_M2C[21:, lo_mode:, :] = self.KLF_M2C_DMonly[:, lo_mode:, :] * self._DMmat_norm
            self.KLFmat[   :, lo_mode:, :] =  self.KLFmat_DMonly[:, lo_mode:, :]
        
        if orthonormalize == True:
            print("--> Re-orthogonalizing...")
            self.KLFmat, self.KLF_M2C = orthonormalize_segment_modes(self.KLFmat, self. KLF_M2C, \
                                            self._smb._npseg, ortho_modes = nmode)
        
        if descaled == True:
            normDiag = 1. / np.concatenate((np.ones(21)*self._PTTmat_norm, \
                                            np.ones(self._nvacts)*self._DMmat_norm))
            for segId in range(7):
                self.KLF_M2C[:,:,segId] = np.diag(normDiag) @ self.KLF_M2C[:,:,segId]
        
        print("--> GAPS segment modes computed successfully.")


def orthonormalize_segment_modes(KLFmat, KLF_M2C, np_per_segment, ortho_modes = None):
    """
    Perform an orthonormalization of segment modes.

    Parameters:
    -----------
    KLFmat : 3-dim numpy array
        modes influence matrix (n_points, n_modes, n_segments)
    KLF_M2C : 3-dim numpy array
        modes to commands matrix (n_actuators, n_modes, n_segments)
    np_per_segment : 7-element numpy array
        number of valid points per segment pupil
    ortho_modes : int
        Number of desired orthonormalized modes. Default: same number of input modes.
    """
    assert KLFmat.ndim == 3 and KLF_M2C.ndim == 3, "'KLFmat' and 'KLF_M2C' format not recognized. See documentation."
    assert KLFmat.shape[1] == KLF_M2C.shape[1], "'KLFmat' and 'KLF_M2C' must have the same number of modes per segment."
    assert KLFmat.shape[2] == 7 and KLF_M2C.shape[2] == 7, "'KLFmat' and 'KLF_M2C' must contain 7 modal basis. This is the GMT!"
    assert len(np_per_segment) == 7, "'n_per_segment' must be a 7-element vector. See documentation."
    
    if ortho_modes is None:
        ortho_modes = KLFmat.shape[1]
    else:
        assert ortho_modes <= KLFmat.shape[1], "max number of modes: %d"%(KLFmat.shape[1])
    
    n_points = KLFmat.shape[0]
    nvacts = KLF_M2C.shape[0]
    KLFmato = np.zeros((n_points, ortho_modes, 7))
    KLFo_M2C = np.zeros((nvacts, ortho_modes, 7))

    for segId in range(7):
        KLF_Dmat = np.matmul(np.transpose(KLFmat[:,0:ortho_modes,segId]), KLFmat[:,0:ortho_modes,segId]) /np_per_segment[segId]
        KLF_Lmat = np.linalg.cholesky(KLF_Dmat)
        KLF_inv_Lmat = np.linalg.pinv(KLF_Lmat)
        KLFmato[:,:,segId] = np.matmul(KLFmat[:,0:ortho_modes,segId], np.transpose(KLF_inv_Lmat))
        KLFo_M2C[:,:,segId] = np.matmul(KLF_M2C[:,0:ortho_modes,segId], np.transpose(KLF_inv_Lmat))
    
    return KLFmato, KLFo_M2C

        
        