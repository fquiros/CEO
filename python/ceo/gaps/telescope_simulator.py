import numpy as np
from gmt_pupil import gmt_pupil
from mems_model import mems_model
from ptt_array_model import ptt_array_model

class telescope_simulator:
    """
    Class that simulates the GAPS telescope simulator (DM and PTT array).
    
    Parameters:
    -----------
    
    1) Parameters to create the GMT pupil and PTT array:
    
        array_size_pix : int
            Size in pixels of simulated array containing the GMT pupil.

        array_size_m : float
            Size in meters of simulated array containing the GMT pupil (should be slightly larger than GMT diameter). Default: 25.5 m

        array_rot_angle : float
            Angle of rotation in degrees of the GMT pupil. Default: 0 deg

        project_truss_onaxis : bool
            If True, truss shadow will be applied to central pupil mask. Default: False
    
    2) Parameters to create the DM model:
    
        mems_ifunc_fname : str
            Name of .npz file containing the reference influence function (see mems_model class for details).
        
        pupil_size_in_mems_pitches : float
            Size of GMT pupil (in pitches). Default: 48
        
        mems_grid_rot_angle : float
            Angle of rotation in degrees of DM grid w.r.t. GMT pupil. Default: 0 deg
    """
    def __init__(self, array_size_pix, mems_ifunc_fname, array_size_m = 25.5, array_rot_angle = 0.0, 
                 project_truss_onaxis = False, pupil_size_in_mems_pitches = 48, mems_grid_rot_angle = 0.0):
        
        #---> GMT pupil
        print("--> Initializing GMT pupil.....")
        array_pixscale = array_size_m / (array_size_pix-1)
        self.pup = gmt_pupil(array_size_pix, array_size_m=array_size_m, array_rot_angle=array_rot_angle,
                       project_truss_onaxis=project_truss_onaxis)
        self.pup.cleanup()

        #---> MEMS model
        print("--> Initializing MEMS model.....")
        act_mask_fname = './data/mems2k/BMC_2k_act_mask.npz'
        Dtel = 25.4 #m
        act_pitch_m = Dtel / pupil_size_in_mems_pitches
        self.mems2k = mems_model(mems_ifunc_fname, act_mask_fname, act_pitch_m=act_pitch_m, 
                                 grid_rot_deg=mems_grid_rot_angle-array_rot_angle)
        self.mems2k.compute_if_cube(array_size_pix, array_size_m)

        #---> PTT array model
        print("--> Initializing PTT array model.....")        
        self.ptt = ptt_array_model(array_size_pix, array_size_m=array_size_m, array_rot_angle=array_rot_angle)
        
        print("Done! .....")
    
    
    def get_wf(self, dm_command=None, ptt_command=None):
        """
        Apply DM and/or PTT command to the GAPS telescope simulator, and return the WF.
        
        Parameters:
        -----------
        dm_command : numpy array [2040]
            DM command vector [m WF]
        
        ptt_command : numpy array [21]
            PTT command vector [m WF]
        
        Returns:
        --------
        wavefront : numpy array (nPx,nPx)
            Sum of DM and PTT array wavefronts.
        """
        if dm_command is not None:
            dmwf = self.mems2k.get_wf(dm_command)
        else:
            dmwf = np.zeros((self.pup.nPx, self.pup.nPx))
        if ptt_command is not None:
            pttwf = self.ptt.get_wf(ptt_command)
        else:
            pttwf = np.zeros((self.pup.nPx, self.pup.nPx))
        
        return (dmwf + pttwf)*self.pup.GMTmask2D

    
    def get_dm_valid_actuators(self, threshold=None):
        """
        Get MEMS DM illuminated actuators within the pupil mask.
        
        Parameters:
        -----------
        threshold : float
            Illumination threshold used to identify illuminated (i.e. valid) actuators.
        
        Returns:
        --------
            1. Valid actuator index vector
            2. Illumination peak value vector representing actuator visibility within GMT pupil.
        """
        assert threshold is not None, "threshold must be defined and be 0.0 <= thr <= 1.0" 
        validacts, ifpeak = self.mems2k.get_valid_actuators(self.pup.GMTmask2D, threshold=threshold)
        return validacts, ifpeak
    
    
    def get_dm_influence_matrices(self, validacts, rcond=1e-15, silent=False):
        """
        Computes the DM Influence Matrix, its norm, and its inverse.
        
        Parameters:
        -----------
        validacts : numpy array (int)
            DM valid actuators index vector.
            
        rcond : float
            SVD threshold for computation of generalized inverse of DMmat. Default: 1e-15
        
        Returns:
        --------
            1. DMmat : DM Influence Matrix
            2. DMmat_norm : norm of DMmat
            3. inv_DMmat : generalized inverse of DMmat
        """
        if not silent:
            print("--> Computing DM influence matrices.....")
        DMmat = (self.mems2k.IFcube[:,:,validacts].reshape((-1,len(validacts))))[self.pup.GMTmask,:]
        if not silent:
            print('DMmat condition number: %f'%np.linalg.cond(DMmat))
        DMmat_norm  = np.linalg.norm(DMmat)
        inv_DMmat = np.linalg.pinv(DMmat, rcond=rcond)
        return DMmat, DMmat_norm, inv_DMmat
    
    
    def get_ptt_influence_matrices(self, rcond=1e-15, silent=False):
        """
        Computes the PTT Influence Matrix, its norm, and its inverse.
        
        Parameters:
        -----------
        rcond : float
            SVD threshold for computation of generalized inverse of PTTmat. Default: 1e-15
        
        Returns:
        --------
            1. PTTmat : PTT Influence Matrix
            2. PTTmat_norm : norm of PTTmat
            3. inv_PTTmat : generalized inverse of PTTmat
        """
        if not silent:
            print("--> Computing PTT influence matrices.....")
        PTTmat = np.zeros((self.pup.nmask, 7*3))
        for jj in range(7*3):
            PTTmat[:,jj] = self.ptt.IFcube[:,:,jj][self.pup.GMTmask2D]
        if not silent:
            print('PTTmat condition number: %f'%np.linalg.cond(PTTmat))
        PTTmat_norm = np.linalg.norm(PTTmat)
        inv_PTTmat = np.linalg.pinv(PTTmat, rcond=rcond)
        return PTTmat, PTTmat_norm, inv_PTTmat
    
    
    def get_merged_influence_matrices(self, validacts, dm_rcond=1e-15, ptt_rcond=1e-15, 
                regularization_factor=5e-4, get_only_descaled_merged_ifmat=False):
        """
        Computes the merged (DM+PTT) Influence Matrix, and its regularized inverse.
        Note: the regularization factor is applied to a term that penalizes DM commands reproducing PTT influence functions (i.e. segment piston, tip, and tilt).
        
        Parameters:
        -----------
        validacts : numpy array (int)
            DM valid actuators index vector.
            
        dm_rcond : float
            SVD threshold for computation of generalized inverse of DMmat. Default: 1e-15
        
        ptt_rcond : float
            SVD threshold for computation of generalized inverse of PTTmat. Default: 1e-15
        
        regularization_factor : float
            Factor applied to the penalizing term. Default: 5e-4
        
        Returns:
        --------
            1. Merged Influence Matrix (PTT, DM)
            2. Regularized inverse of merged influence matrix
            3. DMmat_norm : norm of DMmat
            4. PTTmat_norm : norm of PTTmat
        """
        print("--> Computing Merged influence matrices.....")
        DMmat, DMmat_norm, inv_DMmat = self.get_dm_influence_matrices(validacts, rcond=dm_rcond, silent=True)
        PTTmat, PTTmat_norm, inv_PTTmat = self.get_ptt_influence_matrices(rcond=ptt_rcond, silent=True)
        
        #---> Return only merged influence matrix
        if get_only_descaled_merged_ifmat == True:
            return np.concatenate((PTTmat, DMmat), axis=1)
        
        #---> Merged influence matrix, weighted by the norms.
        mergedIFmat = np.concatenate((PTTmat/PTTmat_norm, DMmat/DMmat_norm), axis=1)
        
        #--- DM best-fit to PTT modes
        sptt_dm_comm = inv_DMmat @ PTTmat
        sptt_dm_bf = DMmat @ sptt_dm_comm
        sptt_dm_comm /= np.max(np.abs(sptt_dm_comm))
        sptt_dm_comm1 = np.concatenate( (np.zeros((21,21)), sptt_dm_comm) )
        Wsptt = sptt_dm_comm1 @ sptt_dm_comm1.T
        
        #--- Regularized inverse of merged influence matrix
        reg_CP_mat = mergedIFmat.T @ mergedIFmat + regularization_factor*Wsptt
        inv_mergedIFmat = np.linalg.solve(reg_CP_mat, mergedIFmat.T)
        
        return mergedIFmat, inv_mergedIFmat, DMmat_norm, PTTmat_norm
        
        
