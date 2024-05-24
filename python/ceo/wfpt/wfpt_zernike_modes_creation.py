from ceo import PhaseProjectionSensor
from ceo.wfpt import wfpt_simul, wfpt_utilities
import numpy as np
import os

class wfpt_zernike_modes_creation:
    """
    A class to produce segment Zernike modes fitted by the WFPT M1 or M2 actuators.
    
    Parameters:
    -----------
    M2_baffle_diam : float
        Diameter of M2 baffle [m]. Default: 3.6
    project_truss_onaxis : bool
        If True, simulates truss shadows. Default: True
    path : string
        WFPT path to retrieve wavefront in exit pupil. Either "SH" or "DFS". Default: 'SH'
    """
    def __init__(self, M2_baffle_diam=3.6, project_truss_onaxis=True, path='SH'):
        assert path in ['SH', 'DFS'], '"path" must be either "SH" or "DFS"'
        
        # ---- Initialize the WFPT model
        self.wfpt = wfpt_simul(M2_baffle_diam=M2_baffle_diam, project_truss_onaxis=project_truss_onaxis)
        self.wfpt.calibrate_sensors(keep_rays_for_plot=False)
        
        # ---- Retrieve path and source to use
        self._path_ = path
        if path=='SH':
            self.path = self.wfpt.shs_path
            self.src = self.wfpt.shs_src
        elif path=='DFS':
            self.path = self.wfpt.dfs_path
            self.src = self.wfpt.dfs_src
        
        #---- Retrieve GMT mask and segment masks
        self.wfpt.reset()
        self.path.propagate(self.src)
        self.GMTmask = self.src._gs.amplitude.host().astype('bool')
        self.nmask = np.sum(self.GMTmask)
        self.segmask = np.array([vec.get() for vec in self.src.piston_mask])
        self.npseg = np.sum(self.segmask, axis=1).astype('int')
        
        #---- Allocations
        self.IFmats = {}
        self.projMats = {}
        self.fitZern = {}


    def get_zernikes(self, zernikes_radial_order=4, CS_rotation=True):
        """
        Get theoretical segment Zernike modes in matrix form.
        
        Parameters:
        -----------
        zernike_radial_order : int
            Last radial order of Zernikes to generate. Default: 4 (i.e. 15 Zernike modes)
        CS_rotation : bool
            If True, define segment Zernikes following the GMT segment LCS system.
        """
        zsensor = PhaseProjectionSensor(zernikes_radial_order)
        
        #---- Generate theoretical Zernike modes
        rot_angle = (self.src._rays_rot_angle + 90) * (np.pi/180)
        zsensor.calibrate(self.src._gs, piston_mask=[self.segmask], CS_rotation=CS_rotation, 
                          rot_angle=rot_angle)
        
        #---- Re-orthonormalize zernike modes over segment pupils.
        Zmat = np.zeros((self.nmask,zsensor.n_mode,7))

        for segid in range(7):
            Zmat1 = np.squeeze(zsensor.Zmat[:,:,segid])
            Dmat1 = np.matmul(np.transpose(Zmat1), Zmat1) / self.npseg[segid]
            Lmat1 = np.linalg.cholesky(Dmat1)
            inv_Lmat1 = np.linalg.pinv(Lmat1)
            Zmato = np.matmul(Zmat1, np.transpose(inv_Lmat1))
            Zmat[:,:,segid] = Zmato[self.GMTmask.ravel(),:]
        
        self.CS_rotation = CS_rotation
        self.Zmat = Zmat
        print("--> Theoretical Zernikes computed successfully.")


    def get_influence_matrices(self, mirror):
        """
        Get the influence matrices.
        
        Parameters:
        -----------
        mirror : string
            Either 'M1' or 'M2'.        
        """
        assert mirror in ['M1', 'M2'], '"mirror" must be either "M1" or "M2"'
        
        self.IFmats[mirror] = \
            {'SPP_IFmat'  : self.wfpt.influence_matrix(path=self._path_, mirror=mirror, 
                                                mode='segment piston', stroke=5e-8),
             'RxRy_IFmat' : self.wfpt.influence_matrix(path=self._path_, mirror=mirror, 
                                                mode='segment tip-tilt', stroke=1e-7),
             'DM_IFmat'   : self.wfpt.influence_matrix(path=self._path_, mirror=mirror, 
                                                mode='actuators', stroke=0.1)}


    def get_dm_influence_matrix(self, mirror, dm_valid_acts_file=None):
        """
        Get the DM influence matrix.
        
        Parameters:
        -----------
        mirror : string
            Either 'M1' or 'M2'.
        dm_valid_acts_file : string
            Name of file that contains the DM valid actuators. Default: None            
        """
        if not mirror in self.IFmats:
            self.get_influence_matrices(mirror)
        
        #--- Get the slaved DM_IFmat with only valid actuators present.
        if not dm_valid_acts_file is None:
            self.IFmats[mirror]['dm_valid_actuators'] = wfpt_utilities.get_dm_valid_actuators(dm_valid_acts_file)
            DMmat = self.IFmats[mirror]['DM_IFmat'] @ self.IFmats[mirror]['dm_valid_actuators']['slaveMat']
            DMmat = DMmat[:, self.IFmats[mirror]['dm_valid_actuators']['valid_acts']]
        else:
            DMmat = self.IFmats[mirror]['DM_IFmat']
        
        return DMmat

    
    def get_ptt_influence_matrix(self, mirror, ptt_valid_segments=[0,1,2,3,4,5,6]):
        """
        Get the PTT array influence matrix.
        
        Parameters:
        -----------
        mirror : string
            Either 'M1' or 'M2'.
        ptt_valid_segments : list
            List of active PTT array segments. Default: all
            Note: segment numbering in GMT convention.
        """
        if not mirror in self.IFmats:
            self.get_influence_matrices(mirror)
        #--- Get the SPP IFmat
        SPP_IFmat  = self.IFmats[mirror]['SPP_IFmat'][:,ptt_valid_segments]
        #--- Get the RxRy IFmat
        ptt_valid_tts = np.concatenate((np.array(ptt_valid_segments), np.array(ptt_valid_segments)+7)).tolist()
        RxRy_IFmat = self.IFmats[mirror]['RxRy_IFmat'][:,ptt_valid_tts]
        #--- Update list of valid DoFs
        ptt_valid_dofs = np.zeros(21, dtype='bool')
        ptt_valid_dofs[ptt_valid_segments] = True
        ptt_valid_dofs[np.array(ptt_valid_tts)+7] = True
        self.IFmats[mirror]['ptt_valid_actuators'] = {'ptt_valid_segments' : ptt_valid_segments,
                                                      'ptt_valid_tiptilts' : ptt_valid_tts,
                                                      'ptt_valid_dofs'   : ptt_valid_dofs,
                                                      'n_ptt_valid_dofs' : np.sum(ptt_valid_dofs) }
        return SPP_IFmat, RxRy_IFmat
        

    def get_merged_influence_matrix(self, mirror, dm_valid_acts_file=None, ptt_valid_segments=[0,1,2,3,4,5,6]):
        """
        Get the merged influence matrix.
        
        Parameters:
        -----------
        mirror : string
            Either 'M1' or 'M2'.
        dm_valid_acts_file : string
            Name of file that contains the DM valid actuators. Default: None            
        """
        if not mirror in self.IFmats:
            self.get_influence_matrices(mirror)
        
        #--- Get the slaved DM_IFmat with only valid actuators present.
        DMmat = self.get_dm_influence_matrix(mirror, dm_valid_acts_file)
        
        #--- Get the PTT SPP and STT Influence matrices (using only valid PTT segments)
        SPP_IFmat, RxRy_IFmat = self.get_ptt_influence_matrix(mirror, ptt_valid_segments=ptt_valid_segments)
        
        #--- Norm-weighted merged IFmat
        DMmat_norm  = np.linalg.norm(DMmat)
        SPP_IFmat_norm = np.linalg.norm(SPP_IFmat)
        RxRy_IFmat_norm = np.linalg.norm(RxRy_IFmat)
        PTTmat = np.concatenate((SPP_IFmat/SPP_IFmat_norm, RxRy_IFmat/RxRy_IFmat_norm), axis=1)
        mergedIFmat = np.concatenate((SPP_IFmat/SPP_IFmat_norm, RxRy_IFmat/RxRy_IFmat_norm, DMmat/DMmat_norm), axis=1)
        self.IFmats[mirror]['PTTmat'] = PTTmat
        self.IFmats[mirror]['mergedIFmat'] = mergedIFmat
        self.IFmats[mirror]['mergedIFmat_norms'] = {'DMmat_norm': DMmat_norm, 'SPP_IFmat_norm': SPP_IFmat_norm, 'RxRy_IFmat_norm': RxRy_IFmat_norm}
        print("--> Merged IFmat computed successfully.")


    def get_projection_matrix(self, mirror, dm_valid_acts_file=None, ptt_valid_segments=[0,1,2,3,4,5,6], projMat_type='merged simple LS', svd_thr=1e-15,
                             independent_PTT_projection=True, regularization_factor=5e-2):
        """
        Computes the matrix that projects segment Zernike modes onto the WFPT active mirrors.
        
        Parameters:
        -----------
        mirror : string
            Either 'M1' or 'M2'.
        dm_valid_acts_file : string
            Name of file that contains the DM valid actuators. Default: None
        ptt_valid_segments : list
            List of active PTT array segments. Default: all
            Note: segment numbering in GMT convention.
        projMat_type : str
            Identify the type of projection. Default: 'merged simple LS'.
                Types supported:
                    'merged simple LS' : projector is computed as the generalized inverse of the merged IF matrix 'mergedIFmat'.
                                         SVD filtering is possible using 'svd_thr' parameter.
                    'merged regularized LS' : projector that penalizes the DM for creating segment piston and segment TT modes.
                                              To be used with the "regularization factor" parameter.
        svd_thr : float
            SVD threshold value to be applied in the computation of the generalized inverse (when projMat_type is 'merged simple LS').
        regularization_factor : float
            Regularization factor applied to the regularization term (when projMat_type is 'merged regularized LS').
        independent_PTT_projection : bool
            If True, segment Zernikes Z1, Z2, Z3 will be fitted using only the PTT array Influence Functions.
        """
        self.get_merged_influence_matrix(mirror, dm_valid_acts_file, ptt_valid_segments=ptt_valid_segments)
        IFmat = self.IFmats[mirror]['mergedIFmat']
        
        if projMat_type == 'merged simple LS':
            inv_IFmat = np.linalg.pinv(IFmat, rcond=svd_thr)
        
        elif projMat_type == 'merged regularized LS':
            # 1.--- Compute the inverse of the DM IF matrix
            DMmat = self.get_dm_influence_matrix(mirror, dm_valid_acts_file)
            inv_DMmat = np.linalg.pinv(DMmat)
            # 2.---- Get the segment Z1-Z3 modes (for valid PTT segments only)
            n_ptt_valid_dofs   = self.IFmats[mirror]['ptt_valid_actuators']['n_ptt_valid_dofs']
            ptt_valid_segments = self.IFmats[mirror]['ptt_valid_actuators']['ptt_valid_segments']
            n_ptt_valid_segments = len(ptt_valid_segments)
            Zpttmat = np.zeros((self.nmask, n_ptt_valid_dofs))
            for idx, segid in enumerate(ptt_valid_segments):
                Zpttmat[:,idx] = self.Zmat[:,0,segid]
                Zpttmat[:,idx+n_ptt_valid_segments]   = self.Zmat[:,1,segid]
                Zpttmat[:,idx+n_ptt_valid_segments*2] = self.Zmat[:,2,segid]
            # 3.--- DM best-fit to Z1-Z3 modes
            sptt_dm_comm = inv_DMmat @ Zpttmat
            sptt_dm_comm /= np.max(np.abs(sptt_dm_comm))
            #self.sptt_dm_comm = sptt_dm_comm #--- debugging
            # 4.--- Compute the regularization term
            sptt_dm_comm1 = np.concatenate( (np.zeros((n_ptt_valid_dofs,n_ptt_valid_dofs)), sptt_dm_comm) )
            Wsptt = sptt_dm_comm1 @ sptt_dm_comm1.T
            # 4.--- Compute regularized merged projector
            inv_IFmat = np.linalg.solve(IFmat.T @ IFmat + regularization_factor*Wsptt, IFmat.T)
        
        self.projMats[mirror] = {'merged': inv_IFmat, 'projMat_type': projMat_type, 'svd_thr': svd_thr, 
                                 'independent_PTT_projection': independent_PTT_projection,
                                'regularization_factor': regularization_factor}
        
        if independent_PTT_projection:
            PTTmat = self.IFmats[mirror]['PTTmat']
            inv_PTTmat = np.linalg.pinv(PTTmat)            
            self.projMats[mirror]['inv_PTTmat'] = inv_PTTmat
        
        print("--> Projection Matrix computed successfully.")


    def get_fitted_zernikes(self, mirror):
        """
        Compute Zernike modes fitted by the WFPT active mirrors (PTT array + DM).
        
        Parameters:
        -----------
        mirror : string
            Either 'M1' or 'M2'.
        """
        if not mirror in self.projMats:
            print("No Projection matrix available. Compute it with 'get_projection_matrix()'...")
            return
        if not hasattr(self, 'Zmat'):
            print("Compute theoretical segment Zernikes first using 'get_zernikes()'...")
            return
        
        IFmat = self.IFmats[mirror]['mergedIFmat']
        inv_IFmat = self.projMats[mirror]['merged']
        ndof = inv_IFmat.shape[0]
        nzern = self.Zmat.shape[1]
        
        _Zmat_M2C_ = np.zeros((ndof, nzern, 7))
        Zmat_bf = np.zeros((self.nmask,nzern,7))
        
        for segid in range(7):
            _Zmat_M2C_[:,:,segid] = inv_IFmat @ self.Zmat[:,:,segid]
            Zmat_bf[:,:,segid] = IFmat @ _Zmat_M2C_[:,:,segid]
        
        if self.projMats[mirror]['independent_PTT_projection'] == True:
            PTTmat = self.IFmats[mirror]['PTTmat']
            n_ptt_valid_dofs = self.IFmats[mirror]['ptt_valid_actuators']['n_ptt_valid_dofs']
            ptt_valid_segments = self.IFmats[mirror]['ptt_valid_actuators']['ptt_valid_segments']
            inv_PTTmat = self.projMats[mirror]['inv_PTTmat']
            for segid in ptt_valid_segments:
                _Zmat_M2C_[:,0:3,segid] *= 0
                _Zmat_M2C_[0:n_ptt_valid_dofs,0:3,segid] = inv_PTTmat @ self.Zmat[:,0:3,segid]
                Zmat_bf[:,0:3,segid] = PTTmat @ _Zmat_M2C_[0:n_ptt_valid_dofs,0:3,segid]
        
        #--- Compute full-sized and de-normalized M2C
        DMmat_norm  = self.IFmats[mirror]['mergedIFmat_norms']['DMmat_norm']
        SPP_IFmat_norm = self.IFmats[mirror]['mergedIFmat_norms']['SPP_IFmat_norm']
        RxRy_IFmat_norm = self.IFmats[mirror]['mergedIFmat_norms']['RxRy_IFmat_norm']
        n_valid_acts = self.IFmats[mirror]['dm_valid_actuators']['n_valid_acts']
        valid_acts   = self.IFmats[mirror]['dm_valid_actuators']['valid_acts']
        slaveMat     = self.IFmats[mirror]['dm_valid_actuators']['slaveMat']
        ptt_valid_dofs   = self.IFmats[mirror]['ptt_valid_actuators']['ptt_valid_dofs']
        #n_ptt_valid_dofs = self.IFmats[mirror]['ptt_valid_actuators']['n_ptt_valid_dofs']
        n_ptt_valid_segments = len(self.IFmats[mirror]['ptt_valid_actuators']['ptt_valid_segments'])
                                
        valid_dofs = np.concatenate( (ptt_valid_dofs, valid_acts) )
        #normDiag = 1. / np.concatenate( (np.ones(n_ptt_valid_dofs)*PTTmat_norm, np.ones(n_valid_acts)*DMmat_norm) )
        normDiag = 1. / np.concatenate( (np.ones(n_ptt_valid_segments)*SPP_IFmat_norm,
                                         np.ones(n_ptt_valid_segments*2)*RxRy_IFmat_norm,
                                         np.ones(n_valid_acts)*DMmat_norm) )
        
        Zmat_M2C = np.zeros((21+292,nzern,7))
        for segid in range(7):
            Zmat_M2C[valid_dofs,:,segid] = np.diag(normDiag) @ _Zmat_M2C_[:,:,segid]
            Zmat_M2C[21:,:,segid] = slaveMat @ Zmat_M2C[21:,:,segid]
        
        self.fitZern[mirror] = {'M2C': Zmat_M2C, 'Zmat': Zmat_bf}
        print("--> Fitted Zernikes computed successfully.")


    def save_zern_m2c(self, mirror, fname):
        """
        Save Zernike Modes-to-commands matrix to dedicated repository in WFPT_model_data/M2C/
        
        Parameters:
        -----------
        fname : string
            Name of M2C file.
        """
        here = os.path.abspath(os.path.dirname(__file__))
        fullname = os.path.join(here, 'WFPT_model_data', 'M2C', fname)
        
        tosave = dict(mirror=mirror, modes_type='segment Zernikes', LCS_rotation=self.CS_rotation,
                    dm_valid_acts_file=self.IFmats[mirror]['dm_valid_actuators']['filename'],
                    projMat_type=self.projMats[mirror]['projMat_type'], svd_thr=self.projMats[mirror]['svd_thr'],
                    independent_PTT_projection=self.projMats[mirror]['independent_PTT_projection'],
                    M2C=self.fitZern[mirror]['M2C'])
        
        np.savez(fullname, **tosave)
        print("Saving to file %s"%fullname)
        