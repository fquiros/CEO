import numpy as np
import os
import sys

from gaps_simul import gaps_simul
from ngao_controller import ao_controller, hdfs_controller, interaction_matrix
from wf_combiner import wf_combiner
from gaps_utilities import load_dictionary_from_file

class zonal_control_test(gaps_simul):
    """
    Zonal control test with PWFS controlling the DM.
    """
    def configure(self, pure_delay=1, vact_thr=0.4, act_amp_wf=100e-9, spp_amp=1e-6,
                 global_pist_reg_factor=1e11):
        """
        Set Zonal Control with the DM only, driven by the PWFS
        
        Parameters:
        -----------
        pure_delay : int
            Pure (discrete) delay, in number of frames. Default: 1
         
        vact_thr : float
            influence function peak value threshold to select illuminated actuators.
            Default : 0.4
        
        act_amp_wf : float
            amplitude applied to DM actuators during interaction matrix calibration [m wf].
            Default: 100e-9
        
        spp_amp : float
            segment piston amplitude applied during HDFS interaction matrix calibration [m wf].
            Default: 1e-6
            NOTE: Segment piston is produced with a best-fit DM command.
        
        global_pist_reg_factor : float
            global piston penalizing factor for the HDFS reconstructor. Default: 1e11
        """
        
        #--> Zonal interaction matrix and reconstructor computation:
        self.calib_repo['zonal-dm-pwfs'] = {}
        self._get_zonal_interaction_matrix(vact_thr, act_amp_wf)
        self._do_svd_and_compute_reconstructor()
        validacts = self.dm_valid_acts_params['dm_valid_acts']
        
        #--> HDFS interaction matrix and reconstructor computation:
        self.calib_repo['spp-hdfs'] = {}
        self._get_hdfs_matrices(spp_amp, global_pist_reg_factor)
        
        #--> Setup atmospheric turbulence projection:
        #mergedIFmat, inv_mergedIFmat = self.tel.get_merged_influence_matrices(validacts)
        #self.atm.set_wavefront_projection(mergedIFmat, inv_mergedIFmat)
        self.atm.set_wavefront_projection(self.tel.DMmat, self.tel.inv_DMmat)
        
        #--> Setup AO controller:
        self.ao_ctrl = ao_controller(self.calib_repo['zonal-dm-pwfs']['recmat'],
                        pure_delay=pure_delay, modal_control=False,
                        Pp2m = self.calib_repo['spp-hdfs']['Pp2m'], 
                        T_out=self.pwfs.T_out, T_d=self.pwfs.T_d)
        
        #--> Setup HDFS segment piston controller (baseline mode):
        self.hdfs_ctrl = hdfs_controller(self.calib_repo['spp-hdfs']['recmat'],
                                         operation_mode = 'baseline', 
                                         T_out = self.hdfs.T_out,
                                         T_d = self.hdfs.T_d)
        
        #--> Setup WF combiner:
        mergedIFmat_descaled = self.tel.get_merged_influence_matrices(validacts,
                        get_only_descaled_merged_ifmat=True)
        self.wf_ctrl = wf_combiner(self.tel.pup, mergedIFmat_descaled)
        
        #--> Connect system components:
        #- Feed WF to WFS
        self.pwfs.register_input_method(self.wf_ctrl.get_wavefront)
        self.hdfs.register_input_method(self.wf_ctrl.get_wavefront)
        #- Feed WFS measurements to controllers
        self.hdfs_ctrl.register_input_method(self.hdfs.get_measurement)
        self.ao_ctrl.register_input_method(self.pwfs.get_measurement,
                                           self.hdfs_ctrl.get_piston_command)
        #- Feed PTT and DM commands (+ atmospheric WF) to WF combiner 
        self.wf_ctrl.register_input_method(self.ao_ctrl.get_ptt_command, 
                                           self.ao_ctrl.get_dm_command,
                                           get_atmo_wf=self.atm.get_data)
        
        #--> Populate Component list (remember: the order matters!!!)
        self.Components = [self.atm, self.wf_ctrl, self.pwfs, self.hdfs, self.hdfs_ctrl, self.ao_ctrl]
    
    
    def _get_zonal_interaction_matrix(self, vact_thr, amp_wf):
        """
        Load from file, or compute relevant interaction matrices.
        """
        #---> Load or compute zonal DM-PYR interaction matrix
        sys.stdout.write("Zonal calibration between DM and PWFS in progress....\n")
        
        #-- Name of interaction matrix file:
        here = os.path.abspath(os.path.dirname(__file__))
        intmat_fname = "ZONAL_PYR_DM_mod%0.1f_vact_thr_%0.2f_gridrot_%0.2f.npz"%( \
                    self.pwfs.modulation, vact_thr,
                    self.simul_params['mems_params']['mems_grid_rot_angle'])
        intmat_full_fname = os.path.join(here, 'data', 'intmats', intmat_fname)
        
        #-- Load interaction matrix from file:
        try:
            data = load_dictionary_from_file(intmat_full_fname)
            intmat = data['intmat']
            self.dm_valid_acts_params = data['dm_valid_acts_params']
            validacts = self.dm_valid_acts_params['dm_valid_acts']
        
        #-- If no file exists, compute the interaction matrix and save it:
        except FileNotFoundError:
            
            #-- Select DM valid actuators:
            self.dm_valid_acts_params = self.tel.get_dm_valid_actuators(threshold = vact_thr)
            validacts = self.dm_valid_acts_params['dm_valid_acts']
            
            #-- Ensure intmat calibration is done without measurement noise:
            simul_noise_status = self.pwfs.simul_noise # ensure noiseless calibration.
            self.pwfs.wfs.simul_noise = False
            
            #-- Calibrate interaction matrix and save it:
            intmat = interaction_matrix(self.pwfs, \
                self.tel.mems2k.IFcube * self.tel.pup.GMTmask2D[:,:,np.newaxis], \
                validacts, amp_wf=amp_wf)
            tosave = dict(intmat=intmat, dm_valid_acts_params=self.dm_valid_acts_params)
            np.savez(intmat_full_fname, **tosave)
            
            #-- Restore original noise setup:
            self.pwfs.wfs.simul_noise = simul_noise_status
            
        #-- Update Calibration Repository:
        self.calib_repo['zonal-dm-pwfs']['intmat'] = intmat
    
    
    def _do_svd_and_compute_reconstructor(self):
        """
        Do SVD analysis of interaction matrix, and compute LS reconstructor.
        """
        intmat = self.calib_repo['zonal-dm-pwfs']['intmat']
        validacts = self.dm_valid_acts_params['dm_valid_acts']
        
        #--> Singular Value Decomposition:
        sys.stdout.write("Starting SVD analysis\n")
        UU1, ss1, VVT1 =np.linalg.svd(intmat)
        
        #--> Compute shape of last eigenmode
        last_eigenvec = np.copy(VVT1[-1,:])
        dmvec = np.zeros(self.tel.mems2k.n_acts)
        dmvec[validacts] = last_eigenvec
        wf_last_eigenmode = self.tel.get_wf(dm_command=dmvec)
        
        #--> Select threshold to filter last eigenmode (associated with global piston)
        svd_thr = (ss1/np.max(ss1))[-2:].sum()/2
        sys.stdout.write("LS reconstructor with %d eigenmodes filtered.\n"%np.sum(ss1/np.max(ss1) < svd_thr))
        recmat = np.linalg.pinv(intmat, rcond=svd_thr)
        
        #--> Update Calibration Repository:
        self.calib_repo['zonal-dm-pwfs'].update({'sv': ss1, 
                                    'wf_last_eigenmode': wf_last_eigenmode,
                                    'svd_thr': svd_thr,
                                    'recmat': recmat})
    
    
    def _get_hdfs_matrices(self, amp_wf=1e-6, global_pist_reg_factor=1e11):
        """
        Get HDFS interaction matrix and reconstructor.
        Note: Segment piston modes are produced with DM best-fit commands!
        """
        sys.stdout.write("HDFS calibration of DM best-fit segment piston in progress...\n")
        #--> Segment piston to DM actuators Projection Matrix
        validacts = self.dm_valid_acts_params['dm_valid_acts']
        
        if not hasattr(self.tel, 'DMmat'):
            self.tel.compute_dm_influence_matrices(validacts, silent=True)
        
        if not hasattr(self.tel, 'PTTmat'):
            self.tel.compute_ptt_influence_matrices(silent=True)
        
        spp_dm_comm = self.tel.inv_DMmat @ self.tel.PTTmat[:,0:7]
        spp_dm_bf = self.tel.DMmat @ spp_dm_comm
        self.calib_repo['spp-hdfs']['Pp2m'] = spp_dm_comm
        
        #--> Best-fit Segment Piston Influence Matrix
        nPx = self.tel.pup.nPx
        spp_dm_bf_cube = np.zeros((nPx,nPx,7))
        wftemp = np.zeros((nPx,nPx))
        for segId in range(7):
            wftemp[self.tel.pup.GMTmask2D] = spp_dm_bf[:, segId]
            spp_dm_bf_cube[:,:,segId] = wftemp
        
        #--> HDFS interaction matrix
        simul_noise_status = self.hdfs.simul_noise
        self.hdfs.wfs.simul_noise = False
        D_DM_HDFS = interaction_matrix(self.hdfs, spp_dm_bf_cube, np.arange(7), amp_wf=amp_wf)
        self.calib_repo['spp-hdfs']['intmat'] = D_DM_HDFS
        self.hdfs.wfs.simul_noise = simul_noise_status
        
        #--> SVD analysis
        sys.stdout.write("Computing HDFS Reconstructor...\n")
        UU1, ss1, VVT1 =np.linalg.svd(D_DM_HDFS)
        
        #--> Compute shape of last eigenmode
        wf_last_eigenmode = np.zeros((nPx,nPx))
        wf_last_eigenmode[self.tel.pup.GMTmask2D] = spp_dm_bf @ VVT1[-1,:]
        
        #--> Regularization penalizing global piston:
        HDFS_DM_reg_mat = D_DM_HDFS.T @ D_DM_HDFS + global_pist_reg_factor * np.ones((7,7))
        UU2, ss2, VVT2 = np.linalg.svd(HDFS_DM_reg_mat, full_matrices=False, hermitian=True)
        
        #--> Regularized reconstruction matrix
        R_DM_HDFS = np.linalg.solve(HDFS_DM_reg_mat, D_DM_HDFS.T)
        
        #--> Update Calibration Repository:
        self.calib_repo['spp-hdfs'].update({'sv': ss1, 
                                    'reg_sv': ss2,
                                    'wf_last_eigenmode': wf_last_eigenmode,
                                    'reg_factor': global_pist_reg_factor,
                                    'recmat': R_DM_HDFS})
    
    
    def update_hdfs_rate(self, T_out: float, T_d: float, ctrl_wait_n_frames: int = 0):
        """
        Update the HDFS (sensor and controller) timing settings.
    
        Parameters:
        -----------
        T_out : float
            Integration time [s].
        
        T_d : float
            Time delay for the HDFS to start operation [in seconds].
    
        ctrl_wait_n_frames : int
            Number of extra HDFS frames that the controller waits before computing commands. Default: 0
        """
        # --The HDFS integration time / HDFS controller's rate must be the same:
        self.hdfs.T_out = T_out
        self.hdfs_ctrl.T_out = T_out
        self.hdfs.T_d = T_d
        self.hdfs_ctrl.T_d = T_d + T_out * ctrl_wait_n_frames