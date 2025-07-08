import numpy as np
import os
import sys

from gaps_simul import gaps_simul
from gaps_modes import gaps_modes
from ngao_controller import ao_controller, hdfs_controller, interaction_matrix
from wf_combiner import wf_combiner
from gaps_utilities import load_dictionary_from_file

class modal_control_test(gaps_simul):
    """
    Modal control test on GAPS.
    """
    def configure(self, m2c_file, pure_delay=1, modal_amp_max=100e-9, spp_amp=1e-6,
                 global_pist_reg_factor=1e11):
        """
        Set Modal Control with the DM+PTT array, driven by the PWFS.
                
        Parameters:
        -----------
        m2c_file : str
            Name of file containing the modal definition.
            NOTE: M2C files are created using 'script_compute_gaps_segment_modes.py'.

        pure_delay : int
            Pure (discrete) delay, in number of frames. Default: 1

        modal_amp_max : float
            maximum modal amplitude applied during interaction matrix calibration [m wf].
            NOTE: the modal amplitude is scaled with the modal radial order for higher-order modes.
        
        spp_amp : float
            segment piston amplitude applied during HDFS interaction matrix calibration [m wf].
            Default: 1e-6
            NOTE: Segment piston is produced with the PTT array.
        
        global_pist_reg_factor : float
            global piston penalizing factor for the HDFS reconstructor. Default: 1e11        
        """
        
        #--> Load modal basis definition:
        self.modes = gaps_modes(m2c_file)
        assert self.modes.M2C_DATA['mems_params'] == self.simul_params['mems_params'], \
                "DM model parameters of M2C on file do not match current simulation setting."
        
        #--> Retrieve DM valid actuators:
        validacts = self.modes.dm_valid_acts
        self.dm_valid_acts_params = self.modes.M2C_DATA['dm_valid_acts_params']
        
        #--> Calibrate PWFS modal interaction matrix:
        self.calib_repo['modal-dm-ptt-pwfs'] = {}
        self._get_modal_interaction_matrix(modal_amp_max)
        self._do_svd_and_compute_reconstructor()
        
        #--> HDFS interaction matrix and reconstructor computation:
        self.calib_repo['spp-hdfs'] = {}
        self._get_hdfs_matrices(spp_amp, global_pist_reg_factor)
        
        #--> Setup atmospheric turbulence projection:
        mergedIFmat, inv_mergedIFmat = self.tel.get_merged_influence_matrices(validacts)
        self.atm.set_wavefront_projection(mergedIFmat, inv_mergedIFmat)
        
        #--> Setup AO controller:
        self.ao_ctrl = ao_controller(self.calib_repo['modal-dm-ptt-pwfs']['recmat'],
                        pure_delay=pure_delay, modal_control=True, modes_obj=self.modes,
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
        

    def _get_modal_interaction_matrix(self, amp_max):
        """
        Load from file, or compute relevant interaction matrices.
        """
        #---> Load or compute zonal DM-PYR interaction matrix
        sys.stdout.write("Modal calibration between DM and PWFS in progress....\n")
        
        #-- Name of interaction matrix file:
        here  = os.path.abspath(os.path.dirname(__file__))
        fname = os.path.splitext(self.modes.M2C_DATA['filename'])[0]
        intmat_fname = "MODAL_PYR_DM_mod%0.1f_%s.npz"%(self.pwfs.modulation, fname)
        intmat_full_fname = os.path.join(here, 'data', 'intmats', intmat_fname)
        
        #-- Load interaction matrix from file:
        try:
            data = load_dictionary_from_file(intmat_full_fname)
            intmat = data['intmat']
            amp_wf = data['modal_amp']
        
        #-- If no file exists, compute the interaction matrix and save it:
        except FileNotFoundError:
            
            #-- Ensure intmat calibration is done without measurement noise:
            simul_noise_status = self.pwfs.simul_noise # ensure noiseless calibration.
            self.pwfs.wfs.simul_noise = False
            
            #-- Calibrate interaction matrix and save it:
            self.modes.compute_modal_shapes(self.tel)
            n_mode = self.modes.n_mode
        
            #--- Modal amplitude for IntMat calibration
            radord = np.floor((np.sqrt(8*np.arange(1,n_mode+1)-7)-1)/2)
            radord[0] = 1
            amp_wf = amp_max / np.sqrt(radord)
            amp_wf = np.tile(amp_wf,7)
        
            #TO DO: use act_list to select subset of modes to calibrate, and extract corresponding columns from M2C.
            #TO DO: Return slope RMS to optimize iteratively modal amplitudes.
            intmat = interaction_matrix(self.pwfs, \
                self.modes.KLFcube * self.tel.pup.GMTmask2D[:,:,np.newaxis], \
                np.arange(n_mode*7), amp_wf = amp_wf)
            tosave = dict(intmat=intmat, modal_amp = amp_wf)
            np.savez(intmat_full_fname, **tosave)
            
            #-- Restore original noise setup:
            self.pwfs.wfs.simul_noise = simul_noise_status
            
        #-- Update Calibration Repository:
        self.calib_repo['modal-dm-ptt-pwfs'].update({'intmat': intmat,
                                            'modal_amp': amp_wf})
    
    
    def _do_svd_and_compute_reconstructor(self):
        """
        Do SVD analysis of interaction matrix, and compute LS reconstructor.
        """
        intmat = self.calib_repo['modal-dm-ptt-pwfs']['intmat']
        
        #--> Singular Value Decomposition:
        sys.stdout.write("Starting SVD analysis\n")
        UU1, ss1, VVT1 =np.linalg.svd(intmat)

        #--> Compute shape of last eigenmode
        last_eigenvec = np.copy(VVT1[-1,:])
        eigenmodevec = self.modes.M2Cmat @ last_eigenvec
        pttvec, dmvec1 = np.split(eigenmodevec, [21])
        dmvec = np.zeros(self.tel.mems2k.n_acts)
        dmvec[self.modes.dm_valid_acts] = dmvec1
        wf_last_eigenmode = self.tel.get_wf(dm_command=dmvec, ptt_command=pttvec)
        
        #--> Select threshold to filter last eigenmode (associated with global piston)
        svd_thr = (ss1/np.max(ss1))[-2:].sum()/2
        sys.stdout.write("LS reconstructor with %d eigenmodes filtered.\n"%np.sum(ss1/np.max(ss1) < svd_thr))
        recmat = np.linalg.pinv(intmat, rcond=svd_thr)
        
        #--> Update Calibration Repository:
        self.calib_repo['modal-dm-ptt-pwfs'].update({'sv': ss1, 
                                    'wf_last_eigenmode': wf_last_eigenmode,
                                    'svd_thr': svd_thr,
                                    'recmat': recmat})
    
    
    def _get_hdfs_matrices(self, amp_wf=1e-6, global_pist_reg_factor=1e11):
        """
        Get HDFS interaction matrix and reconstructor.
        Note: Segment piston modes are assumed to be part of the segment KL modal basis
        and produced with solely the PTT array.
        """
        sys.stdout.write("HDFS calibration of segment piston in progress...\n")
        
        #--> Segment Piston to Modes projection matrix
        KL0_idx = np.arange(7) * self.modes.n_mode
        Pp2m = np.zeros((self.modes.n_mode*7, 7))
        for segId in np.arange(7):
            Pp2m[KL0_idx[segId], segId] = 1 
        self.calib_repo['spp-hdfs']['Pp2m'] = Pp2m
            
        #--> HDFS interaction matrix
        simul_noise_status = self.hdfs.simul_noise
        self.hdfs.wfs.simul_noise = False
        D_HDFS = interaction_matrix(self.hdfs, \
            self.tel.ptt.IFcube[:,:,0:7] * self.tel.pup.GMTmask2D[:,:,np.newaxis], \
            np.arange(7), amp_wf=amp_wf)
        self.calib_repo['spp-hdfs']['intmat'] = D_HDFS
        self.hdfs.wfs.simul_noise = simul_noise_status
        
        #--> SVD analysis
        sys.stdout.write("Computing HDFS Reconstructor...\n")
        UU1, ss1, VVT1 = np.linalg.svd(D_HDFS)
        
        #--> Compute shape of last eigenmode
        ptt_comm = np.zeros(21)
        ptt_comm[0:7] = VVT1[-1,:]
        wf_last_eigenmode = self.tel.ptt.get_wf(ptt_comm)
        
        #--> Regularization penalizing global piston:
        HDFS_reg_mat = D_HDFS.T @ D_HDFS + global_pist_reg_factor * np.ones((7,7))
        UU2, ss2, VVT2 = np.linalg.svd(HDFS_reg_mat, full_matrices=False, hermitian=True)
        
        #--> Regularized reconstruction matrix
        R_HDFS = np.linalg.solve(HDFS_reg_mat, D_HDFS.T)
        
        #--> Update Calibration Repository:
        self.calib_repo['spp-hdfs'].update({'sv': ss1, 
                                    'reg_sv': ss2,
                                    'wf_last_eigenmode': wf_last_eigenmode,
                                    'reg_factor': global_pist_reg_factor,
                                    'recmat': R_HDFS})

    
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