import numpy as np
import os
from modal_basis import segment_modal_basis, fitted_segment_modal_basis
from telescope_simulator import telescope_simulator

def compute_gaps_segment_modes():
    
    here = os.path.abspath(os.path.dirname(__file__))
    
    #---> Pupil sampling parameters
    #=============================================
    D = 25.5    # [m] Diameter of simulated square (slightly larger than GMT diameter) 
    nPx = 460
    pup_angle = 0.0
    array_params = dict(array_size_m=D, array_size_pix=nPx, array_rot_angle=pup_angle)


    #---> Theoretical segment KL modes
    #=============================================
    kls = segment_modal_basis(**array_params)
    KLamp = 100e-9
    kls.do_segment_kls(KLamp=KLamp)


    #---> GAPS telescope simulator (DM+PTT) model
    #=============================================
    ref_ifunc_fname = 'MagAO-X_MEMS2k_ReferenceIF.npz'
    mems_ifunc_fname = os.path.join(here, 'data', 'mems2k', ref_ifunc_fname)
    pupil_size_in_mems_pitches = 48
    grid_rot_deg = -2.5

    gaps_params = dict(mems_ifunc_fname=mems_ifunc_fname, 
                       pupil_size_in_mems_pitches=pupil_size_in_mems_pitches,
                       mems_grid_rot_angle=grid_rot_deg)

    project_truss_onaxis = False

    gaps = telescope_simulator(**array_params, **gaps_params, 
                        project_truss_onaxis = project_truss_onaxis)

    #-- DM valid actuators
    #=============================================
    vact_thr = 0.4 # threshold to select illuminated actuators.
    validacts, ifpeak = gaps.get_dm_valid_actuators(threshold=vact_thr)


    #---> Fitted segment modes for GAPS
    #=============================================
    fito_nmode = 153 # final number of modes per segment (radial order 16)
    lo_mode = 66
    regularization_factor = 5e-4
    orthonormalize = True
    descaled = True

    fkls_params = dict(nmode=fito_nmode, lo_mode=lo_mode,
                       regularization_factor=regularization_factor,
                       orthonormalize=orthonormalize, 
                       descaled=descaled)

    fkls = fitted_segment_modal_basis(kls, gaps, validacts)
    fkls.gaps_segment_modes(**fkls_params)

    #---> Save M2C
    #=============================================
    fname = 'KLF_M2C_20250520_v0.npz'
    fullname = os.path.join(here, 'data', 'M2C', fname)

    tosave = dict(array_params=array_params, KLamp=KLamp, gaps_params=gaps_params,
                 dm_valid_acts=validacts, dm_valid_acts_thr=vact_thr, ifpeak=ifpeak, 
                 fkls_params=fkls_params, KLF_M2C=fkls.KLF_M2C)

    np.savez(fullname, **tosave)
    print("Saving to file %s"%fullname)

if __name__ == "__main__":
    compute_gaps_segment_modes()