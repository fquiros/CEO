import numpy as np
import os

def influence_matrix_filename(device, nPx, M2_baffle_diam, project_truss_onaxis):
    """
    Returns the full path filename to an IFmat file with appropiate parameters.
    Parameters:
    -----------
    device : string
        Either 'M1_DM', 'M2_DM', 'M1_PTT_SPP', 'M2_PTT_SPP', 'M1_PTT_sTT', or 'M2_PTT_sTT'.
    nPx : int
        Number of pixels across exit pupil wavefront array.
    M2_baffle_diam : float
        Size of simulated M2 baffle [m].
    project_truss_onaxis : bool
        Flag indicating truss shadow applied.
    """
    DM_IFmat_file = device+'_IFmat_nPx%d_M2baffle%0.1fm_%s.npz'%(nPx, M2_baffle_diam,
                        'wTruss' if project_truss_onaxis else 'woTruss')
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(here, 'WFPT_model_data', 'influence_matrices', DM_IFmat_file)