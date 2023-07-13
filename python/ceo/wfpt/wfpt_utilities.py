import numpy as np
import os
import errno

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


def load_dictionary_from_file(fname):
    """
    Load dictionary from file.
    
    Parameters:
    -----------
    filename : string
        Full path name to file.
    """
    if os.path.isfile(fname):
        print("Restoring data from file: %s"%os.path.basename(fname))
        mydict = {}
        with np.load(fname) as data:
            for key in data.keys():
                try:
                    mydict[key] = data[key].item()
                except:
                    mydict[key] = data[key]
        mydict['filename'] = os.path.basename(fname)
    else:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), fname)
    return mydict


def get_dm_valid_actuators(dm_valid_acts_file):
    """
    Loads DM valid actuators data file.

    Parameters:
    -----------
    dm_valid_acts_file : string
        Name of file that contains the DM valid actuators.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    fname = os.path.join(here, 'WFPT_model_data', 'dm_valid_actuators', dm_valid_acts_file)
    return load_dictionary_from_file(fname)




        
