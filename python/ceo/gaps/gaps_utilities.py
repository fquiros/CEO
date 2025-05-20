import numpy as np
import os
import errno

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
        with np.load(fname,allow_pickle=True) as data:
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