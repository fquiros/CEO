import numpy as np
import matplotlib.pyplot as plt

def pyr_display_signals(pwfs, sx,sy, title=None, fig=None, ax1=None, ax2=None):
    """
    Display PWFS slopes (sx, sy) in 2D.
    
    Parameters:
    -----------
    pwfs : pwfs_model object
        The model of the NGWS-P PWFS.
    
    sx : numpy array or list
        Sx slope vector
    
    sy : numpy array or list
        Sy slope vector
    
    title : 2-element list with strings
        Title of Sx and Sy figures. Default: ['Sx', 'Sy'].
    
    fig : matplotlib figure
        Figure containing the display. If none provided, one will be created.
    
    ax1 : matplotlib axis
        Axis for Sx display.
    
    ax2 : matplotlib axis
        Axis for Sy display.
    
    Usage:
    ------
    Use fig, ax1 and ax2 when you want to integrate this display into another canvas.
    """
    sx2d = pwfs.get_sx2d(this_sx=sx)
    sy2d = pwfs.get_sy2d(this_sy=sy)

    if fig is None:
        fig, (ax1,ax2) = plt.subplots(ncols=2)
        fig.set_size_inches(12,4)
    
    if title is None:
        title = ['Sx', 'Sy']
        
    ax1.set_title(title[0])
    ax1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    imm = ax1.imshow(sx2d, interpolation='None',origin='lower')#, vmin=-1, vmax=1)
    clb = fig.colorbar(imm, ax=ax1, format="%.4f")
    clb.ax.tick_params(labelsize=12)    

    ax2.set_title(title[1])
    ax2.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    imm2 = ax2.imshow(sy2d, interpolation='None',origin='lower')#, vmin=-1, vmax=1)
    clb2 = fig.colorbar(imm2, ax=ax2, format="%.4f")  
    clb2.ax.tick_params(labelsize=12)    

    return (sx2d,sy2d)


def hdfs_show_fringes(ps, fringes=None, apodize=True, normalize=False, derotate=True, full=True):
    """
    Display HDFS fringes.
    
    Parameters:
    -----------
    ps : hdfs_model object
        The model of the NGWS-P HDFS.
    
    fringes : cube of numpy arrays.
        Cube of fringes to display. If none are provided, the fringes will be extracted from the 'ps' object.
    
    apodize : bool
        Apodize fringes when extracting from 'ps' object. Default: True
    
    normalize : bool
        Normalize fringes when extracting from 'ps' object. Default: False
    
    derotate : bool
        Derotate fringes when extracting from 'ps' object. Default: True
    
    full : bool
        If True, show all 14 fringes. Otherwise show only the first 7. Default: True
    """
    if fringes is None:
        fringes = ps.extract_fringes(apodize=apodize, normalize=normalize, derotate=derotate).get()

    if full:
        fig, ax = plt.subplots(ncols=ps._N_FRINGES//2, nrows=2)
        fig.set_size_inches((10,4))
        fig.dpi=300
        for k in range(ps._N_FRINGES):
            (ax.ravel())[k].imshow(fringes[:,:,k], cmap=plt.cm.gist_earth_r)#, origin='lower')
            (ax.ravel())[k].autoscale(False)
            (ax.ravel())[k].set_title('%d'%(k%ps._N_FRINGES+1), fontsize=12)
            (ax.ravel())[k].axis('off')
    else:
        fig, ax = plt.subplots(ncols=ps._N_FRINGES//2, nrows=1)
        fig.set_size_inches((10,1.5))
        for k in range(ps._N_FRINGES//2):
            (ax.ravel())[k].imshow(fringes[:,:,k], cmap=plt.cm.gist_earth_r)#, origin='lower')
            (ax.ravel())[k].autoscale(False)
            (ax.ravel())[k].set_title('%d'%(k%ps._N_FRINGES+1), fontsize=12)
            (ax.ravel())[k].axis('off')

    plt.tight_layout()