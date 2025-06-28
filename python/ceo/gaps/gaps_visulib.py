import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from SimBlock import SimBlock

def pyr_display_signals(pwfs, sx,sy, title=None, fig=None, ax1=None, ax2=None, clb_shrink=1.0, clim_step = 0.2):
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

    clb_shrink : float
        Colorbar shrinking factor [0,1]. Default: 1.0

    clim_step : float
        Minimum color range will be [-clim_step, clim_step]. Default: 0.2
    
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
    vlim = np.ceil(np.max(np.abs(sx2d)) / clim_step) * clim_step
    imm = ax1.imshow(sx2d, interpolation='none', vmin=-vlim, vmax=vlim, origin='lower')
    clb = fig.colorbar(imm, ax=ax1, format="%.2f", shrink=clb_shrink)
    clb.ax.tick_params(labelsize=12)    

    ax2.set_title(title[1])
    ax2.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    vlim = np.ceil(np.max(np.abs(sy2d)) / clim_step) * clim_step
    imm2 = ax2.imshow(sy2d, interpolation='none', vmin=-vlim, vmax=vlim, origin='lower')
    clb2 = fig.colorbar(imm2, ax=ax2, format="%.2f", shrink=clb_shrink)
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


def show_live_loop(gaps):
    """
    Live display for closed loop simulation.
    """
    clear_output(wait=True)
    fig = plt.figure()
    fig.set_size_inches((15,10))
    fig.dpi = 75
    
    #------------ Residual WF ---------------------------------
    ax1 = fig.add_subplot(2,3,1)
    init_wf = gaps.wf_ctrl.get_wavefront() * 1e6
    vlim1 = np.ceil(np.max(np.abs(init_wf)) / 0.25) * 0.25
    im1 = ax1.imshow(init_wf, cmap='viridis', vmin=-vlim1, vmax=vlim1,
        extent=[-25.5/2, 25.5/2, -25.5/2, 25.5/2], origin='lower')
    clb1 = fig.colorbar(im1, ax=ax1, shrink=0.6)
    clb1.set_label('microns WF')
    clb1.ax.tick_params(labelsize=12)
    ax1.set_xlabel('m')
    wf_title ="WF RMS [nm]: %.1f"%(gaps.wf_ctrl.get_wfe()*1e9)
    ax1.set_title(wf_title)

    #------------ Show DM command ------------------------------
    ax2 = fig.add_subplot(2,3,2)
    dmcomm = np.zeros(gaps.tel.mems2k.n_acts)
    dmcomm[gaps.dm_valid_acts_params['dm_valid_acts']] = gaps.ao_ctrl.get_dm_command() + gaps.wf_ctrl.dm_offset
    dmcomm2D = gaps.tel.mems2k.get_comm_2D(dmcomm) * 1e6
    vlim2 = np.ceil(np.nanmax(np.abs(dmcomm2D)) / 0.25) * 0.25
    im2 = ax2.imshow(dmcomm2D, cmap='viridis', vmin=-vlim2, vmax=vlim2, interpolation='none')#, origin='lower')
    clb2 = fig.colorbar(im2, ax=ax2, shrink=0.6)
    clb2.set_label('microns WF')
    clb2.ax.tick_params(labelsize=12)
    ax2.set_xlabel('act #')
    ax2.set_title('DM commands')
    
    #------------- PTT WF -----------------------
    ax3 = fig.add_subplot(2,3,3)
    pttcomm = gaps.ao_ctrl.get_ptt_command() + gaps.wf_ctrl.ptt_offset
    wf_ptt = gaps.tel.ptt.get_wf(pttcomm) * 1e6
    vlim3 = np.ceil(np.max(np.abs(wf_ptt)) / 0.25) * 0.25
    im3= ax3.imshow(wf_ptt, cmap='viridis', vmin=-vlim3, vmax=vlim3, interpolation='none', origin='lower')
    clb3 = fig.colorbar(im3, ax=ax3, shrink=0.6)
    clb3.set_label('microns WF')
    clb3.ax.tick_params(labelsize=12)
    ax3.set_title('PTT wavefront')
    ax3.axis('off')

    #---------- PWFS Sx and Sy signals ----------
    ax4 = fig.add_subplot(2,3,4)
    ax5 = fig.add_subplot(2,3,5)
    pyr_display_signals(gaps.pwfs, *gaps.pwfs.get_measurement(out_format='list'), \
                            fig=fig, ax1=ax4, ax2=ax5, clb_shrink=0.6)
    
    #--------- Show HDFS frame -----------------
    ax6 = fig.add_subplot(2,3,6)
    im6 = ax6.imshow(gaps.hdfs.ccd_frame, origin='lower', interpolation='none', \
            cmap=plt.cm.gist_earth_r, extent = gaps.hdfs._im_range_mas.tolist() * 2)
    ax6.set_xlabel('mas')

    gaps._tid.toc()
    fig_title = "iter: %d/%d, ET: %.3f s"%(SimBlock.current_iteration(), 
                        gaps.totSimulIter, gaps._tid.elapsedTime*1e-3)
    fig.suptitle(fig_title)

    plt.tight_layout()
    plt.show()
    plt.close(fig)    