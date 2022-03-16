import numpy as np
import matplotlib.pyplot as plt

#================== Visualization Routines ========================

def dm292_actmap():
    """
    ALPAO DM292 actuator geometry map
    """
    vv = np.linspace(-0.98,0.98,20)
    xxa, yya = np.meshgrid(vv,vv)
    actmap = xxa**2 + yya**2 < 1
    actmap[5,1] = False
    actmap[1,5] = False
    actmap[14,1] = False
    actmap[1,14] = False
    actmap[18,5] = False
    actmap[5,18] = False
    actmap[14,18] = False
    actmap[18,14] = False
    return actmap


def show_dm292(comvec, fig=None, ax=None):
    """
    Show a command vector to the DM292 as a 2D map.
    Note: actuator #1 is at the top-right corner to match actuator mapping in WFPT exit pupil.
    """
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches((6,5))

    com2d = np.full((20,20), np.nan)
    com2d[dm292_actmap()] = comvec
    im = ax.imshow(com2d, interpolation='None', aspect='equal')
    clb = fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(0,20,1)-0.5)
    ax.set_yticks(np.arange(0,20,1)-0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(alpha=0.5)
    fig.gca().invert_xaxis()


def show_wavefront(opd, fig=None, ax=None, clb_label=None, title=None):
    """
    Show OPD in exit pupil.
    """
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches((6,5))

    im = ax.imshow(opd, origin='lower', interpolation='None', aspect='equal')
    clb = fig.colorbar(im, ax=ax)
    ax.axis('off')
    
    if clb_label is not None:
        clb.set_label(clb_label)
    
    if title is not None:
        ax.set_title(title)


def show_rays(xyz, klm, sid, vig, fig=None, ax=None, rays_color='gray', label_surf_from=0):
    """
    Show rays diagram.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(16,14))
        
    for k in range(1,vig.sum(),10000):
        ray = np.vstack([w[vig,:][k] for w in xyz[:len(sid)+1]])
        ax.plot(ray[:,2], ray[:,1], rays_color, alpha=0.25)    

    chief_ray = np.vstack([w[vig,:][0] for w in xyz[:len(sid)+1]])
    htz = [ax.text(chief_ray[k+1,2],chief_ray[k+1,1],f"  {sid[k]}",
            fontdict={'color':'darkred'}) for k in range(label_surf_from,len(sid))]
            
    ax.grid('on')
    ax.set_aspect('equal');
    ax.set_xlabel('Z [m]');
    ax.set_ylabel('Y [m]');


def show_SH_slopes(sx2d, sy2d, fig=None, ax=None, title=None, clb_label=None):
    """
    Show SH slopes as 2D maps
    """
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches((12,5))
    
    im = ax.imshow(np.hstack([sx2d,sy2d]), interpolation='None', origin='lower')
    clb = fig.colorbar(im, ax=ax)
    ax.axis('off')
    
    if clb_label is not None:
        clb.set_label(clb_label)
    
    if title is not None:
        ax.set_title(title)
        

def show_DFS_data(data, fig=None, ax=None):
    """
    Show DFS data (fringes, fftlets) passed as a data cube.
    See also: DFS method "get_data_cube".
    """
    if fig is None:
        fig, ax = plt.subplots(ncols=6, nrows=2)
        fig.set_size_inches((10,3))
        fig.dpi=300
        for k in range(12):
            (ax.ravel())[k].imshow(data[:,:,k], cmap=plt.cm.gist_earth_r, origin='lower', interpolation='None')
            (ax.ravel())[k].autoscale(False)
            (ax.ravel())[k].set_title('%d'%(k+1), fontsize=12)
            (ax.ravel())[k].axis('off')
    fig.tight_layout()