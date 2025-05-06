import numpy as np
from ceo import GMT_MX, Source

class gmt_pupil:
    """
    Class to encapsulate CEO mask creation for an on-axis direction.
    
    Parameters:
    -----------
    array_size_pix : int
        Size in pixels of simulated array containing the GMT pupil. Default: 512
    
    array_size_m : float
        Size in meters of simulated array containing the GMT pupil (should be slightly larger than GMT diameter). Default: 25.5 m
    
    array_rot_angle : float
        Angle of rotation in degrees of the GMT pupil. Default: 0 deg
    
    project_truss_onaxis : bool
        If True, truss shadow will be applied to central pupil mask. Default: False
    """
    def __init__(self, array_size_pix = 512, array_size_m = 25.5, array_rot_angle = 0.0,
                project_truss_onaxis = False):
        
        #---> GMT object:
        gmt = GMT_MX()
        gmt.project_truss_onaxis = project_truss_onaxis
        
        #---> Source object:
        gs = Source('R+I', rays_box_size = array_size_m, 
                rays_box_sampling = array_size_pix, 
                rays_rot_angle = array_rot_angle * np.pi / 180)
        
        #-- Extract the GMT segment masks
        gmt.reset()
        gs.reset()
        gmt.propagate(gs)
        self._phase_ref = gs.wavefront.phase.host()
        self._nPx = array_size_pix
        self._D = array_size_m
        self._rot_angle = array_rot_angle
        self._P = np.squeeze(np.array(gs.rays.piston_mask))
        self._GMTmask = np.sum(self._P,axis=0).astype('bool')
        self._GMTmask2D = self._GMTmask.reshape((array_size_pix,array_size_pix))
        self._npseg = np.sum(self._P, axis=1)
        self._nmask = np.sum(self._P)
        
        #-- Store CEO objects
        self.gmt_obj = gmt
        self.gs_obj = gs
    
    
    @property
    def phase_ref(self):
        return self._phase_ref
    
    @property
    def nPx(self):
        return self._nPx
    
    @property
    def D(self):
        return self._D
    
    @property
    def rot_angle(self):
        return self._rot_angle
    
    @property
    def P(self):
        return self._P
    
    @property
    def GMTmask(self):
        return self._GMTmask
    
    @property
    def GMTmask2D(self):
        return self._GMTmask2D
    
    @property
    def npseg(self):
        return self._npseg
    
    @property
    def nmask(self):
        return self._nmask
    
    
    def cleanup(self):
        """
        Delete GMT and Source objects to free up memory space.
        """
        if hasattr(self, 'gmt_obj'):
            del self.gmt_obj
        if hasattr(self, 'gs_obj'):
            del self.gs_obj
        
    