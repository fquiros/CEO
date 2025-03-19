import numpy as np
from scipy.interpolate import RegularGridInterpolator

class mems_model:
    """
    Computes a set of MEMS influence functions using a single influence function provided as reference.
    
    Parameters:
    -----------
    reference_ifunc_file : string
        npz file containing:
            1) the reference influence function : (nPx,nPx) array.
            2) the actuator pitch value in pixels
            3) the location of the IF peak in pixels
    
    act_mask_file : string
        npz file containing the actuator mask (e.g. the 2040 valid actuators in a 50x50 array for the MEMS2K).
    
    act_pitch_m : float
        desired actuator pitch in meters
    
    grid_rot_deg : float
        DM grid clocking [deg]. Default: 0 deg
    """
    def __init__(self, reference_ifunc_file, act_mask_file, act_pitch_m, grid_rot_deg=0.0):
        
        #--- Load reference influence function model
        ref_data = dict(np.load(reference_ifunc_file))
        self.__ref_ifunc = ref_data['ifunc']
        self.__ref_act_pitch_pix = ref_data['act_pitch_pix'][()]
        self.__ref_ifunc_row_center = ref_data['if_row_center'][()]
        self.__ref_ifunc_col_center = ref_data['if_col_center'][()]
        self.__ref_nPx = self.__ref_ifunc.shape[0]
        
        #--- Load actuator mask
        act_mask_data = dict(np.load(act_mask_file))
        self._act_mask = act_mask_data['actmask']
        self._grid_n_acts = self._act_mask.shape[0]
        self.n_acts = self._act_mask.sum()
        
        #--- Desired actuator pitch in meters
        self.act_pitch_m = act_pitch_m
        ref_pixscale = self.act_pitch_m / self.__ref_act_pitch_pix
        
        #--- Sets interpolator function
        ref_row = np.arange(self.__ref_nPx) - self.__ref_ifunc_row_center
        ref_col = np.arange(self.__ref_nPx) - self.__ref_ifunc_col_center
        self.__ref_row_meters = ref_row * ref_pixscale
        self.__ref_col_meters = ref_col * ref_pixscale
        self.set_interpolator()
        
        #--- Sets MEMS grid (centered on pupil)
        self._grid_rot_deg = grid_rot_deg
        self._set_mems_grid()
    
    
    def set_interpolator(self, bounds_error=False, method='linear', fill_value=0):
        """
        Sets the interpolator function (currently using RegularGridInterpolator).
        
        Note:
            See RegularGridInterpolator documentation for description of parameters.
        """
        self.interp = RegularGridInterpolator((self.__ref_row_meters, 
                                               self.__ref_col_meters),
                                               self.__ref_ifunc,
                                                bounds_error = bounds_error,
                                                method = method,
                                                fill_value = fill_value)
    
    def _set_mems_grid(self):
        """
        Define MEMS grid.
        """
        grid_n_acts = self._grid_n_acts
        rot_angle = self._grid_rot_deg * np.pi / 180.
        act_grid_row = (np.arange(grid_n_acts) - (grid_n_acts-1)/2) * self.act_pitch_m
        act_grid_col = (np.arange(grid_n_acts) - (grid_n_acts-1)/2) * self.act_pitch_m
        act_row_meters, act_col_meters = np.meshgrid(act_grid_row, act_grid_col, indexing='ij')
        
        #-- Flip row coordinate to match BMC actuator numbering
        _act_row_meters_ = -act_row_meters[self._act_mask]
        _act_col_meters_ =  act_col_meters[self._act_mask]
        
        #-- Rotate grid
        rotmat = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
                           [np.sin(rot_angle),  np.cos(rot_angle)]])
        ijtemp = rotmat @ np.array([_act_col_meters_.ravel(), _act_row_meters_.ravel()])
        self._act_row_meters = np.squeeze(ijtemp[1,:])
        self._act_col_meters = np.squeeze(ijtemp[0,:])
    
    
    def gridact_to_act(self, gridact_idx):
        """
        Convert an actuator index vector defined on a (grid_n_acts**2 x 1) sized vector to
        an actuator index vector defined on a (n_acts x 1) sized vector.
        
        Parameters:
        -----------
        gridact_idx : numpy 1d-array
            actuator index vector defined on a (grid_n_acts**2 x 1) sized vector.
        
        Returns:
        -----------
        act_idx : numpy 1d-array
            actuator index vector defined on a (n_acts x 1) sized vector.
        """
        act_idx = []
        for jj in gridact_idx:
            try:
                idx = int(np.flatnonzero(np.flatnonzero(self._act_mask) == jj))
                act_idx += [idx]
            except:
                pass
        return np.array(act_idx)
    
    
    def compute_if_cube(self, array_size_pix, array_size_m):
        """
        Generate the data cube of MEMS influence functions. 
        The size of the cube will be: array_size_pix x array_size_pix x grid_n_acts**2
        
        Parameters:
        -----------
        array_size_pix : int
            size of array in pixels.
        
        array_size_m : m
            size of array in meters.
        """
        array_pixscale = array_size_m / (array_size_pix-1)
        gridrow = (np.arange(array_size_pix) - (array_size_pix-1)/2) * array_pixscale
        gridcol = (np.arange(array_size_pix) - (array_size_pix-1)/2) * array_pixscale
        self.IFcube = np.zeros((array_size_pix, array_size_pix, self.n_acts))
        
        for this_act in range(self.n_acts):
            _gridrow_ = gridrow - self._act_row_meters.ravel()[this_act]
            _gridcol_ = gridcol - self._act_col_meters.ravel()[this_act]
            gridrow_2d, gridcol_2d = np.meshgrid(_gridrow_, _gridcol_, indexing='ij')
            self.IFcube[:,:,this_act] = self.interp((gridrow_2d, gridcol_2d))
        
        print("Completed creation of influence function cube of size %d x %d x %d."%self.IFcube.shape)
        
        
        
        

    
        
        

            
        
    