import numpy as np
from ceo import GMT_MX, Source

class ptt_array_model:
    """
    Generates the model of the PTT array.
    
    Parameters:
    ------------
    array_size_pix : int
        Size in pixels of simulated array containing the PTT array
    
    array_size_m : float
        Size in meters of simulated array containing the PTT array (should be slightly larger than GMT diameter). Default: 25.5 m
    
    array_rot_angle : float
        Angle of rotation in degrees of the PTT array. Default: 0 deg
        
    """
    def __init__(self, array_size_pix, array_size_m = 25.5, array_rot_angle = 0.0):
        #--------------------- Create the PTT array
        
        #-- Initialize CEO objects used to generate the PTT array
        gmt = GMT_MX()
        gmt.project_truss_onaxis = False
        gs = Source('R+I', rays_box_size = array_size_m, 
                        rays_box_sampling = array_size_pix, 
                        rays_rot_angle = array_rot_angle * np.pi / 180)

        #-- Extract the GMT segment masks
        gmt.reset()
        gs.reset()
        gmt.propagate(gs)
        P = np.squeeze(np.array(gs.rays.piston_mask))
        GMTmask = np.sum(P,axis=0).astype('bool')
        GMTmask2D = GMTmask.reshape((array_size_pix,array_size_pix))
        npseg = np.sum(P, axis=1)
        nmaskPup = np.sum(P)
        
        #-- Create coordinate vectors over array
        vv = np.linspace(-1,1,array_size_pix) * (array_size_m/2)
        [x_ep,y_ep] = np.meshgrid(vv,vv) # rows x cols
        rotmat = np.array([[np.cos(gs.rays.rot_angle), -np.sin(gs.rays.rot_angle)],
                           [np.sin(gs.rays.rot_angle),  np.cos(gs.rays.rot_angle)]])
        xytemp = rotmat @ np.array([x_ep.ravel(),y_ep.ravel()])
        x_epr = np.reshape(xytemp[0,:],(array_size_pix,array_size_pix))
        y_epr = np.reshape(xytemp[1,:],(array_size_pix,array_size_pix))
        
        #-- Generate global PTT modes from which to carve out segment PTT modes
        PTTmat = np.zeros((nmaskPup,3))
        PTTmat[:,0] = 1
        PTTmat[:,1] = x_epr[GMTmask2D]
        PTTmat[:,2] = y_epr[GMTmask2D]
        
        #-- Make sure global PTT modes are orthonormal
        PTT_Dmat = np.matmul(np.transpose(PTTmat), PTTmat) / nmaskPup
        PTT_Lmat = np.linalg.cholesky(PTT_Dmat)
        PTT_inv_Lmat = np.linalg.pinv(PTT_Lmat)
        PTTmato = np.matmul(PTTmat, np.transpose(PTT_inv_Lmat))

        #print("WF RMS of global PTT modes:")
        #print(np.array_str(np.sum(PTTmato**2,axis=0)/nmaskPup, precision=2))

        #-- Generate segment PTT modes
        segPTTmat = np.zeros((nmaskPup,7*3))
        for gidx in range(3):
            tempMat = np.zeros(array_size_pix**2)
            tempMat[GMTmask] = PTTmato[:,gidx]
            for segid in range(7):
                ptt_ifunc = np.zeros(array_size_pix**2)
                ptt_ifunc[P[segid,:]] = tempMat[P[segid,:]]
                segPTTmat[:,gidx*7+segid] = ptt_ifunc[GMTmask]
        
        #-- Orthonormalize segment PTT modes
        segPTT_Dmat = np.matmul(np.transpose(segPTTmat), segPTTmat)/np.tile(npseg,3)
        segPTT_Lmat = np.linalg.cholesky(segPTT_Dmat)
        segPTT_inv_Lmat = np.linalg.pinv(segPTT_Lmat)
        segPTTmato = np.matmul(segPTTmat, np.transpose(segPTT_inv_Lmat))
        
        #print("WF RMS of segment PTT modes (over corresponding segment pupil):")
        #print(np.array_str(np.sum(segPTTmato**2,axis=0)/np.tile(npseg,3), precision=2))
        
        #-- Make sure P2V of segment TT modes equals 4
        for ttidx in range(7*2):
            segPTTmato[:,7+ttidx] *= (2.0 / np.max(segPTTmato[:,7+ttidx]))
        
        #-- Create PTT array Influence Functions Cube
        self.IFcube = np.zeros((array_size_pix,array_size_pix,7*3))
        for gidx in range(3):
            for segid in range(7):
                wf1 = np.zeros((array_size_pix**2))
                wf1[GMTmask] = segPTTmato[:,gidx*7+segid]
                self.IFcube[:,:,gidx*7+segid] = wf1.reshape((array_size_pix,array_size_pix))
    
    
    def get_wf(self, ptt_command):
        """
        Generate PTT wavefront corresponding to the input PTT command vector.
        
        Parameters:
        -----------
        ptt_command : numpy array.
            21-element PTT command.
            Note: The command must be ordered in this way:
                1. segment piston (x7)
                2. segment x-tilt (x7)
                3. segment y-tilt (x7)
        """
        return np.sum(self.IFcube * ptt_command, axis=2)        

        

        