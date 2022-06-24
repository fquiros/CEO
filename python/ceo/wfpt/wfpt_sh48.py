from ceo import ShackHartmann
import numpy as np

class wfpt_sh48(ShackHartmann):
    """
    ShackHartmann wrapper class for the Probe Zero SH48
    """

    def __init__(self, **kwargs):
        
        #--- Detector noise parameters:
        nBackgroundPhoton = 0.0
        readOutNoiseRms = 1.0
        noiseFactor = np.sqrt(2)
        photoElectronGain = 0.48 * 0.8
        
        ShackHartmann.__init__(self, noiseFactor=noiseFactor, photoElectronGain=photoElectronGain, readOutNoiseRms=readOutNoiseRms, nBackgroundPhoton=nBackgroundPhoton, **kwargs)
        
    
    def calibrate(self, src, threshold=0.0):
        """
        Calibrate SH sensor valid sub-aperture mask and reference slope vector.
        """
        if src.fwhm == 0.0:
            import warnings
            warnings.warn("Calibrating the SH with a point source is not recommended. Set src.fwhm to simulate extended source.")
        else:
            print("SH source FWHM: %.3f arcsec"%(src.fwhm * self.camera.pixelScaleArcsec(src._gs) /
                                                 self.BIN_IMAGE))

        super().calibrate(src._gs, threshold)
        self.__valid_lenslet_mask = self.valid_lenslet.f.host().reshape((self.N_SIDE_LENSLET,
                                                            self.N_SIDE_LENSLET)).astype('bool')

        print("Total SH valid sub-apertures: %d"%self.n_valid_lenslet)
    
    
    def propagate(self, src):
        super().propagate(src._gs)
    
    def analyze(self, src):
        super().analyze(src._gs)
    
    def readOut(self, exposureTime):
        super().camera.readOut(exposureTime, self.camera.readOutNoiseRms, 
                               nBackgroundPhoton = self.camera.nBackgroundPhoton, 
                               noiseFactor = self.camera.noiseFactor)
    
    @property
    def valid_lenslet_mask(self):
        return self.__valid_lenslet_mask
    
    @property
    def N_PX_SUBAP(self):
        return self.N_PX_IMAGE // self.BIN_IMAGE
    
    def slopes2d(self, slopes_vector=None):
        """
        Returns the sx and sy slopes as maps.
        """
        if slopes_vector is None:
            slopes_vector = self.get_measurement()
        
        sx2d = np.full((self.N_SIDE_LENSLET, self.N_SIDE_LENSLET), np.nan)
        sx2d[self.valid_lenslet_mask] = slopes_vector[0:self.n_valid_lenslet]
        sy2d = np.full((self.N_SIDE_LENSLET, self.N_SIDE_LENSLET), np.nan)
        sy2d[self.valid_lenslet_mask] = slopes_vector[self.n_valid_lenslet:]
        return sx2d, sy2d
    
    def get_spots_cube(self, frame=None):
        """
        Returns a data cube with the SH spots of ALL sub-apertures (valid or not).
        """
        if frame is None:
            frame = self.camera.frame.host()
        
        spots = []
        for ii in range(self.N_SIDE_LENSLET):
            for jj in range(self.N_SIDE_LENSLET):
                spots += [frame[ ii*self.N_PX_SUBAP : (ii+1)*self.N_PX_SUBAP, 
                                 jj*self.N_PX_SUBAP : (jj+1)*self.N_PX_SUBAP ]]
        return spots    
