#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:25:40 2020

Simulate fabrication errors on optics using PSD

@author: bmcleod
"""

import numpy as np

class WFPT:

    """
    Initialize a wfpt aberration object
    
    Arguments:
        npix -- number of pixels in the output wavefront (square)
        maxfied -- maximum field angle (mm or equivalently arcmin at GMT)
        seed   -- integer random number seed. Keep this the same for multiple calls at different angles, but the same random wavefront (default = 0)

    """

    def __init__(self, npix=1000, maxfield=10, seed=0):
         self.npix = npix
         self.maxfield = maxfield
         self.seed = seed
         self.opticslist = []
         np.random.seed(seed)
         
    class Optic:
        
        def __init__(self, rms, normrad, p, shift):
            self.rms = rms
            self.normrad  = normrad
            self.p = p
            self.shift = shift
    
    def addoptic(self, nsurf, rms, normrad, p, shift):
        """ Add an optical element 
        
        Arguments:
        nsurf -- Number of surfaces with this property
        rms -- rms figure error in nm
        normrad -- Normalization radius (currently ignored)
        p -- slope of PSD (specified as positive number)
        shift -- fractional shift of pupil per unit of field position
        """
        
        optic = self.Optic( rms, normrad, p, shift)
        oversizefactor = 1.5
        npix = int((1 + abs(self.maxfield * shift)) * oversizefactor * self.npix)
  
        optic.wf = makewavefront(rms * np.sqrt(nsurf), rmax=1,rhoHP=1./1000.,npix=npix,p=p)

        self.opticslist.append(optic)
        
    
    def wavefront(self, fieldx=0,fieldy=0):
        """Generate a wavefront for a particular field angle
        
        Keyword arguments:
        fieldx -- x field angle in mm (equivalent to arcmin on GMT) (default = 0)
        fieldy -- y field angle (default = 0)
        
        Returns:
            wavefront: npix x npix numpy array with units of nm
        """
        
        wavefront = np.zeros((self.npix, self.npix))
        
        for optic in self.opticslist:
                       
            shiftpix_x = int(optic.shift * fieldx * self.npix)
            shiftpix_y = int(optic.shift * fieldy * self.npix)
            
            ny, nx = optic.wf.shape
            
            ny0 = ny//2 - self.npix//2 + shiftpix_y
            nx0 = nx//2 - self.npix//2 + shiftpix_x
    
            wavefront += optic.wf[ny0 : ny0 + self.npix, nx0 : nx0 + self.npix]
            
        return wavefront
       
def makewavefront(rms, rmax=25.4/2,rhoHP=25.4/2/1000.,npix=1000,p=2):

    rv,ru = (np.indices((npix,npix)) - npix/2)  * rmax / (npix/2)
    rho = np.sqrt(rv*rv + ru*ru)
    
    
    PSD = 1 / (1 + (rho / rhoHP)**p)
    
    aper = np.where(rho<rmax)
    
    
    # Compute a wavefront realization from the PSD
    
    
    phase = np.random.random((npix,npix)) * 2 * np.pi
    
    wavefront = np.fft.fft2(np.fft.fftshift(np.sqrt(PSD) * np.exp (1j * phase))).real
 #   A = np.array([np.ones((npix,npix)),rv,ru,(rho*rho)])
    Aaper = np.array([np.ones((npix,npix))[aper],rv[aper],ru[aper],(rho*rho)[aper]])


    # Subtract tip/tilt/piston/focus over aperture before normalizing

    fit = np.linalg.lstsq(Aaper.T,wavefront[aper],rcond=None)[0]
    resid = np.zeros((npix,npix))
    
    resid[aper] =  wavefront[aper] - fit.dot(Aaper)

    rmsresid = resid[aper].std()
    
    

    return wavefront / rmsresid * rms

def OAP( wfsize=1000, PVperoptic=633/10., PSDslope=1.6, seed=0):

    rmsperoptic = PVperoptic / 5
    maxfield = 10      
    dn = 2   # Change in refractive index (mirror)

    oaps = WFPT(wfsize,maxfield,seed)
    # Phase screen relays
    # PTT relays
    # DM relays
    for beam_diam in [44, 52, 38]:
        oaps.addoptic(2, rmsperoptic * dn, 1, PSDslope,  1. / beam_diam)  
        oaps.addoptic(2, rmsperoptic * dn, 1, PSDslope, -1. / beam_diam) 
    return oaps 
  
def Refractive( wfsize=1000, PVperoptic=633/10., PSDslope=2, seed=0):

    rmsperoptic = PVperoptic / 5
    maxfield = 10      
    dn = 0.6   # Change in refractive index (lens)
   
    refr = WFPT(wfsize,maxfield,seed)
    # Phase screen relays
    # PTT relays
    # DM relays
    for beam_diam in [44, 52, 38]:
        refr.addoptic(2 * 6, rmsperoptic * dn, 1, PSDslope,  1. / beam_diam)  
        refr.addoptic(2 * 6, rmsperoptic * dn, 1, PSDslope, -1. / beam_diam) 
    return refr
      
#%%
if __name__ == "__main__":
    
    import pylab as pl
    
    # For testing
    # Make a grid of DFS apertures in the wavefront and compute Piston error and Strehl
    def dfs(wf, apsizepix):
        npix = len(wf)
        iy,ix = np.indices((apsizepix,apsizepix))
        left = ix < apsizepix / 3
        right = ix  >= apsizepix * 2 / 3
        both = left + right
        Aaper = np.array([np.ones((apsizepix,apsizepix))[both],iy[both],ix[both]])   
        strehls = []
        pistons = []
        
        for y0 in np.arange(0,npix,apsizepix).astype(int)[:-1]:
            for x0 in np.arange(0,npix,apsizepix).astype(int)[:-1]:
                phasing_aperture = wf[y0:y0+apsizepix,x0:x0+apsizepix]
    
         
                # Subtract tip/tilt/piston over both sides of phasing  aperture 
                
                fit = np.linalg.lstsq(Aaper.T,phasing_aperture[both],rcond=None)[0]
                resid = np.zeros((apsizepix,apsizepix))
                resid[both] =  phasing_aperture[both] - fit.dot(Aaper)
        
                piston_error = resid[left].mean()-resid[right].mean()
                strehl = np.exp(-(resid[both].std()/1200 * 2 * np.pi)**2)
                
                strehls.append(strehl)    
                pistons.append(piston_error)
    
        return len(strehls), np.array(pistons).std(), np.array(strehls).mean()
    
    # Initialize the wavefronts
    lambda_by = 10
    PSD = 1.6
    oaps = OAP(wfsize = 1000, PVperoptic = 633. / lambda_by, PSDslope=PSD, seed=0)
    
    # Return wavefronts for particular field positions
    x = oaps.wavefront(0,0)  # On axis
    y = oaps.wavefront(0,-1) # 1 mm Off-axis
    z = oaps.wavefront(0,1)  # Another off-axis
    
    pl.imshow(np.hstack([x,y-x,z-x]))
    
    pl.colorbar()
    print("OAPS: with lambda/%d and PSD=%.1f" % (int(lambda_by),PSD))
    print("Before DM correction:")
    print("Full aperture RMS wavefront: %.0f" % (x.std()))
    print("Full aperture J-Strehl:", np.exp(-(x.std()/1200 * 2 * np.pi)**2))
    
    print("RMS piston error and DFS Strehl: ", dfs(x,60))
    
    # Initialize the wavefronts
    lambda_by = 10
    PSD = 2
    refr = Refractive(wfsize = 1000, PVperoptic = 633./lambda_by, PSDslope=PSD, seed=0)
    
    # Return wavefronts for particular field positions
    x = refr.wavefront(0,0)  # On axis
    y = refr.wavefront(0,-1) # 1 mm Off-axis
    z = refr.wavefront(0,1)  # Another off-axis
    
    pl.imshow(np.hstack([x,y-x,z-x]))
    
    pl.colorbar()
    print("")
    print("Refractive: with lambda/%d and PSD=%.1f" % (int(lambda_by),PSD))
    print("Before DM correction:")
    print("Full aperture RMS wavefront: %.0f" % (x.std()))
    print("Full aperture J-Strehl:", np.exp(-(x.std()/1200 * 2 * np.pi)**2))
    
    print("RMS piston error and DFS Strehl: ", dfs(x,60))


