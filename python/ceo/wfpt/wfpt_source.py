from ceo import FanBundle, Source, Mask, cuFloatArray
from ceo.tools import ascupy
import numpy as np
import cupy as cp

class wfpt_source:
    """
    A class to simulate the illumination fiber of the WFPT.
    
    Parameters
    ----------
    photometric_band : string
        The source photometric band (e.g. R+I, R, J, ...)
    rays_box_sampling : integer
        The linear sampling of the ray bundle.
    rays_box_size : float
        The size of the ray bundle [m].
    mag : float, optional
        equivalent R-magnitude of source. Default: 0.0
    zenith : float, optional
    azimuth : float, optional
        field angle of source (zen,azi) in radians. Default: (0,0).
    fwhm : float, optional
        The fwhm of the source intensity distribution in detector pixel unit (before binning), defaults to None.
    rays_rot_angle : float, optional
        The rays coordinate system rotation [deg] to render the correct orientation of the GMT pupil w.r.t. sensor subapertures. Default: 0.0
    """
    def __init__(self, photometric_band, rays_box_sampling, rays_box_size, mag=0.0, zenith=0.0, azimuth=0.0, fwhm=None, rays_rot_angle = 0.0):
        
        #-- WFPT / GMT optical parameters
        wfpt_entrance_pupil_diam=8.5e-3
        
        #-- Parameters for rays that will propagate through WFPT Zemax model
        self._D = np.double(wfpt_entrance_pupil_diam)
        self._nPx = rays_box_sampling
        
        u = np.linspace(-1,1,self._nPx)*self._D/2
        x,y = np.meshgrid(u,u)
        xp = x.ravel()
        yp = y.ravel()

        #--- Rotate the coordinate system of the rays bundle
        zp = (xp + 1j * yp)*(np.exp(1j*(rays_rot_angle)*np.pi/180))
        xp = np.real(zp).copy()
        yp = np.imag(zp).copy()

        #->>>>>>TODO: compute origin (x,y,0) coordinates for a given (zenith,azimuth)
        self._rays_prms = {"x":xp,"y":yp,"origin":[0.0,0.0,0.0],"z_position":64.912333333329997e-3}

        self._gs = Source(photometric_band, magnitude=mag, zenith=zenith, azimuth=azimuth,
            rays_box_sampling=self._nPx, rays_box_size=rays_box_size, rays_origin=[0,0,25], fwhm=fwhm)

        self.__WFphase = ascupy(self._gs.wavefront.phase)
        #self.__WFamplitude = ascupy(self._gs.wavefront.amplitude)
        self.__WFphase_ref = cp.zeros(self.__WFphase.shape, dtype=np.float32)

        #-- Needed to calibrate a segment piston mask:
        vv = cp.linspace(-1,1,self._nPx)*(rays_box_size/2)
        self.__xx, self.__yy = cp.meshgrid(vv,vv) # rows x cols
        outersegrad = 8.710 # separation between central and outer segments
        seg_angle = cp.array([180., 120., 60., 0., -60., -120.]) - float(rays_rot_angle) # following official LCS ordering
        seg_angle *= cp.pi/180
        seg_xc, seg_yc = outersegrad*cp.cos(seg_angle), outersegrad*cp.sin(seg_angle)
        self.__xc = cp.append(seg_xc, 0)
        self.__yc = cp.append(seg_yc, 0)
        self.__pupmask = ascupy(self._gs.amplitude)

    
    @property
    def fwhm(self):
        return self._gs.fwhm
    @fwhm.setter
    def fwhm(self, value):
        self._gs.fwhm = value
    
    @property
    def wavelength(self):
        return self._gs.wavelength

    
    def reset_rays(self):
        """
        Resets the rays bundle.
        """
        self.rays = FanBundle(**self._rays_prms)
        
    
    def reset(self):
        """
        Resets the rays bundle and the source object.
        """
        self.reset_rays()
        self._gs.reset()
        
    
    def applyOPD(self):
        """
        Compute the OPD in the exit pupil of the WFPT testbed.
        """
        #opd = self.rays.optical_path_difference.host().ravel()
        #self._gs.wavefront.addPhase(cuFloatArray(host_data=opd))

        opd = ascupy(self.rays.optical_path_difference).astype(np.float32)
        self.__WFphase += opd
        
        vig = self.rays.vignetting.host().ravel()>0
        self.__tel = Mask(self._nPx * self._nPx)
        self.__tel.alter(cuFloatArray(host_data=vig))
        self._gs.masked(self.__tel)


        # The code below didn't work.....
        #vig = ascupy(self.rays.vignetting).ravel() > 0
        #self.tel = Mask(self._nPx * self._nPx)
        #self.tel.alter(cuFloatArray(vig))
        #self.__WFamplitude[:] = vig.astype(np.float32)
        
    @property
    def piston_mask(self):
        xx = self.__xx * self.__pupmask
        yy = self.__yy * self.__pupmask
        Dseg = 8.370 # segment diameter (a bit oversized)
        return [(((xx-self.__xc[idx])**2 + (yy-self.__yc[idx])**2 <= (Dseg/2)**2) * 
                 self.__pupmask.astype('bool')).ravel() for idx in range(7)]


    def set_reference_wavefront(self):
        self.__WFphase_ref[:] = self.__WFphase[:]


    @property
    def reference_wavefront(self):
        return self.__WFphase_ref.reshape((self._nPx,self._nPx)).get()


    def phaseRms(self, where='pupil'):
        wf = (self.__WFphase - self.__WFphase_ref).ravel()
        if where == 'pupil':
            return cp.std(wf[self.__pupmask.astype('bool').ravel()]).get()
        elif where == 'segments':
            segmask = self.piston_mask
            #wf = self._gs.wavefront.phase.host().ravel()
            return np.array([cp.std(wf[segmask[idx]]).get() for idx in range(7)])


    def piston(self, where='pupil'):
        wf = (self.__WFphase - self.__WFphase_ref).ravel()
        if where == 'pupil':
            return cp.mean(wf[self.__pupmask.astype('bool').ravel()]).get()
        elif where == 'segments':
            segmask = self.piston_mask
            #wf = self._gs.wavefront.phase.host().ravel()
            return np.array([cp.mean(wf[segmask[idx]]).get() for idx in range(7)])
