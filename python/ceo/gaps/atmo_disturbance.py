from ceo import constants, Atmosphere
from SimBlock import SimBlock
import numpy as np


class atmo_disturbance(SimBlock):
    """
    Class that simulates the atmospheric disturbance generation at full resolution.
    
    Parameters:
    -----------
    pup_obj : gmt_pupil object
        GMT pupil object encapsulating the GMT and Source CEO objects required to propagate a wavefront. See "gmt_pupil" class.
    
    r0 : float
        Fried parameter [in meters @ 500 nm]. Default: 16.0 cm
    
    L0 : float
        Outer scale [in meters]. Default: 25.0 m
    
    turb_type : str
        Type of turbulence simulation. Options are:
            1. "LCO typical-typical" (Default)
                Simulates the 7-layer turbulence profile.
            2. "Single Layer"
                Simulates a single turbulence layer shifting according to the Taylor hypothesis.
            3. "Independent Realizations"
                Statistically decorrelated phase screen realizations (Note: do not use for closed-loop simulations.)
    
    single_layer_wind_speed : float
        Wind speed in m/s when selecting "Single Layer" type. Default: 13.5 m/s
    
    single_layer_wind_direction : float
        Direction of wind flow in degrees when selecting "Single Layer" type. Default: 0 deg (i.e. from left to right).
    
    Notes:
    ------
    This class inherits timing properties from the abstract class "SimBlock".
    """
    def __init__(self, pup_obj, r0=16.0, L0=25.0, turb_type='LCO typical-typical', single_layer_wind_speed=13.5, single_layer_wind_direction=0.0):
        
        assert turb_type in ["LCO typical-typical", "Single Layer", "Independent Realizations"], "Turbulence type not recognized. See documentation."
        
        #----------- SimBlock timing parameters ----------------------
        super().__init__()
        
        nPx = pup_obj.nPx
        D = pup_obj.D
        self.pixscale = D / (nPx-1)
        
        #------------ Initialize turbulence model --------------------
        print("Turbulence model: %s"%turb_type)
        if turb_type == "LCO typical-typical":
            altitude       = np.array([25, 275, 425, 1250, 4000, 8000, 13000])  # [m]
            xi0            = np.array([0.1257, 0.0874, 0.0666, 0.3498, 0.2273, 0.0681, 0.0751]) #Cn2 weights
            wind_speed     = np.array([5.6540, 5.7964, 5.8942, 6.6370, 13.2925, 34.8250, 29.4187]) # [m/s]
            wind_direction = np.array([0.0136, 0.1441, 0.2177, 0.5672, 1.2584, 1.6266, 1.7462])  # rad
            meanV = np.sum(wind_speed**(5.0/3.0)*xi0)**(3./5.)
            
            atm = Atmosphere(r0, L0, len(altitude), altitude, xi0, wind_speed, wind_direction,
                    L=D, NXY_PUPIL=nPx, duration=1.0, N_DURATION=15)
            
        elif turb_type == "Single Layer":
            meanV = single_layer_wind_speed
            
            atm = Atmosphere(r0, L0, wind_speed=meanV, wind_direction=(90+single_layer_wind_direction)*np.pi/180, 
                    L=D, NXY_PUPIL=nPx, duration=1.0, N_DURATION=15)
        
        elif turb_type == "Independent Realizations":
            meanV = np.nan
            atm = Atmosphere(r0, L0)

            
        #----------- Summary of turbulence parameters -----------------
        seeing = 0.9759*500e-9/r0*constants.RAD2ARCSEC
        print('            r0 @ 500nm : %2.1f cm'%(r0*1e2))
        print('        seeing @ 500nm : %2.2f arcsec'%seeing)
        print('                    L0 : %2.1f m'%L0)
        if turb_type != "Independent Realizations":
            tau0 = 0.314*r0/meanV
            print('       Mean wind speed : %2.1f m/s'%meanV)
            print('                  tau0 : %2.2f ms'%(tau0*1e3))
        
        #----- Store object references and properties
        self._atm = atm
        self._pup = pup_obj
        self._turb_type = turb_type
        self._meanV = meanV
        self.__Data = None
    
    
    @property
    def r0(self):
        return self._atm.r0
    
    @r0.setter
    def r0(self, _r0_):
        self._atm.r0 = _r0_
    
    @property
    def L0(self):
        return self._atm.L0
    
    @property
    def seeing(self):
        return 0.9759 * 500e-9 / (self.r0*constants.ARCSEC2RAD)
    
    @property
    def turb_type(self):
        return self._turb_type
    
    @property
    def meanWindSpeed(self):
        return self._meanV
    
    @property
    def tau0(self):
        return 0.314*self.r0/self.meanWindSpeed
    
        
    def trigger(self):
        """
        Triggers propagation through turbulence layers, and updates the output wavefront.
        Note: Use "get_data()" to retrieve the output wavefront.
        """
        gs = self._pup.gs_obj
        gmt = self._pup.gmt_obj
        nPx = self._pup.nPx
        
        #----- Propagate through turbulence
        gs.reset()
        if self.turb_type == "Independent Realizations":
            self._atm.reset()
            self._atm.get_phase_screen(gs, self.pixscale, nPx, self.pixscale, nPx, 0.0)
        else:
            self._atm.ray_tracing(gs, self.pixscale, nPx, self.pixscale, nPx, SimBlock.CURRENT_TIME)        
        gmt.propagate(gs)

        #----- Extract the turbulence phase map in the exit pupil
        PhaseTur = np.squeeze((gs.wavefront.phase.host() - self._pup.phase_ref) * gs.wavefront.amplitude.host())
        PhaseTur[self._pup.GMTmask] -= np.mean(PhaseTur[self._pup.GMTmask])  # global piston removed
        self.__Data = PhaseTur.reshape((nPx,nPx))
    
    
    def get_data(self):
        """
        Gets the current turbulence phase realization.
        """
        return self.__Data
    
    
    def get_wfe(self):
        """
        Returns the WFE of the current turbulence realization.
        """
        return np.sqrt(np.sum(self.__Data**2) / self._pup.nmask)


