from EXOSIMS.Prototypes.Observatory import Observatory
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
import numpy as np
import os, inspect
import scipy.interpolate as interpolate
try:
    import cPickle as pickle
except:
    import pickle
from scipy.io import loadmat

class WFIRSTObservatoryL2(Observatory):
    """ WFIRST Observatory at L2 implementation. 
    Only difference between this and the Observatory implementation
    is the orbit method, and carrying an internal equinox time value.
    
    Orbit is stored in pickled dictionary on disk (generated by MATLAB
    code adapted from E. Kolemen (2008).  Describes approx. 6 month halo
    which is then patched for the entire mission duration).
    
    """

    def __init__(self, equinox=60575.25, orbit_datapath=None, **specs):
        
        # run prototype constructor __init__ 
        Observatory.__init__(self,**specs)
        
        # set equinox value
        self.equinox = Time(np.array(equinox, ndmin=1, dtype=float),
                format='mjd', scale='tai')
        
        # find and load halo orbit data in heliocentric ecliptic frame
        if orbit_datapath is None:
            classpath = os.path.split(inspect.getfile(self.__class__))[0]
            print classpath
            filename = 'L2_halo_orbit_six_month.p'
            orbit_datapath = os.path.join(classpath, filename)
        if not os.path.exists(orbit_datapath):
            matname = 'L2_halo_orbit_six_month.mat'
            mat_datapath = os.path.join(classpath, matname)
            if not os.path.exists(mat_datapath):
                raise Exception("Orbit data file not found.")
            else:
                halo = loadmat(mat_datapath)
                pickle.dump(halo, open(orbit_datapath, 'wb'))
        else:
            halo = pickle.load(open(orbit_datapath, 'rb'))
        
        # unpack orbit properties in heliocentric ecliptic frame 
        self.period_halo = halo['te'][0,0]/(2*np.pi)
        self.t_halo = halo['t'][:,0]/(2*np.pi)*u.year # 2\pi = 1 sideral year
        self.r_halo = halo['state'][:,0:3]*u.AU
        self.v_halo = halo['state'][:,3:6]*u.AU/u.year*(2*np.pi)
        # position wrt Earth
        self.r_halo[:,0] -= 1*u.AU
        
        # create interpolant for position (years & AU units)
        self.r_halo_interp = interpolate.interp1d(self.t_halo.value,
                self.r_halo.value.T, kind='linear')
        # create interpolant for orbital velocity (years & AU/yr units)
        self.v_halo_interp = interpolate.interp1d(self.t_halo.value,
                self.v_halo.value.T, kind='linear')
                
        # orbital properties used in Circular Restricted 3 Body Problem
        self.L2_dist = halo['x_lpoint'][0][0]*u.AU
        self.r_halo_L2 = halo['state'][:,0:3]*u.AU
        # position wrt L2
        self.r_halo_L2[:,0] -= self.L2_dist 
        
        # create new interpolant for CR3BP (years & AU units)
        self.r_halo_interp_L2 = interpolate.interp1d(self.t_halo.value,
                self.r_halo_L2.value.T, kind='linear')


    def orbit(self, currentTime, eclip=False):
        """Finds observatory orbit positions vector in heliocentric equatorial (default)
        or ecliptic frame for current time (MJD).
        
        This method returns the WFIRST L2 Halo orbit position vector.
        
        Args:
            currentTime (astropy Time array):
                Current absolute mission time in MJD
            eclip (boolean):
                Boolean used to switch to heliocentric ecliptic frame. Defaults to 
                False, corresponding to heliocentric equatorial frame.
        
        Returns:
            r_obs (astropy Quantity nx3 array):
                Observatory orbit positions vector in heliocentric equatorial (default)
                or ecliptic frame in units of AU
        
        Note: Use eclip=True to get ecliptic coordinates.
        
        """
        
        # find time from Earth equinox and interpolated position
        dt = (currentTime - self.equinox).to('yr').value
        t_halo = dt % self.period_halo
        r_halo = self.r_halo_interp(t_halo).T
        # find Earth positions in heliocentric ecliptic frame
        r_Earth = self.solarSystem_body_position(currentTime, 'Earth',
                eclip=True).to('AU').value
        # adding Earth-Sun distances (projected in ecliptic plane)
        r_Earth_norm = np.linalg.norm(r_Earth[:,0:2], axis=1)
        r_halo[:,0] = r_halo[:,0] + r_Earth_norm
        # Earth ecliptic longitudes
        lon = np.sign(r_Earth[:,1])*np.arccos(r_Earth[:,0]/r_Earth_norm)
        # observatory positions vector in heliocentric ecliptic frame
        r_obs = np.array([np.dot(self.rot(-lon[x], 3), 
                r_halo[x,:]) for x in range(currentTime.size)])*u.AU
        
        assert np.all(np.isfinite(r_obs)), \
                "Observatory positions vector r_obs has infinite value."
        
        if not eclip:
            # observatory positions vector in heliocentric equatorial frame
            r_obs = self.eclip2equat(r_obs, currentTime)
        
        return r_obs
