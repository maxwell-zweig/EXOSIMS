# -*- coding: utf-8 -*-
import time
import numpy as np
from scipy import interpolate
import astropy.units as u
import astropy.constants as const
import os, inspect
try:
    import cPickle as pickle
except:
    import pickle
import hashlib
from EXOSIMS.Prototypes.Completeness import Completeness
from EXOSIMS.util.eccanom import eccanom
from EXOSIMS.util.deltaMag import deltaMag

class BrownCompleteness(Completeness):
    """Completeness class template
    
    This class contains all variables and methods necessary to perform 
    Completeness Module calculations in exoplanet mission simulation.
    
    Args:
        \*\*specs: 
            user specified values
    
    Attributes:
        Nplanets (integer):
            Number of planets for initial completeness Monte Carlo simulation
        classpath (string):
            Path on disk to Brown Completeness
        filename (string):
            Name of file where completeness interpolant is stored
        updates (float nx5 ndarray):
            Completeness values of successive observations of each star in the
            target list (initialized in gen_update)
        
    """
    
    def __init__(self, Nplanets=1e8, **specs):
        
        # bring in inherited Completeness prototype __init__ values
        Completeness.__init__(self, **specs)
        
        # Number of planets to sample
        self.Nplanets = int(Nplanets)
        
        # get path to completeness interpolant stored in a pickled .comp file
        self.classpath = os.path.split(inspect.getfile(self.__class__))[0]
        self.filename = specs['modules']['PlanetPopulation'] + specs['modules']['PlanetPhysicalModel']
        # get path to dynamic completeness array in a pickled .dcomp file
        self.dfilename = specs['modules']['PlanetPopulation'] + \
                        specs['modules']['PlanetPhysicalModel'] + \
                        specs['modules']['OpticalSystem'] + \
                        specs['modules']['StarCatalog'] + \
                        specs['modules']['TargetList']
        atts = self.PlanetPopulation.__dict__.keys()
        self.extstr = ''
        for att in sorted(atts, key=str.lower):
            if not callable(getattr(self.PlanetPopulation, att)) and att != 'PlanetPhysicalModel':
                self.extstr += '%s: ' % att + str(getattr(self.PlanetPopulation, att)) + ' '
        ext = hashlib.md5(self.extstr).hexdigest()
        self.filename += ext

    def target_completeness(self, TL):
        """Generates completeness values for target stars
        
        This method is called from TargetList __init__ method.
        
        Args:
            TL (TargetList module):
                TargetList class object
            
        Returns:
            comp0 (float ndarray): 
                Completeness values for each target star
        
        """
        
        # set up "ensemble visit photometric and obscurational completeness"
        # interpolant for initial completeness values
        # bins for interpolant
        bins = 1000
        # xedges is array of separation values for interpolant
        if self.PlanetPopulation.constrainOrbits:
            xedges = np.linspace(0.0, self.PlanetPopulation.arange[1].to('AU').value, bins+1)
        else:
            xedges = np.linspace(0.0, self.PlanetPopulation.rrange[1].to('AU').value, bins+1)
        
        # yedges is array of delta magnitude values for interpolant
        ymin = -2.5*np.log10(float(self.PlanetPopulation.prange[1]*\
                (self.PlanetPopulation.Rprange[1]/self.PlanetPopulation.rrange[0])**2))
        ymax = -2.5*np.log10(float(self.PlanetPopulation.prange[0]*\
                (self.PlanetPopulation.Rprange[0]/self.PlanetPopulation.rrange[1])**2)*1e-11)
        yedges = np.linspace(ymin, ymax, bins+1)
        # number of planets for each Monte Carlo simulation
        nplan = int(np.min([1e6,self.Nplanets]))
        # number of simulations to perform (must be integer)
        steps = int(self.Nplanets/nplan)
        
        # path to 2D completeness pdf array for interpolation
        Cpath = os.path.join(self.classpath, self.filename+'.comp')
        Cpdf, xedges2, yedges2 = self.genC(Cpath, nplan, xedges, yedges, steps)

        xcent = 0.5*(xedges2[1:]+xedges2[:-1])
        ycent = 0.5*(yedges2[1:]+yedges2[:-1])
        xnew = np.hstack((0.0,xcent,self.PlanetPopulation.rrange[1].to('AU').value))
        ynew = np.hstack((ymin,ycent,ymax))
        Cpdf = np.pad(Cpdf,1,mode='constant')

        #save interpolant to object
        self.Cpdf = Cpdf
        self.EVPOCpdf = interpolate.RectBivariateSpline(xnew, ynew, Cpdf.T)
        self.EVPOC = np.vectorize(self.EVPOCpdf.integral)
        self.xnew = xnew
        self.ynew = ynew  
            
        # calculate separations based on IWA and OWA
        OS = TL.OpticalSystem
        mode = filter(lambda mode: mode['detectionMode'] == True, OS.observingModes)[0]
        IWA = mode['IWA']
        OWA = mode['OWA']
        smin = np.tan(IWA)*TL.dist
        if np.isinf(OWA):
            smax = xedges[-1]*u.AU
        else:
            smax = np.tan(OWA)*TL.dist
        
        # limiting planet delta magnitude for completeness
        dMagMax = self.dMagLim
        
        # calculate dMags based on maximum dMag
        if self.PlanetPopulation.scaleOrbits:
            L = np.where(TL.L>0, TL.L, 1e-10) #take care of zero/negative values
            smin = smin/np.sqrt(L)
            smax = smax/np.sqrt(L)
            dMagMax -= 2.5*np.log10(L)
            comp0 = np.zeros(smin.shape)
            comp0[dMagMax>ymin] = self.EVPOC(smin[dMagMax>ymin].to('AU').value, \
                    smax[dMagMax>ymin].to('AU').value, 0.0, dMagMax[dMagMax>ymin])
        else:
            comp0 = self.EVPOC(smin.to('AU').value, smax.to('AU').value, 0.0, dMagMax)
        comp0[comp0<1e-6] = 0.0
        
        return comp0

    def gen_update(self, TL):
        """Generates dynamic completeness values for multiple visits of each 
        star in the target list
        
        Args:
            TL (TargetList module):
                TargetList class object
        
        """
        
        OS = TL.OpticalSystem
        PPop = TL.PlanetPopulation
        
        # limiting planet delta magnitude for completeness
        dMagMax = self.dMagLim
        
        # get name for stored dynamic completeness updates array
        # inner and outer working angles for detection mode
        mode = filter(lambda mode: mode['detectionMode'] == True, OS.observingModes)[0]
        IWA = mode['IWA']
        OWA = mode['OWA']
        extstr = self.extstr + 'IWA: ' + str(IWA) + ' OWA: ' + str(OWA) + \
                ' dMagMax: ' + str(dMagMax) + ' nStars: ' + str(TL.nStars)
        ext = hashlib.md5(extstr).hexdigest()
        self.dfilename += ext 
        self.dfilename += '.dcomp'
        
        path = os.path.join(self.classpath, self.dfilename)
        # if the 2D completeness update array exists as a .dcomp file load it
        if os.path.exists(path):
            print 'Loading cached dynamic completeness array from "%s".' % path
            self.updates = pickle.load(open(path, 'rb'))
            print 'Dynamic completeness array loaded from cache.'
        else:
            # run Monte Carlo simulation and pickle the resulting array
            print 'Cached dynamic completeness array not found at "%s".' % path
            print 'Beginning dynamic completeness calculations'
            # dynamic completeness values: rows are stars, columns are number of visits
            self.updates = np.zeros((TL.nStars, 5))
            # number of planets to simulate
            nplan = int(2e4)
            # sample quantities which do not change in time
            a, e, p, Rp = PPop.gen_plan_params(nplan)
            a = a.to('AU').value
            # sample angles
            I, O, w = PPop.gen_angles(nplan)
            I = I.to('rad').value
            O = O.to('rad').value
            w = w.to('rad').value
            Mp = PPop.gen_mass(nplan) # M_earth
            rmax = a*(1.+e) # AU
            # sample quantity which will be updated
            M = np.random.uniform(high=2.*np.pi,size=nplan)
            newM = np.zeros((nplan,))
            # population values
            smin = (np.tan(IWA)*TL.dist).to('AU').value
            if np.isfinite(OWA):
                smax = (np.tan(OWA)*TL.dist).to('AU').value
            else:
                smax = np.array([np.max(PPop.arange.to('AU').value)*\
                        (1.+np.max(PPop.erange))]*TL.nStars)
            # fill dynamic completeness values
            for sInd in xrange(TL.nStars):
                mu = (const.G*(Mp + TL.MsTrue[sInd])).to('AU3/day2').value
                n = np.sqrt(mu/a**3) # in 1/day
                # normalization time equation from Brown 2015
                dt = 58.0*(TL.L[sInd]/0.83)**(3.0/4.0)*(TL.MsTrue[sInd]/(0.91*u.M_sun))**(1.0/2.0) # days
                # remove rmax < smin 
                pInds = np.where(rmax > smin[sInd])[0]
                # calculate for 5 successive observations
                for num in xrange(5):
                    if num == 0:
                        self.updates[sInd, num] = TL.comp0[sInd]
                    if not pInds.any():
                        break
                    # find Eccentric anomaly
                    if num == 0:
                        E = eccanom(M[pInds],e[pInds])
                        newM[pInds] = M[pInds]
                    else:
                        E = eccanom(newM[pInds],e[pInds])
                    
                    r1 = a[pInds]*(np.cos(E) - e[pInds])
                    r1 = np.hstack((r1.reshape(len(r1),1), r1.reshape(len(r1),1), r1.reshape(len(r1),1)))
                    r2 = (a[pInds]*np.sin(E)*np.sqrt(1. -  e[pInds]**2))
                    r2 = np.hstack((r2.reshape(len(r2),1), r2.reshape(len(r2),1), r2.reshape(len(r2),1)))
                    
                    a1 = np.cos(O[pInds])*np.cos(w[pInds]) - np.sin(O[pInds])*np.sin(w[pInds])*np.cos(I[pInds])
                    a2 = np.sin(O[pInds])*np.cos(w[pInds]) + np.cos(O[pInds])*np.sin(w[pInds])*np.cos(I[pInds])
                    a3 = np.sin(w[pInds])*np.sin(I[pInds])
                    A = np.hstack((a1.reshape(len(a1),1), a2.reshape(len(a2),1), a3.reshape(len(a3),1)))
                    
                    b1 = -np.cos(O[pInds])*np.sin(w[pInds]) - np.sin(O[pInds])*np.cos(w[pInds])*np.cos(I[pInds])
                    b2 = -np.sin(O[pInds])*np.sin(w[pInds]) + np.cos(O[pInds])*np.cos(w[pInds])*np.cos(I[pInds])
                    b3 = np.cos(w[pInds])*np.sin(I[pInds])
                    B = np.hstack((b1.reshape(len(b1),1), b2.reshape(len(b2),1), b3.reshape(len(b3),1)))
                    
                    # planet position, planet-star distance, apparent separation
                    r = (A*r1 + B*r2) # position vector (AU)
                    d = np.linalg.norm(r,axis=1) # planet-star distance
                    s = np.linalg.norm(r[:,0:2],axis=1) # apparent separation
                    beta = np.arccos(r[:,2]/d) # phase angle
                    Phi = self.PlanetPhysicalModel.calc_Phi(beta*u.rad) # phase function
                    dMag = deltaMag(p[pInds],Rp[pInds],d*u.AU,Phi) # difference in magnitude
                    
                    toremoves = np.where((s > smin[sInd]) & (s < smax[sInd]))[0]
                    toremovedmag = np.where(dMag < dMagMax)[0]
                    toremove = np.intersect1d(toremoves, toremovedmag)
                    
                    pInds = np.delete(pInds, toremove)
                    
                    if num == 0:
                        self.updates[sInd, num] = TL.comp0[sInd]
                    else:
                        self.updates[sInd, num] = float(len(toremove))/nplan
                    
                    # update M
                    newM[pInds] = (newM[pInds] + n[pInds]*dt)/(2*np.pi) % 1 * 2.*np.pi
                    
                if (sInd+1) % 50 == 0:
                    print 'stars: %r / %r' % (sInd+1,TL.nStars)
            # store dynamic completeness array as .dcomp file
            pickle.dump(self.updates, open(path, 'wb'))
            print 'Dynamic completeness calculations finished'
            print 'Dynamic completeness array stored in %r' % path

    def completeness_update(self, TL, sInds, visits, dt):
        """Updates completeness value for stars previously observed by selecting
        the appropriate value from the updates array
        
        Args:
            TL (TargetList module):
                TargetList class object
            sInds (integer array):
                Indices of stars to update
            visits (integer array):
                Number of visits for each star
            dt (astropy Quantity array):
                Time since previous observation
        
        Returns:
            dcomp (float ndarray):
                Completeness values for each star
        
        """
        # if visited more than five times, return 5th stored dynamic 
        # completeness value
        visits[visits > 4] = 4
        dcomp = self.updates[sInds, visits]
        
        return dcomp

    def genC(self, Cpath, nplan, xedges, yedges, steps):
        """Gets completeness interpolant for initial completeness
        
        This function either loads a completeness .comp file based on specified
        Planet Population module or performs Monte Carlo simulations to get
        the 2D completeness values needed for interpolation.
        
        Args:
            Cpath (string):
                path to 2D completeness value array
            nplan (float):
                number of planets used in each simulation
            xedges (float ndarray):
                x edge of 2d histogram (separation)
            yedges (float ndarray):
                y edge of 2d histogram (dMag)
            steps (integer):
                number of simulations to perform
                
        Returns:
            H (float ndarray):
                2D numpy ndarray containing completeness probability density values
        
        """
        
        # if the 2D completeness pdf array exists as a .comp file load it
        if os.path.exists(Cpath):
            print 'Loading cached completeness file from "%s".' % Cpath
            H = pickle.load(open(Cpath, 'rb'))
            print 'Completeness loaded from cache.'
        else:
            # run Monte Carlo simulation and pickle the resulting array
            print 'Cached completeness file not found at "%s".' % Cpath
            print 'Beginning Monte Carlo completeness calculations.'
            
            t0, t1 = None, None # keep track of per-iteration time
            for i in xrange(steps):
                t0, t1 = t1, time.time()
                if t0 is None:
                    delta_t_msg = '' # no message
                else:
                    delta_t_msg = '[%.3f s/iteration]' % (t1 - t0)
                print 'Completeness iteration: %5d / %5d %s' % (i+1, steps, delta_t_msg)
                # get completeness histogram
                h, xedges, yedges = self.hist(nplan, xedges, yedges)
                if i == 0:
                    H = h
                else:
                    H += h
            
            H = H/(self.Nplanets*(xedges[1]-xedges[0])*(yedges[1]-yedges[0]))
                        
            # store 2D completeness pdf array as .comp file
            pickle.dump(H, open(Cpath, 'wb'))
            print 'Monte Carlo completeness calculations finished'
            print '2D completeness array stored in %r' % Cpath
        
        return H, xedges, yedges

    def hist(self, nplan, xedges, yedges):
        """Returns completeness histogram for Monte Carlo simulation
        
        This function uses the inherited Planet Population module.
        
        Args:
            nplan (float):
                number of planets used
            xedges (float ndarray):
                x edge of 2d histogram (separation)
            yedges (float ndarray):
                y edge of 2d histogram (dMag)
        
        Returns:
            h (ndarray):
                2D numpy ndarray containing completeness histogram
        
        """
        
        s, dMag = self.genplans(nplan)
        # get histogram
        h, yedges, xedges = np.histogram2d(dMag, s.to('AU').value, bins=1000,
                range=[[yedges.min(), yedges.max()], [xedges.min(), xedges.max()]])
        
        return h, xedges, yedges

    def genplans(self, nplan):
        """Generates planet data needed for Monte Carlo simulation
        
        Args:
            nplan (integer):
                Number of planets
                
        Returns:
            s (astropy Quantity array):
                Planet apparent separations in units of AU
            dMag (ndarray):
                Difference in brightness
        
        """
        
        PPop = self.PlanetPopulation
        
        nplan = int(nplan)
        
        # sample uniform distribution of mean anomaly
        M = np.random.uniform(high=2.0*np.pi,size=nplan)
        # sample quantities
        a, e, p, Rp = PPop.gen_plan_params(nplan)
        # check if circular orbits
        if np.sum(PPop.erange) == 0:
            r = a
            e = 0.0
            E = M
        else:
            E = eccanom(M,e)
            # orbital radius
            r = a*(1.0-e*np.cos(E))

        beta = np.arccos(1.0-2.0*np.random.uniform(size=nplan))*u.rad
        s = r*np.sin(beta)
        # phase function
        Phi = self.PlanetPhysicalModel.calc_Phi(beta)
        # calculate dMag
        dMag = deltaMag(p,Rp,r,Phi)
        
        return s, dMag

    def comp_per_intTime(self, intTimes, TL, sInds, fZ, fEZ, WA, mode):
        """Calculates completeness for integration time
        
        Args:
            intTimes (astropy Quantity array):
                Integration times
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            fZ (astropy Quantity array):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            fEZ (astropy Quantity array):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
            WA (astropy Quantity):
                Working angle of the planet of interest in units of arcsec
            mode (dict):
                Selected observing mode
                
        Returns:
            comp (array):
                Completeness values
        
        """
        
        # cast inputs to arrays and check
        sInds = np.array(sInds, ndmin=1, copy=False)
        intTimes = np.array(intTimes.value, ndmin=1)*intTimes.unit
        fZ = np.array(fZ.value, ndmin=1)*fZ.unit
        fEZ = np.array(fEZ.value, ndmin=1)*fEZ.unit
        WA = np.array(WA.value, ndmin=1)*WA.unit
        assert len(intTimes) == len(sInds), "intTimes and sInds must be same length"
        assert len(fZ) in [1, len(intTimes)], "fZ must be constant or have same length as intTimes"
        assert len(fEZ) in [1, len(intTimes)], "fEZ must be constant or have same length as intTimes"
        assert len(WA) == 1, "WA must be constant"
        
        dMag = TL.OpticalSystem.calc_dMag_per_intTime(intTimes, TL, sInds, fZ, fEZ, WA, mode).reshape((len(intTimes),))
        smin = (np.tan(TL.OpticalSystem.IWA)*TL.dist[sInds]).to('AU').value
        smax = (np.tan(TL.OpticalSystem.OWA)*TL.dist[sInds]).to('AU').value
        comp = self.EVPOC(smin, smax, 0., dMag)
        
        return comp

    def dcomp_dt(self, intTimes, TL, sInds, fZ, fEZ, WA, mode):
        """Calculates derivative of completeness with respect to integration time
        
        Args:
            intTimes (astropy Quantity array):
                Integration times
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            fZ (astropy Quantity array):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            fEZ (astropy Quantity array):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
            WA (astropy Quantity):
                Working angle of the planet of interest in units of arcsec
            mode (dict):
                Selected observing mode
                
        Returns:
            dcomp (array):
                Derivative of completeness with respect to integration time
        
        """
        
        # cast inputs to arrays and check
        intTimes = np.array(intTimes.value, ndmin=1)*intTimes.unit
        sInds = np.array(sInds, ndmin=1)
        fZ = np.array(fZ.value, ndmin=1)*fZ.unit
        fEZ = np.array(fEZ.value, ndmin=1)*fEZ.unit
        WA = np.array(WA.value, ndmin=1)*WA.unit
        assert len(intTimes) == len(sInds), "intTimes and sInds must be same length"
        assert len(fZ) in [1, len(intTimes)], "fZ must be constant or have same length as intTimes"
        assert len(fEZ) in [1, len(intTimes)], "fEZ must be constant or have same length as intTimes"
        assert len(WA) == 1, "WA must be constant"
        
        dMag = TL.OpticalSystem.calc_dMag_per_intTime(intTimes, TL, sInds, fZ, fEZ, WA, mode).reshape((len(intTimes),))
        smin = (np.tan(TL.OpticalSystem.IWA)*TL.dist[sInds]).to('AU').value
        smax = (np.tan(TL.OpticalSystem.OWA)*TL.dist[sInds]).to('AU').value
        ddMag = TL.OpticalSystem.ddMag_dt(intTimes, TL, sInds, fZ, fEZ, WA, mode).reshape((len(intTimes),))
        dcomp = np.zeros(len(intTimes))
        for k,(dm,ddm) in enumerate(zip(dMag,ddMag)):
            dcomp[k] = interpolate.InterpolatedUnivariateSpline(self.xnew,self.EVPOCpdf(self.xnew,dm),ext=1).integral(smin[k],smax[k])
        
        return dcomp*ddMag