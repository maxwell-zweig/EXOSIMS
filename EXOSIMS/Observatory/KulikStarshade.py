from EXOSIMS.Observatory.ObservatoryL2Halo import ObservatoryL2Halo
from EXOSIMS.Prototypes.TargetList import TargetList
import numpy as np
import astropy.units as u
from scipy.integrate import solve_bvp
import astropy.constants as const
import hashlib
import scipy.optimize as optimize
import scipy.interpolate as interp
import time
import os
import pickle
from scipy.linalg import lu_factor, lu_solve
from STMint.STMint import STMint
from sympy import *
import scipy 
from EXOSIMS.util.OrbitVariationalFirstOrder import OrbitVariationalDataFirstOrder
from EXOSIMS.util.OrbitVariationalDataSecondOrder import OrbitVariationalDataSecondOrder
import math
from astropy import units as u 

EPS = np.finfo(float).eps

class KulikStarshade(ObservatoryL2Halo):


    """StarShade Observatory class
    This class is implemented at L2 and contains all variables, functions,
    and integrators to calculate occulter dynamics.
    """
    def __init__(self, mode="impulsive", dynamics=0, exponent=10, precompfname="haloImpulsive", starShadeRadius = 10, **specs):
        """Initializes StarShade class. Checks if variational data has already been precomputed for given mode and dynamics.

        Args:
            orbit_datapath (float 1x3 ndarray):
                TargetList class object
            dynamics (integer):
                0, 1, 2, 3. 0 for default CRTBP. 1 for SRP. 2 for moon. 3 for SRP and moon.
            mode (string):
                One of "energyOptimal" or "impulsive"
        """
        
        ObservatoryL2Halo.__init__(self, use_alt=True, **specs)

        self.mode = mode
        self.dynamics = dynamics
        self.exponent = exponent
        self.precompfname = precompfname

        # should be given in m -- converting to km 
        self.starShadeRad = starShadeRadius / 1000 

        if mode=="energyOptimal":
            if dynamics == 0:
                def optControlDynamics():
                    x, y, z, vx, vy, vz, lx, ly, lz, lvx, lvy, lvz, En = symbols(
                        "x,y,z,vx,vy,vz,lx,ly,lz,lvx,lvy,lvz,En"
                    )
                    mu = 3.00348e-6
                    mu1 = 1.0 - mu
                    mu2 = mu
                    r1 = sqrt((x + mu2) ** 2 + (y**2) + (z**2))
                    r2 = sqrt((x - mu1) ** 2 + (y**2) + (z**2))
                    U = (-1.0 / 2.0) * (x**2 + y**2) - (mu1 / r1) - (mu2 / r2)
                    dUdx = diff(U, x)
                    dUdy = diff(U, y)
                    dUdz = diff(U, z)

                    RHS = Matrix(
                        [vx, vy, vz, ((-1 * dUdx) + 2 * vy), ((-1 * dUdy) - 2 * vx), (-1 * dUdz)]
                    )

                    variables = Matrix([x, y, z, vx, vy, vz, lx, ly, lz, lvx, lvy, lvz, En])

                    dynamics = Matrix(
                        BlockMatrix(
                            [
                                [RHS - Matrix([0, 0, 0, lvx, lvy, lvz])],
                                [
                                    -1.0
                                    * RHS.jacobian(Matrix([x, y, z, vx, vy, vz]).transpose())
                                    * Matrix([lx, ly, lz, lvx, lvy, lvz])
                                ],
                                [0.5 * Matrix([lvx**2 + lvy**2 + lvz**2])],
                            ]
                        )
                    )
                    # return Matrix([x,y,z,vx,vy,vz]), RHS
                    return variables, dynamics
           
                fileName = self.precompfname
                trvFileName =  fileName + "_trvs.mat"
                STMFileName =  fileName + "_STMs.mat"
                STTFileName =  fileName + "_STTs.mat"
                # initial conditions for a sun-earth halo orbit
                # with zero initial conditions for costates and energy
                ics = [
                    1.00822114953991,
                    0.0,
                    -0.001200000000000000,
                    0.0,
                    0.010290010931740649,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
                T = 3.1002569555488506
                # use 2^8 subdivisions when calculating the values of the STM
                exponent = self.exponent

                # precompute variational data if data file does not already exist
                if not os.path.isfile(trvFileName):
                    variables, dynamics = optControlDynamics()
                    cur_time = time.time()
                    print("Precomputing variational data")
                    threeBodyInt = STMint(variables, dynamics, variational_order=2)
                    t_step = T / 2.0**exponent
                    curState = ics
                    states = [ics]
                    # STMs = [np.identity(12)]
                    # STTs = [np.zeros((12,12))]
                    STMs = []
                    STTs = []
                    tVals = np.linspace(0, T, num=2**exponent + 1)
                    for i in range(2**exponent):
                        print(f"Integrating: {i}")
                        [state, STM, STT] = threeBodyInt.dynVar_int2(
                            [0, t_step], curState, output="final", max_step=0.001
                        )
                        states.append(state)
                        STMs.append(STM[:12, :12])
                        STTs.append(STT[12, :12, :12])
                        curState = state
                    scipy.io.savemat(
                        trvFileName, {"trvs": np.hstack((np.transpose(np.array([tVals])), states))}
                    )
                    scipy.io.savemat(STMFileName, {"STMs": STMs})
                    scipy.io.savemat(STTFileName, {"STTs": STTs})

                                # load data from file
                trvmat = list(scipy.io.loadmat(trvFileName).values())[-1]
                # period
                T = trvmat[-1, 0]
                print(T)
                # Take off last element which is same as first element up to integration error tolerances (periodicity)
                trvmat = trvmat[:-1]

                STMmat = list(scipy.io.loadmat(STMFileName).values())[-1]
                STTmat = list(scipy.io.loadmat(STTFileName).values())[-1]
                # initialize object used for computation
                self.orb = OrbitVariationalDataSecondOrder(STTmat, STMmat, trvmat, T, exponent)        
            elif dynamics == 1:
                raise Exception('Unimplemented')
            elif dynamics == 2:
                raise Exception('Unimplemented')
            elif dynamics == 3:
                raise Exception('Unimplemented')             

        elif mode=="impulsive":
            if dynamics == 0:
                def optControlDynamics():
                    x, y, z, vx, vy, vz = symbols("x,y,z,vx,vy,vz")
                    mu = 3.00348e-6
                    mu1 = 1.0 - mu
                    mu2 = mu
                    r1 = sqrt((x + mu2) ** 2 + (y**2) + (z**2))
                    r2 = sqrt((x - mu1) ** 2 + (y**2) + (z**2))
                    U = (-1.0 / 2.0) * (x**2 + y**2) - (mu1 / r1) - (mu2 / r2)
                    dUdx = diff(U, x)
                    dUdy = diff(U, y)
                    dUdz = diff(U, z)

                    RHS = Matrix(
                        [vx, vy, vz, ((-1 * dUdx) + 2 * vy), ((-1 * dUdy) - 2 * vx), (-1 * dUdz)]
                    )
                    variables = Matrix([x, y, z, vx, vy, vz])
                    return variables, RHS
                
                # store precomputed data in the following files
                fileName = self.precompfname
                trvFileName = f"{fileName}_trvs.mat"
                STMFileName = f"{fileName}_STMs.mat"
                # initial conditions for a sun-earth halo orbit
                # with zero initial conditions for costates and energy
                ics = [
                    1.00822114953991,
                    0.0,
                    -0.001200000000000000,
                    0.0,
                    0.010290010931740649,
                    0.0,
                ]
                T = 3.1002569555488506

                exponent = self.exponent

                # precompute variational data if data file does not already exist
                if not os.path.isfile(trvFileName):
                    variables, dynamics = optControlDynamics()
                    cur_time = time.time()
                    print("Precomputing variational data")
                    threeBodyInt = STMint(variables, dynamics, variational_order=1)
                    t_step = T / 2.0**exponent
                    print(time.time() - cur_time)
                    curState = ics
                    states = [ics]
                    # STMs = [np.identity(12)]
                    # STTs = [np.zeros((12,12))]
                    STMs = []
                    tVals = np.linspace(0, T, num=2**exponent + 1)
                    for i in range(2**exponent):
                        print(f"Integrating: {i}")
                        [state, STM] = threeBodyInt.dynVar_int(
                            [0, t_step], curState, output="final", max_step=0.001
                        )
                        states.append(state)
                        STMs.append(STM[:6, :6])
                        curState = state
                    scipy.io.savemat(
                        trvFileName, {"trvs": np.hstack((np.transpose(np.array([tVals])), states))}
                    )
                    scipy.io.savemat(STMFileName, {"STMs": STMs})
                # load data from file
                trvmat = list(scipy.io.loadmat(trvFileName).values())[-1]
                # period
                T = trvmat[-1, 0]
                # Take off last element which is same as first element up to integration error tolerances (periodicity)
                trvmat = trvmat[:-1]
                STMmat = list(scipy.io.loadmat("haloImpulsive_STMs.mat").values())[-1]
                # initialize object used for computation
                self.orb = OrbitVariationalDataFirstOrder(STMmat, trvmat, T, exponent)
            elif dynamics == 1:
                raise Exception('Unimplemented')
            elif dynamics == 2:
                raise Exception('Unimplemented')
            elif dynamics == 3:
                raise Exception('Unimplemented')      
        else:
            raise Exception('Mode must be one of "energyOptimal" or "impuslive"')

    def calculate_dV(self, TL, old_sInd, sInds, slewTimes, tmpCurrentTimeAbs):
        IWA = TL.OpticalSystem.IWA
        d = self.starShadeRad / math.tan(IWA.value * math.pi / (180 * 3600))  # confirm units 
        
        slewTimes += np.random.rand(slewTimes.shape[0], slewTimes.shape[1]) / 10000
        if old_sInd is None:
            dV = np.zeros(slewTimes.shape)
        else:
            dV = np.zeros(slewTimes.shape)
            if isinstance(slewTimes, u.Quantity):
                badSlews_i, badSlew_j = np.where(slewTimes.value < self.occ_dtmin.value)  
            else:
                badSlews_i, badSlew_j = np.where(slewTimes < self.occ_dtmin.value)
            t0 = tmpCurrentTimeAbs

            # gets inital target star positions in heliocentric ecliptic intertial frame
            # TL.starprop handles a list of times
            starPost0 = TL.starprop(old_sInd, t0, eclip=True) # confirm units

            # calculating starshade position at t0 in au, in inertial CRTBP frame, units don't matter since normalizing to a unitvector 
            obsPost0 = self.orbit(t0, eclip=True) # confirm units
            starShadePost0InertRel = d * (starPost0 - obsPost0) / np.linalg.norm(starPost0 - obsPost0) * 6.68459e-9

            # converting t0 to CRTBP canonical time units --- going to need to implement a time shift to match Jackson's CRTBP implementation /w the one
            # or calculate Halo Orbit positions by interpolating the state done in Jackson's code
            # the units of starPostf and obPostf don't actually matter for this calculation, since we are normalizing 
            canonical_unit = 365.2515 / (2 * math.pi) # in days  # confirm units

            t0Can = t0[0].value / canonical_unit  # confirm units

            transformmat0 = np.array([[np.cos(t0Can), np.sin(t0Can), 0], [-np.sin(t0Can), np.cos(t0Can), 0], [0, 0, 1]])
            starShadePost0SynRel = transformmat0 @ starShadePost0InertRel.squeeze() 
            
            tfs = tmpCurrentTimeAbs + slewTimes
            tfs_flattened = tfs.flatten()
            starPosttfs = TL.starprop(sInds, tfs_flattened, eclip=True)  # confirm flattening done correctly 
            # starposttfs is a num stars by num times by 3 array, unless all the times are the same 
            # starposttfs are units of parsecs, whereas absoervatory positions are in AU, this is getting normalized out anyways though 
            # fixing the output of starPosttfs if all times are the same -- TL.starprop is implemented in such a silly way 
            
            # times are the same value so being reduced to a scalar 
            obsPosttfs = self.orbit(tfs_flattened, eclip=True)
            for t in range(slewTimes.shape[1]):
                for i in range(slewTimes.shape[0]):
                    # gets final target star positions in heliocentric ecliptic inertial frame 
                    starPostf = starPosttfs[i * slewTimes.shape[1] + t, i].value
                    obsPostf = obsPosttfs[i * slewTimes.shape[1] + t].value

                    starShadePostfInertRel = d * (starPostf - obsPostf) / np.linalg.norm(starPostf - obsPostf) * 6.68459e-9

                    tfCan = tfs_flattened[i * slewTimes.shape[1] + t].value / canonical_unit

                    transformmatf = np.array([[np.cos(tfCan), np.sin(tfCan), 0], [-np.sin(tfCan), np.cos(tfCan), 0], [0, 0, 1]])
                    starShadePostfSynRel = transformmatf @ starShadePostfInertRel


                    precomputeData = self.orb.precompute_lu(t0Can, tfCan)
                    dV[i, t] = self.orb.solve_deltaV_convenience(precomputeData, starShadePost0SynRel, starShadePostfSynRel)
            
            dV[badSlews_i, badSlew_j] = np.Inf

            # must convert from AU / canonical time unit to m / s 
        au_to_m = u.au.to(u.m)
        day_to_s = u.day.to(u.s)
        dV = dV * u.AU / (365.2515 / (2 * math.pi) * u.day) 
        dV = (dV * au_to_m / day_to_s).to(u.m / u.s)
        return dV



