import scipy 
import EXOSIMS.Observatory.ObservatoryL2Halo as ObservatoryL2Halo
import EXOSIMS.util.OrbitVariationalFirstOrder  as ObsPreComp
import math
import astropy.units as u
import numpy as np

trvFileName = "./" + "haloImpulsive" + "_trvs.mat"

trvmat = list(scipy.io.loadmat(trvFileName).values())[-1]
# period
T = trvmat[-1, 0]

# print(np.array(trvmat).shape)

#print(T)
# Take off last element which is same as first element up to integration error tolerances (periodicity)
# trvmat = trvmat[:-1]
obs = ObservatoryL2Halo.ObservatoryL2Halo(use_alt=True)
canonical_unit = u.year / (2 * math.pi)
# print(canonical_unit.value * T)
# print(trvmat[:, 0])
# print(obs.equinox)
print(obs.haloVelocity(obs.equinox).value / (2 * math.pi))
print(obs.haloPosition(obs.equinox))

print(canonical_unit * T)

print(obs.haloVelocity(obs.equinox + canonical_unit * T).value / (2 * math.pi))
print(obs.haloPosition(obs.equinox + canonical_unit * T))



