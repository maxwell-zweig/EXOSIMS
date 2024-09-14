import EXOSIMS.Observatory.KulikStarshade as KulikStarshade
import EXOSIMS.TargetList.KnownRVPlanetsTargetList as KnownRVPlanetsTargetList
import numpy as np

starshade = KulikStarshade.KulikStarshade(mode="impulsive", dynamics=0, exponent=10, precompfname="haloImpulsive", starShadeRadius = 10)
targets = KnownRVPlanetsTargetList.KnownRVPlanetsTargetList(modules= {"StarCatalog": "EXOCAT1", "OpticalSystem": " ", "ZodiacalLight": "  ", "PostProcessing": " ", "Completeness": " ", "PlanetPopulation" : "KnownRVPlanets", "BackgroundSources" : " ", "PlanetPhysicalModel" : " "})

dV = starshade.calculate_dV(targets, 5, np.array([13, 14, 17, 20]), np.ones((4, 50)) + np.random.rand(4,50), starshade.equinox)

print(dV)
