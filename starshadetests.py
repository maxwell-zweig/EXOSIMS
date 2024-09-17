import EXOSIMS.Observatory.KulikStarshade as KulikStarshade
import EXOSIMS.TargetList.KnownRVPlanetsTargetList as KnownRVPlanetsTargetList
import numpy as np

starshade = KulikStarshade.KulikStarshade(mode="impulsive", dynamics=0, exponent=8, precompfname="haloEnergy", starShadeRadius = 10)
targets = KnownRVPlanetsTargetList.KnownRVPlanetsTargetList(modules= {"StarCatalog": "EXOCAT1", "OpticalSystem": " ", "ZodiacalLight": "  ", "PostProcessing": " ", "Completeness": " ", "PlanetPopulation" : "KnownRVPlanets", "BackgroundSources" : " ", "PlanetPhysicalModel" : " "})

dV = starshade.calculate_dV(targets, 5, np.array([21]), np.ones((1, 50)), starshade.equinox)

print(dV)
