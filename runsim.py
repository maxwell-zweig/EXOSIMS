import EXOSIMS,EXOSIMS.MissionSim,os.path
scriptfile = os.path.join(EXOSIMS.__path__[0],'Scripts','TestScripts/07_KulikStarshade.json')
sim = EXOSIMS.MissionSim.MissionSim(scriptfile)

sim.run_sim()
DRM = sim.SurveySimulation.DRM