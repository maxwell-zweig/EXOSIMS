import EXOSIMS,EXOSIMS.MissionSim,os.path
scriptfile = os.path.join(EXOSIMS.__path__[0],'Scripts','sampleScript_occulter.json')
sim = EXOSIMS.MissionSim.MissionSim(scriptfile)

sim.run_sim()
DRM = sim.SurveySimulation.DRM