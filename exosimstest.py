import EXOSIMS,EXOSIMS.MissionSim,os.path
from EXOSIMS.StarCatalog.GaiaCat1 import GaiaCat1

import EXOSIMS.Prototypes.Observatory
from EXOSIMS.Observatory.SotoStarshade import SotoStarshade as SotoStarshade
from EXOSIMS.util.get_dirs import get_downloads_dir
import shutil


starshade = SotoStarshade()

downloads_path = get_downloads_dir()
if not os.path.exists(downloads_path + "/GaiaCatGVTest.gz"):
    shutil.copy(
        "tests/TestSupport/test-scripts/GaiaCatGVTest.gz", downloads_path
    )
    catalog = GaiaCat1(catalogfile="GaiaCatGVTest.gz")
    