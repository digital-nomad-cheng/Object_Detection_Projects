import os, sys
sys.path.append('.')

import config
from retinaface import RetinaFace

cfg = config.cfg_mnet
retinaface = RetinaFace(cfg)
