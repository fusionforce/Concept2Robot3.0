#!/usr/bin/env python3
import time
import math
from datetime import datetime
from time import sleep
import numpy as np
import random
import cv2
import os
import argparse
import torch

import sys
sys.path.append('./Eval')
sys.path.append('./')

from env_56 import Engine56
from utils_env import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code

class Engine160(Engine56):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine160,self).__init__(worker_id, opti, p_id, taskId=taskId, maxSteps=maxSteps, n_dmps=n_dmps, cReward=cReward)
        self.opti = opti

