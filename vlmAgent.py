import pdb
from collections import Counter

import numpy as np

import habitat_sim
import cv2
from utils import *


class VLMAgent:

    def __init__(self, **kwargs):
        pass

    def call(self, visual_prompt: np.array, text_prompt: str, num_samples):
        out = {}
        for i in range(num_samples):
            out[i+1] = ['right', 'neutral', 'in front']
        return out
