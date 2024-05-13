import pdb
from collections import Counter

import numpy as np
import cv2
from src.utils import *


class VLMAgent:
    """
    Trivial agent for testing
    """
    def __init__(self, **kwargs):
        pass

    def call(self, visual_prompt: np.array, text_prompt: str, num_samples):
        out = {}
        for i in range(num_samples):
            out[i+1] = ['right', 'neutral', 'in front']
        return out
