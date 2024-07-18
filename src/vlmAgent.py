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
        self.name = "not implemented"

    def call(self, visual_prompt: np.array, text_prompt: str, num_samples):
        return ""
