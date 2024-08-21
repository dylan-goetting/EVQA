import json
import os
from sqlite3 import DatabaseError
import sys
import pdb
import pickle
import random
from matplotlib import pyplot as plt
import numpy as np
import datetime
import cv2
import ast
import pandas as pd
from PIL import Image
from src.utils import *
from vlm import VLMAgent
from src.annoatedSimulator import AnnotatedSimulator
import habitat_sim


class DefaultAgent:
    def __init__(self, sys_instruction=None):
        self.sys_instruction = sys_instruction

    def act(self, observation):
        # Implement the logic for the agent's action based on the observation
        pass