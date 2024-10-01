import ast
import logging
from cv2 import log
import numpy as np

import csv
import gzip
import json
import math
import os
import pdb
import random
from sqlite3 import DatabaseError
from turtle import update
from habitat.core import agent
from networkx import shortest_path
import numpy as np
import pandas as pd
from PIL import Image
from regex import D, E
from sympy import im
from src.utils import *
from src.vlm import VLM, GPTModel, GeminiModel
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis
import concurrent.futures
import numpy as np
from src.utils import *


class ActionDistribution2D:

    def __init__(self, image_width, image_height, mean=None, std_dev=None):

        self.image_width = image_width
        self.image_height = image_height
        # Centering at the bottom-center of the image (robot's position)
        self.mean = mean if mean is not None else np.array([image_width / 2, image_height/2])
        self.std_dev = std_dev if std_dev is not None else np.array([image_width / 4, image_height / 4])

    def sample(self, num_samples):

        return np.random.normal(self.mean, self.std_dev, (num_samples, 2))

    def fit_to_selected_actions(self, selected_actions):
        if selected_actions == []:
            return
        # Update mean to the mean of the selected actions
        self.mean = np.mean(selected_actions, axis=0)
        
        # Update standard deviation to the standard deviation of the selected actions
        self.std_dev = np.std(selected_actions, axis=0)


class PIVOT:
    
    def __init__(self, vlm: GeminiModel, fov, image_dim):
        self.image_width = image_dim[1]
        self.image_height = image_dim[0]
        self.vlm = vlm
        self.fov = fov
        fov_radians = np.deg2rad(fov)
        self.focal_length = (self.image_width / 2) / np.tan(fov_radians / 2)

    
    def run(self, rgb_image, depth_image, instruction, agent_state, sensor_state, num_iter=3, num_parralel=3, num_samples=8, goal_image=None):

        image_width, image_height = self.image_width, self.image_height
        action_dist_2d = ActionDistribution2D(image_width, image_height)
        start_pt = [0, 0, 0]
        global_p = local_to_global(agent_state.position, agent_state.rotation, start_pt)
        local_point = global_to_local(sensor_state.position, sensor_state.rotation, global_p)
        if local_point[2] > 0:
            return None
        point_3d = [local_point[0], -local_point[1], -local_point[2]] #inconsistency between habitat camera frame and classical convention
        if point_3d[2] == 0:
            point_3d[2] = 0.0001
        x = self.focal_length * point_3d[0] / point_3d[2]
        x_pixel = int(image_width / 2 + x)

        y = self.focal_length * point_3d[1] / point_3d[2]
        y_pixel = int(image_height / 2 + y)

        self.start_pxl = [x_pixel, y_pixel]
        log_images = []
        for itr in range(num_iter):
        # Sample 5 actions from the initial distribution
            ns = num_samples - 2*itr
            sampled_actions = action_dist_2d.sample(ns)
            # pdb.set_trace()
            actions, annotated_image = self.annotate_on_image(rgb_image, sampled_actions)

            log_images.append(annotated_image)
            K = 4 - itr
            if itr == num_iter - 1:
                K = 1
            prompt = (
            'I am a wheeled robot that cannot go over objects. This is the image I’m seeing right '
            'now. I have annotated it with numbered circles. Each number represent a general '
            'direction I can follow. Now you are a five-time world-champion navigation agent and '
            f'your task is to tell me which circle I should pick for the task of: \n{instruction}\n'
            f'Choose {K} of the best candidate numbers. Do NOT choose routes that goes through objects. '
            'Skip analysis and provide your answer at the end in a json file of this form:\n'
            '{"points": [] }'
            )
            ims = [annotated_image]
            if goal_image is not None:
                ims.append(goal_image)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for _ in range(num_parralel):

                    future = executor.submit(self.vlm.call, ims, prompt)
                    futures.append(future)
                
                results = []
                for future in concurrent.futures.as_completed(futures):
                    result, _ = future.result()
                    results.append(result)
                
            all_points = []
            for response in results:
                try:
                    eval_resp = ast.literal_eval(response[response.rindex('{'):response.rindex('}')+1])
                    all_points += eval_resp['points']
                except Exception as e:
                    logging.error(f'pivot error parsing', e)
                    print(e)

            all_points = set(all_points)
            if len(all_points) == 0:
                selected_actions = []
            else:
                selected_actions = [actions[i-1] for i in all_points if i <= len(actions)]
            action_dist_2d.fit_to_selected_actions(selected_actions)
        if selected_actions == []:
            return[('forward', 0.2)], log_images
        acts = self.get_actions(selected_actions[0], depth_image)

        return acts, log_images
    
    def get_actions(self, end_px, depth_image):
        end_px = [int(end_px[0]), int(end_px[1])]
        image_width = depth_image.shape[1]
        depth_value = depth_image[end_px[1], end_px[0]]
        z = depth_value
        x = (end_px[0] - image_width / 2) * z / self.focal_length

        mag = np.sqrt(x**2 + z**2)
        theta = np.arctan2(x, z)
        return (['rotate', -theta], ['forward', min(mag, 2.3)],)

    def annotate_on_image(self, rgb_image, sampled_actions):
        scale_factor = rgb_image.shape[0]/1080
        annotated_image = rgb_image.copy()
        action_name = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (0, 0, 0) 
        text_size = 2*scale_factor
        text_thickness = math.ceil(2*scale_factor)
        annotated = []
        for end_px in sampled_actions:

            # Convert the action from pixel coordinates to the agent's coordinate frame

            min_dist = float('inf')
            for annotated_px in annotated:
                dist = np.linalg.norm(np.array(end_px) - np.array(annotated_px))
                min_dist = min(min_dist, dist)

            if min_dist > int(100*scale_factor) and 0.05 * rgb_image.shape[1] <= end_px[0] <= 0.95 * rgb_image.shape[1] and 0.05 * rgb_image.shape[0] <= end_px[1] <= 0.95 * rgb_image.shape[0]:
                cv2.arrowedLine(annotated_image, tuple(self.start_pxl), (int(end_px[0]), int(end_px[1])), (255, 0, 0), math.ceil(5*scale_factor), tipLength=0.)
                text = str(action_name) 
                (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
                circle_center =  (int(end_px[0]), int(end_px[1]))
                circle_radius = max(text_width, text_height) // 2 + math.ceil(15*scale_factor)

                cv2.circle(annotated_image, circle_center, circle_radius, (255, 255, 255), -1)
                cv2.circle(annotated_image, circle_center, circle_radius, (255, 0, 0), math.ceil(2*scale_factor))
                text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
                cv2.putText(annotated_image, text, text_position, font, text_size, text_color, text_thickness)
                annotated.append(end_px)
                action_name += 1
        # Image.fromarray(annotated_image, mode='RGB').save('logs/temp.png')
        # print(f'annotated {len(annotated)} images, action name = {action_name-1}')
        return annotated, annotated_image

