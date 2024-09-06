import csv
import gzip
import json
from math import e
import math
import os
from sqlite3 import DatabaseError
import sys
import pdb
import pickle
import random
from turtle import distance, position
from typing import Counter
from habitat.tasks.nav.nav import TopDownMap
from habitat.utils.visualizations import maps
from matplotlib import pyplot as plt
import numpy as np
import cv2
import ast
import pandas as pd
from PIL import Image
from src.utils import *
from src.vlm import VLM, GPTModel, GeminiModel
from src.annoatedSimulator import AnnotatedSimulator
import habitat_sim
import cv2
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis
from scipy.ndimage import map_coordinates
import concurrent.futures

class DynamicBench: 

    task = 'Not defined'

    def __init__(self, sim_kwargs, vlm_agent: VLM, exp_kwargs, outer_run_name):

        self.sim_kwargs = sim_kwargs
        self.vlm = vlm_agent
        self.map_vlm = GeminiModel('gemini-1.5-pro', 'You are an assistant that specializes in maps. You analyze the map and provide action for the agent to take')
        # self.map_vlm = GPTModel('gpt-4o', sys_instruction='You are an assistant that specializes in maps. You analyze the map and provide action for the agent to take')

        self.init_pos = None
        self.exp_kwargs = exp_kwargs

        self.df = pd.DataFrame({})
        self.random_seed = sim_kwargs['random_seed']
        self.outer_run_name = self.task + '_' + outer_run_name
        self.curr_run_name = "Not started"
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.annotatedSimulator = None
        self.setup_experiment(**self.exp_kwargs)

    def setup_experiment(self, **exp_kwargs):
        raise NotImplementedError

    def run_experiment(self, outer_loop, inner_loop, log_freq, **run_kwargs):
        for i in range(outer_loop):
            try:
                self.run_trajectory(inner_loop, log_freq, **run_kwargs)
            except DatabaseError as e:
                print(e)
                print(f'Moving on to iter {i+1}')
                try:
                    self.annotatedSimulator.sim.close()
                except:
                    pass

    def run_trajectory(self, inner_loop, log_freq, **run_kwargs):
        self.step = 0
        self.init_pos = None
        self.df = pd.DataFrame({})
        self.log_freq = log_freq
        obs = self.setup_run(**run_kwargs)
        self.turned = -3
        topdown_map = maps.get_topdown_map_from_sim(self.annotatedSimulator.sim, map_resolution=2048)
        recolor_map = np.array(
        [[40, 40, 40], [200, 200, 200], [0, 0, 0]], dtype=np.uint8)
        topdown_map = recolor_map[topdown_map]
        
        is_grey = topdown_map == (200, 200, 200)
        grey_indices = np.where(is_grey)
        min_x, max_x = np.min(grey_indices[1]), np.max(grey_indices[1])
        min_y, max_y = np.min(grey_indices[0]), np.max(grey_indices[0])

        padding = 50
        min_x = max(min_x - padding, 0)
        max_x = min(max_x + padding, topdown_map.shape[1] - 1)
        min_y = max(min_y - padding, 0)
        max_y = min(max_y + padding, topdown_map.shape[0] - 1)
        self.croppings = (min_x, max_x, min_y, max_y)
        # Crop the topdown_map to the bounding box with padding

        self.topdown_map = topdown_map
        self.new_topdown_map = topdown_map.copy()


        print(f'\n===================STARTING RUN: {self.curr_run_name} ===================\n')
        for _ in range(inner_loop):
            try:
                print('STEP ', self.step)
                actions = self.step_env(obs)
                if actions is None:
                    break
                obs = self.annotatedSimulator.step(actions)
                self.step += 1
            except Exception as e:
                print(e)
                print('ERROR OCCURRED')
                
        self.post_run()
    
    def setup_run(self, **run_kwargs):
        raise NotImplementedError

    def step_env(self, obs):
        raise NotImplementedError

    def post_run_log(self):
        s = self.df['agent_location']
        pairs = list(zip(s[:-1], s[1:]))

        for loc1, loc2 in pairs:
            c1 = self.toGrid(loc1)
            c2 = self.toGrid(loc2)
            cv2.arrowedLine(self.new_topdown_map, c1, c2, (0, 150, 0), 10)

        path = f'logs/{self.outer_run_name}/{self.curr_run_name}/step_FINAL'
        os.makedirs(path)
        im = Image.fromarray(self.new_topdown_map, mode='RGB')
        im.save(f'{path}/final_map.png')

    def post_run(self):
        self.post_run_log()
        self.df.to_pickle(f'logs/{self.outer_run_name}/{self.curr_run_name}/df_results.pkl')
        self.vlm.reset()
        self.annotatedSimulator.sim.close()
        self.get_costs()
        print('\n===================RUN COMPLETE===================\n')
        gif(f'logs/{self.outer_run_name}/{self.curr_run_name}')
        print('saved gif')

    def log(self, images, response, success, metadata, copy_images=[]):
        
        path = f'logs/{self.outer_run_name}/{self.curr_run_name}/step{self.step}'
        if success == 0:
            path += '_ERROR'
        os.makedirs(path)
        for ndx, im in enumerate(images):
            im.save(f'{path}/image{ndx}.png')
        for ndx, im in enumerate(copy_images):
            im.save(f'{path}/copy_image{ndx}.png')
        with open(f'{path}/details.txt', 'w') as file:
            file.write(f'[MODEL OUTPUT]\n{response}\n\n')
     
            if success:
                for k, v in metadata.items():
                    file.write(f'{k}\n{v}\n\n')

    def parse_response(self, response):
        try:
            eval_resp = ast.literal_eval(response[response.rindex('{'):response.rindex('}')+1])
            if isinstance(eval_resp, dict):
                return eval_resp
            else:
                return {'action': list(eval_resp)[0]}
        except (ValueError, SyntaxError):
            return {'action': -10}


    def get_agent_state(self, agent_id=0):
        return self.annotatedSimulator.sim.get_agent(agent_id).get_state()
    
    def set_state(self, pos=None, quat=None, agent_id=0):
        if pos is None:
            pos = self.init_pos 
        if quat is None:
            quat = self.default_quat

        init_state = habitat_sim.AgentState()
        init_state.position = pos
        init_state.rotation = quat
        self.annotatedSimulator.sim.get_agent(agent_id).set_state(init_state)
    
    def agent_self_consitency(self, prompt, images, original_row, consistency):
        action_counter = {}
        num_calls = 0
        while True:
            num_calls += 1
            
            resp, performance = self.vlm.call_chat(self.run_metadata['history'], images, prompt, add_timesteps_prompt=self.run_metadata['add_timesteps_prompt'], step=self.step)
            self.total_input_tokens += performance['input_tokens']
            self.total_output_tokens += performance['tokens_generated']

            metadata = {}
            row = original_row.copy()
            try:
                resp_dict = self.parse_response(resp)
                row['actions'] = resp_dict['action']
                if row['actions'] == 0:
                    if self.step - self.turned < 3:
                        row['actions'] = -10
                    else:
                        self.turned = self.step
            except (IndexError, KeyError, TypeError) as e:
                print(e)
                row['success'] = 0
            finally:
                row.update(resp_dict)
                metadata['ACTIONS'] = row['actions']
                metadata['PROMPT'] = prompt

            if row['actions'] in action_counter:
                action_counter[row['actions']]+= 1
            else:
                action_counter[row['actions']] = 1
            
            if action_counter[row['actions']] == consistency:
                print(f'Stepping, took {num_calls} calls')
                break
            else:
                if row['success']==1:
                    self.vlm.rewind()
        row['num_calls'] = num_calls

        return row, metadata, resp

    def get_sensor_images(self, obs, convert=False):
        ims = [obs[f'color_sensor_{sensor}']['image'] for sensor in self.annotatedSimulator.sensors]
        if convert:
            images = []
            for im in ims:
                if im.shape[-1] == 4:
                    im = im[:, :, 0:3]
                images.append(Image.fromarray(im, mode='RGB'))
            return images
        return ims

    def get_costs(self):
        self.vlm.get_spend()
        print('\n')
        # print(f'GPT Mini would cost: {np.round(self.total_input_tokens*0.15/1000000 + self.total_output_tokens*0.6/1000000, 2)}')
        # print(f'GPT 4o would cost: {np.round(self.total_input_tokens*5/1000000 + self.total_output_tokens*15/1000000, 2)}')
        # print(f'Gemini 1.5pro would cost: {np.round(self.total_input_tokens*3.5/1000000 + self.total_output_tokens*10.50/1000000, 2)}')
        # print(f'Gemini flash would cost: {np.round(self.total_input_tokens*0.35/1000000 + self.total_output_tokens*0.150/1000000, 2)}')
        
    def agent_frame_to_image_coords(self, point, agent_state, sensor_state, resolution=None):
        global_p = local_to_global(agent_state.position, agent_state.rotation, point)
        camera_pt = global_to_local(sensor_state.position, sensor_state.rotation, global_p)
        if camera_pt[2] > 0:
            return None
        return self.annotatedSimulator.project_2d(camera_pt, resolution)

    def get_arrow_options(self, depth_image, agent_state, sensor_state, rnge=1.5):
        if self.run_metadata['uniform']:
            return self.run_metadata['points']
        height_map = depth_to_height1(depth_image, self.annotatedSimulator.fov, sensor_state.position, 
                                      sensor_state.rotation, )
        height_map = abs(height_map - (agent_state.position[1] - 0.06)) < 0.12
        
        arrowData = []
        points = [(1, val) for val in np.linspace(-rnge, rnge, 30)]
        start = self.agent_frame_to_image_coords([0, 0, 0], agent_state, sensor_state, resolution = depth_image.shape)
        arrowData = []
        for _, theta in points:
            
            arrow = self.get_end_pxl(start, theta, height_map, agent_state, sensor_state, depth_image)
            if arrow is not None:
                arrowData.append(arrow) 
        return arrowData


    def draw_arrows(self, points, rgb_image, agent_state, sensor_state, chosen_action=None, real_actions={}):
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (0, 0, 0) 
        circle_color = (255, 255, 255) 

        if chosen_action == -1:
            put_text_on_image(rgb_image, 'MODEL THINKS DONE', text_color=(0, 255, 0), text_size=4, location='center', text_thickness=3, highlight=False)
        start_px = self.agent_frame_to_image_coords([0, 0, 0], agent_state, sensor_state)
        for _, (mag, theta) in enumerate(points):
            text_size = np.clip(5/(1.2*mag+1), 0.7, 3)
            text_thickness = 2 if text_size < 1.5 else 3
            
            cart = [mag*np.sin(theta), 0, -mag*np.cos(theta)]
            end_px = self.agent_frame_to_image_coords(cart, agent_state, sensor_state)
            if end_px is None:
                continue
            if 0.05 * rgb_image.shape[1] <= end_px[0] <= 0.95 * rgb_image.shape[1] and 0.05 * rgb_image.shape[0] <= end_px[1] <= 0.95 * rgb_image.shape[0]:
                if (mag, theta) in real_actions:
                    action_name = real_actions[(mag, theta)]
                else:
                    action_name = len(real_actions) + 1
                    real_actions[(mag, theta)] = action_name

                cv2.arrowedLine(rgb_image, tuple(start_px), tuple(end_px), (255, 0, 0), 5, tipLength=0.)
                text = str(action_name) 
                (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
                circle_center = (end_px[0], end_px[1])
                circle_radius = max(text_width, text_height) // 2 + 15

                if chosen_action is not None and action_name == chosen_action:
                    cv2.circle(rgb_image, circle_center, circle_radius, (0, 255, 0), -1)
                else:
                    cv2.circle(rgb_image, circle_center, circle_radius, circle_color, -1)
                cv2.circle(rgb_image, circle_center, circle_radius, (255, 0, 0), 2)
                text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
                cv2.putText(rgb_image, text, text_position, font, text_size, text_color, text_thickness)

        if (self.step - self.turned) >= 3 or self.step == self.turned:
            text = '0'
            text_size = 3
            text_thickness = 3
            (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
            
            circle_center = (int(0.05 * rgb_image.shape[1]), int(rgb_image.shape[0] / 2))
            circle_radius = max(text_width, text_height) // 2 + 15
            if chosen_action is not None and chosen_action==0:
                cv2.circle(rgb_image, circle_center, circle_radius, (0, 255, 0), -1)
            else:
                cv2.circle(rgb_image, circle_center, circle_radius, circle_color, -1)
            cv2.circle(rgb_image, circle_center, circle_radius, (255, 0, 0), 2)
            text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
            cv2.putText(rgb_image, text, text_position, font, text_size, text_color, text_thickness)

            cv2.putText(rgb_image, 'TURN AROUND', (text_position[0]//2, text_position[1] + 80), font, text_size*0.75, (255, 0, 0), 3)

        return real_actions
    
    def select_arrrows(self, arrowData, min_angle=0.3):
        out = []
        filtered = list(filter(lambda x: x[0] > 0.65 or (x[0] > 0.3 and abs(x[1]) > 1), arrowData))
        filtered.sort(key=lambda x: x[1])
        if filtered == []:
            return []
        longest = max(filtered, key=lambda x: x[0])
        longest_theta = longest[1]
        smallest_theta = longest[1]
        longest_ndx = filtered.index(longest)
        out.append(longest)
        
        for i in range(longest_ndx+1, len(filtered)):
            if filtered[i][1] - longest_theta > min_angle:
                out.append(filtered[i])
                longest_theta = filtered[i][1]
        for i in range(longest_ndx-1, -1, -1):
            if smallest_theta - filtered[i][1] > min_angle:
                out.append(filtered[i])
                smallest_theta = filtered[i][1]

        out.sort(key=lambda x: x[1])
        return out

    def get_end_pxl(self, start, theta, height_map, agent_state, sensor_state, depth_image):
        cart = [2*np.sin(theta), 0, -2*np.cos(theta)]
        end = self.agent_frame_to_image_coords(cart, agent_state, sensor_state)
        if end is None:
            return None

        H, W = height_map.shape
    
        a = self.find_intersections(start[0], start[1], end[0], end[1], W, H)
        if a is None:
            return None
        (x1, y1), (x2, y2) = a
        # Calculate the number of points needed along the line
        num_points = max(abs(x2 - x1), abs(y2 - y1)) + 1
        
        # Generate equally spaced points along the line
        x_coords = np.linspace(x1, x2, num_points)
        y_coords = np.linspace(y1, y2, num_points)
        
        depth_values = map_coordinates(height_map, [y_coords, x_coords], order=1, mode='nearest')
        ray = [(int(x), int(y), depth) for x, y, depth in zip(x_coords, y_coords, depth_values)]
        
        out = (ray[-1][0], ray[-1][1])
        end_ndx = len(ray) - 1
        for i in range(len(ray)-4):
            st = ray[i:i+4]
            if sum(s[2] for s in st) <= 2:
                end_ndx = i
                break

        out = (ray[end_ndx][0], ray[end_ndx][1])
        out = (np.clip(out[0], 0, W-1), np.clip(out[1], 0, H-1))
               
        camera_coords = self.annotatedSimulator.unproject_2d(*out, depth_image[out[1], out[0]]) 
        local_coords = global_to_local(agent_state.position, agent_state.rotation, 
                                       local_to_global(sensor_state.position, sensor_state.rotation, camera_coords))   
        mag = np.linalg.norm([local_coords[0], local_coords[2]])
        self.update_topdown(mag, theta, agent_state)
        mag = min(0.65*mag, 3)
        return (mag, theta)

        # return (((x1, y1), out), (mag, theta))

    def find_intersections(self, x1, y1, x2, y2, img_width, img_height):
        if x2 != x1:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
        else:
            m = None  # Vertical line
            b = None
        
        intersections = []
        if m is not None and m != 0:  # Avoid division by zero for horizontal lines
            x_at_yh = int((img_height - b) / m)  # When y = img_height, x = (img_height - b) / m
            if 0 <= x_at_yh <= img_width:
                intersections.append((x_at_yh, img_height))

        if m is not None:
            y_at_x0 = int(b)  # When x = 0, y = b
            if 0 <= y_at_x0 <= img_height:
                intersections.append((0, y_at_x0))
        
        if m is not None:
            y_at_xw = int(m * img_width + b)  # When x = img_width, y = m * img_width + b
            if 0 <= y_at_xw <= img_height:
                intersections.append((img_width, y_at_xw))
        
        if m is not None and m != 0:  # Avoid division by zero for horizontal lines
            x_at_y0 = int(-b / m)  # When y = 0, x = -b / m
            if 0 <= x_at_y0 <= img_width:
                intersections.append((x_at_y0, 0))
        

        if m is None:
            intersections.append((x1, img_height))  # Bottom edge
            intersections.append((x1, 0))  # Top edge
        
        if len(intersections) == 2:
            return intersections
        return None

    def update_topdown(self, mag, theta, agent_state, clip=2):
        mag = min(0.8*mag, clip)
        local_coords = np.array([mag*np.sin(theta), 0, -mag*np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        grid_coords = self.toGrid(global_coords)
        agent_coords = self.toGrid(agent_state.position)

        cv2.line(self.new_topdown_map, agent_coords, grid_coords, (0, 255, 0), 50)
        # pass

    def toGrid(self, position):
        c = maps.to_grid(position[2], position[0], self.topdown_map.shape[0:2] , self.annotatedSimulator.sim)
        return (c[1], c[0])


    def generate_topdown(self, real_actions, agent_id=0, goal=None, zoom=12):

        agent_state = self.get_agent_state(agent_id)
        agent_coords = self.toGrid(agent_state.position)

        # if goal is not None:
        #     goal_coords = self.toGrid(goal)
        #     cv2.circle(self.topdown_map, goal_coords, radius=25, color=(255, 255, 0), thickness=-1)
            # cv2.putText(self.topdown_map, 'GOAL', (goal_coords[0] + 10, goal_coords[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  
        
        topdown_map = self.topdown_map.copy()
        self.topdown_map = self.new_topdown_map
        self.new_topdown_map = self.topdown_map.copy()
        text_size = 1.25
        text_thickness = 1
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        if self.step - self.turned >= 3:
            real_actions[(0.75, np.pi)] = 0
        for (mag, theta), action in real_actions.items():
            local_pt = np.array([mag * np.sin(theta), 0, -mag * np.cos(theta)])
            global_pt = local_to_global(agent_state.position, agent_state.rotation, local_pt)
            act_coords = self.toGrid(global_pt)

            cv2.arrowedLine(topdown_map, tuple(agent_coords), tuple(act_coords), (255, 0, 0), 5, tipLength=0.05)
            #cv2.line(self.topdown_map, agent_coords, act_coords, (0, 255, 0), 50)
            text = str(action) 
            (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
            circle_center = (act_coords[0], act_coords[1])
            circle_radius = max(text_width, text_height) // 2 + 15
            cv2.circle(topdown_map, circle_center, circle_radius, (255, 255, 255), -1)
            text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
            # put_text_on_image(self.topdown_map, '', background_color=(255, 255, 255), location='top_left', text_size=text_size, text_thickness=text_thickness+1)
            cv2.putText(topdown_map, text, text_position, font, text_size, (0, 0, 0), text_thickness+1)

        # label_text = f"Agent{agent_id}"
        # text_offset = (10, -10)  # Offset to position text near the circle
        # text_position = (agent_coords[0] + text_offset[0], agent_coords[1] + text_offset[1])
            # cv2.putText(topdown_map, label_text, text_position, font, text_size+0.5, colors[agent_id], font_thickness, cv2.LINE_AA)

        # Zoom into agent_coords
        cv2.circle(topdown_map, agent_coords, radius=15, color=(255, 0, 0), thickness=-1)
        right = (agent_state.position[0] + zoom, 0, agent_state.position[2])
        right_coords = self.toGrid(right)
        delta = abs(agent_coords[0] - right_coords[0])
        x, y = agent_coords
        # Calculate crop boundaries
        (min_x, max_x, min_y, max_y)  = self.croppings
        x1 = max(min_x, x - delta)
        x2 = min(max_x, x + delta)
        y1 = max(min_y, y - delta)
        y2 = min(max_y, y + delta)
        
        # Crop the topdown_map
        zoomed_map = topdown_map[y1:y2, x1:x2]
        
        # put_text_on_image(zoomed_map, label_text, background_color=(255, 255, 255), location='top_left', text_size=text_size+1, text_thickness=text_thickness+1)
        return zoomed_map

    # def draw_slice(self, topdown_map, agent_state):
    #     map_resolution = topdown_map.shape[0:2]
    #     agent_coords = maps.to_grid(agent_state.position[2], agent_state.position[0], map_resolution, self.annotatedSimulator.sim)
    #     agent_coords = (agent_coords[1], agent_coords[0])

    #     fov_rad = self.annotatedSimulator.fov * np.pi / 180
    #     fov_rad = fov_rad / 2

    #     local_pt = np.array([1 * np.sin(fov_rad), 0, -1 * np.cos(fov_rad)])
    #     global_pt = local_to_global(agent_state.position, agent_state.rotation, local_pt)
    #     act_coords = maps.to_grid(global_pt[2], global_pt[0], map_resolution, self.annotatedSimulator.sim)
    #     act_coords = (act_coords[1], act_coords[0]) 
    #     agent_x, agent_y = agent_coords
    #     act_x, act_y = act_coords
        
    #     # Compute differences
    #     dx = act_x - agent_x
    #     dy = act_y - agent_y
        
    #     # Calculate the angle in radians
    #     angle_rad = np.arctan2(dy, dx)
        
    #     # Convert the angle to degrees
    #     end_ang = np.degrees(angle_rad)


    #     local_pt = np.array([1 * np.sin(-fov_rad), 0, -1 * np.cos(-fov_rad)])
    #     global_pt = local_to_global(agent_state.position, agent_state.rotation, local_pt)
    #     act_coords = maps.to_grid(global_pt[2], global_pt[0], map_resolution, self.annotatedSimulator.sim)
    #     act_coords = (act_coords[1], act_coords[0]) 
    #     agent_x, agent_y = agent_coords
    #     act_x, act_y = act_coords
        
    #     # Compute differences
    #     dx = act_x - agent_x
    #     dy = act_y - agent_y
        
    #     # Calculate the angle in radians
    #     angle_rad = np.arctan2(dy, dx)
        
    #     # Convert the angle to degrees
    #     start_ang = np.degrees(angle_rad)

    #     cv2.ellipse(topdown_map, agent_coords, (200, 200), 0, end_ang, start_ang, (0, 255, 0), -1)

class NavBench(DynamicBench):

    task = 'NAV_BENCH'
    default_quat = quaternion.quaternion(0.70536607503891, 0, 0.708843231201172, 0)

    def setup_experiment(self, split, scene_ids):

        self.split = split
        self.sim_kwargs['scene_config'] =  f"datasets/hm3d/hm3d_annotated_{self.split}_basis.scene_dataset_config.json"
        json_file =  f"datasets/hm3d/hm3d_annotated_{split}_basis.scene_dataset_config.json"
        with open(json_file, 'r') as f:
            data = json.load(f)
            scenes = data['stages']['paths']['.glb']
            sids = set(int(s[2:5]) for s in scenes)
        files = [f for f in os.listdir(f'datasets/hm3d/{self.split}/') if int(f[2:5]) in sids]

        if scene_ids:
            files = [f for f in files if int(f[2:5]) in scene_ids]
        files.sort()
        random.shuffle(files)
        self.files = files        
        
    
    def setup_run(self, history=7, mask_thinking=True, add_timesteps_prompt=True, draw_arrows=True,
            points=None, consistency=1, goals = [], priv_actions=False, uniform=True, use_map=True):
        self.goal_loc = [0, 0, 0]
        # self.run_metadata['use_map'] = use_map
        while True:
            try:
                f = random.choice(self.files)
                hsh = f[6:]
                self.sim_kwargs['scene_id'] = f[2:5]
                self.sim_kwargs['scene_path'] = f'datasets/hm3d/{self.split}/00{f[2:5]}-{hsh}/{hsh}.basis.glb'
                self.annotatedSimulator = AnnotatedSimulator(**self.sim_kwargs)
                if f[2:5] == '814':
                    self.goal_loc = np.array([-1.5041651e+01, 4.8131943e-03, -2.1016624e+00])
                if f[2:5] == '891':
                    self.goal_loc = np.array([-10.326839,     0.07216382,   2.4435563 ])
                if f[2:5] == '871':
                    self.goal_loc = np.array([ 8.283967,    0.04654188, -2.368276  ])
                self.curr_target = "DIAMOND RING" #'WASHER AND DRYER'
                self.curr_related_objects = []
                while True:
                    point = self.annotatedSimulator.sim.pathfinder.get_random_navigable_point()
                    
                    # self.init_pos = point
                    # break

                    if abs(point[1] - self.goal_loc[1]) < 0.1 and np.linalg.norm(self.goal_loc-point) > 0:
                        self.init_pos = point
                        break
                break
                # random.shuffle(goals)            
                # for target, related in goals:
                #     tries = 0
                #     if os.path.exists(f'logs/{self.outer_run_name}/{target}_{self.annotatedSimulator.scene_id}'):
                #         print(f'{target}_{self.annotatedSimulator.scene_id} ALREADY EXISTS')
                #         continue
                #     self.curr_target = target
                #     self.curr_related_objects = []
                #     for word in related + [target]:
                #         self.curr_related_objects += self.annotatedSimulator.search_objects(word, exact=False)
                #     print(f'Targeting object: {target}')
                #     print(f'related objects: {len([obj.category.name() for obj in self.curr_related_objects])}')
                #     if len(self.curr_related_objects) == 0:
                #         continue
                #     for _ in range(200):
                #         point = self.annotatedSimulator.sim.pathfinder.get_random_navigable_point()
                #         for idx, floor_height in enumerate(self.annotatedSimulator.floors):
                #             tries += 1
                #             if abs(point[1] - floor_height) < 0.1:
                #                 floor = idx
                #                 distances = [np.linalg.norm(point - obj.aabb.center) for obj in self.curr_related_objects if obj.aabb.center[1] < self.annotatedSimulator.floors[floor+1] and obj.aabb.center[1] > floor_height]
                #                 min_dist = 7 if target in ['kitchen', 'living room'] else 5.5
               
                #                 if len(distances) > 0 and min(distances) > min_dist and min(distances) < min_dist + 10:
                #                     # print('found point, min_dist', min(distances), f'thresh: {min_dist}')
                #                     self.init_pos = point
                #                     break
                #         if self.init_pos is not None:
                #             break
                #     if self.init_pos is not None:
                #         break
                #     print('sampling again')
                # if self.init_pos is not None:
                #     break
                # self.init_pos = None
                # print(f'Scene id {self.annotatedSimulator.scene_id} Could not find a valid starting position')

            except Exception as e:
                print(e)
                print('\n\n\n')
                continue    

        self.run_metadata = {
            'task': self.curr_target,
            'history': history,
            'points': tuple(points) if points else 0,
            'arrows': draw_arrows,
            'consistency': consistency,
            'mask_thinking': mask_thinking,
            'add_timesteps_prompt': add_timesteps_prompt,
            'sensors': self.annotatedSimulator.sensors,
            'fov': self.annotatedSimulator.fov,
            'seed': self.random_seed,
            'scene_id': self.annotatedSimulator.scene_id,
            'init_pos': self.init_pos,
            'uniform': uniform,
            'use_map': 2
        }
        self.annotatedSimulator.priv_actions = False
        self.annotatedSimulator.do_annotate_image = False
        self.annotatedSimulator.objects_to_annotate = self.curr_related_objects
        self.set_state()
        # if random.random() < 0.5:
        #     self.curr_target = "Large painting of an owl, on the wall behind a table"
        # else:
        #     self.curr_target = "staircase"

        self.curr_run_name = f'{self.curr_target}_{self.annotatedSimulator.scene_id}_{random.randint(0, 1000)}'
        obs = self.annotatedSimulator.step([('forward', 0)])
        return obs

    def step_env(self, obs):
        agent_state = self.get_agent_state()
        points = []
        rnge = 1.5 if len(self.annotatedSimulator.sensors) == 1 else 2.2
        spacing = 0.35 if len(self.annotatedSimulator.sensors) == 1 else 0.29
        

        for sensor in self.annotatedSimulator.sensors:
            points += self.get_arrow_options(obs[f'depth_sensor_{sensor}'], agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], rnge)
        points = self.select_arrrows(points, spacing)
        real_actions = {}    
        for sensor in self.annotatedSimulator.sensors:
            real_actions = self.draw_arrows(points, obs[f'color_sensor_{sensor}']['image'], agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], real_actions=real_actions)
        
        zoomed_map = self.generate_topdown(real_actions, zoom=10)

        multi = len(self.annotatedSimulator.sensors) > 1
        prompt = (
        # f"First, analyze your updated camera observation and tell me the spatial layout of what you see. "
        f"I have lost my diamond ring! Your task is to search every inch of this floor to help me find it. It is a huge, bright diamond that is unmistakable. "
        f"There are {len(real_actions)} red arrows superimposed onto your observation{'s' if multi else''}, which represent potential actions. " 
        f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'Note that special action 0 turns you around completely' if self.step - self.turned >= 3 else ''}"
        #f"Your task is to navigate to the {self.curr_target.upper()}. Think of a high level plan on how you can reach the {self.curr_target.upper()} from where you are now. If you have already reached the {self.curr_target.upper()} choose special action -1 (done). "
        f"First, tell me what you see in your sensor observations. Then, tell me a high level plan on how you will find the ring and where you will go next"
        f"Think about how each action will move you. Lastly, select one action from the image and explain how it helps you reach your goal. Return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS"
        )

        images = self.get_sensor_images(obs, convert=False)
        if self.run_metadata['use_map']:
            prompt = (

            # f"Your task is to navigate to a {self.curr_target.upper()}, and you have {len(real_actions)} actions available to you. "
            # "You have two different sources of information that show these actions: \n1. your RGB sensors show your current view of the environment, and the actions are superimposed onto the images as red arrows to white circles. The white cicles show the exactly where the action will move you. "
            # "\n2. you have a topdown map of the environment, with navigable area shown in light grey and obstacles shown in black. This map shows the trajectory of where you have been in the past, shown GREEN. Your current location is shown by a RED dot. "
            # "The same actions you see superimposed on the RGB image are also shown on the top-down map. These actions also represented by red arrows and white circles, and show the location you would move to if you took that action. "
            # f"Carefully analyze this map and make sure you understand what it means. {'Remember you have action 0 for when you are in a dead end or want to turn around' if self.step - self.turned >= 3 else ''}"
            # f"\n\nFirst, tell me what you see in your sensor observations, and which way you should go to reach your goal. Second, describe the map and think about which actions leave your curent room. Combine both sources of information to make an informed decision on what action to take. "
            # f"If you have already comleted your goal choose special action -1. Return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS "
            # # Think of a high level plan on how you can reach a {self.curr_target.upper()} from where you are now. ")
            # )
            f"I have lost my diamond ring! Your task is to search every inch of this floor to help me find it. It is a huge, bright diamond that is unmistakable. "

            # f"Your task is to navigate to the {self.curr_target.upper()}, and "you have {len(real_actions)} actions available to you. "
            f"There are {len(real_actions)} red arrows superimposed onto your observation{'s' if multi else''}, which show where you would move to if you chose that action number ." 
            "\nTo help you with exploration, you have a topdown map of the environment, with navigable area shown in LIGHT GREY and obstacles shown in BLACK. This map shows where you have been in the past, in GREEN. Your current location is shown by a RED dot. "
            f"There are {len(real_actions)} actions, which are red arrows and white circles, and show the location you would move to if you took that action. Use the map to identify unexplored rooms (light grey) and strategically plan out actions that help you reach these unexplored areas. Note you will sometimes need to backtrack through green areas to reach new rooms. "
            # "If you have already reached the {self.curr_target.upper()} choose special action -1 (done). "
            "First, describe the map you see, as well as your sensor observations. Then, use your map and your observations to strategically think and tell me which actions will help cover new territory the most, and remember the light grey areas are navigable and the black areas are not "    
            f"Return your action as {{'action': <action_number>}}"
            # Think of a high level plan on how you can reach a {self.curr_target.upper()} from where you are now. ")
            )
            # self.vlm = self.map_vlm
            images.append(zoomed_map)
            # images = [zoomed_map]
        
        row = {'actions': -10, 'tokens_generated':0, 'success': 1, 'metadata': self.run_metadata,
        'speed': 0, 'scene_id': self.annotatedSimulator.scene_id,
        'model': self.vlm.name, 'input_tokens': 0, 'agent_location': agent_state.position}
        row, metadata, resp = self.agent_self_consitency(prompt, images, row, self.run_metadata['consistency'])

        row['goal_object'] = self.curr_target

        rgb = self.topdown_map
        green = np.sum(rgb == (0, 255, 0))
        light_grey = np.sum(rgb == (200, 200, 200))
        row['explored'] = green / (green + light_grey) 
        min_dist = 1000
        closest_object = None
        # if not self.run_metadata['use_map']: 

        #     images.append(zoomed_map)
        if row["actions"] <= len(list(real_actions.keys())) and row["actions"] > 0:
            mag, theta = list(real_actions.keys())[row["actions"]-1]
            self.update_topdown(mag, theta, agent_state, mag)
        images = self.get_sensor_images(obs, convert=False) + [zoomed_map]

        copies = []
        for sensor in self.annotatedSimulator.sensors:
            annotations = obs[f'color_sensor_{sensor}']['annotations']
            for obj in annotations:
                dist = np.linalg.norm(obj['curr_local_coords'])
                print('object', obj['obj'], 'distance', dist)
                if dist < min_dist:
                    min_dist = dist
                    closest_object = obj['obj']

            copy = obs[f'color_sensor_{sensor}']['image'].copy()
            self.draw_arrows(real_actions, copy, agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], chosen_action=row['actions'], real_actions=real_actions)
            copies.append(copy)
        copies.append(self.topdown_map)
        row['closest_object'] = closest_object
        # row['distance_to_goal'] = min_dist
        row['distance_to_goal'] = np.linalg.norm(agent_state.position - self.goal_loc)
        metadata['DIST TO GOAL'] = row['distance_to_goal']

        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        if self.run_metadata['mask_thinking'] and row['success'] == 1 and self.run_metadata['history'] > 0:
            self.vlm.session.history[-1].parts[0].text = f'\n' + '{"action": ' + str(row["actions"]) + '}'

        if self.step % self.log_freq == 0 or row['success'] == 0:
            images = [Image.fromarray(im[:, :, 0:3], mode='RGB') for im in images]
            copies = [Image.fromarray(im[:, :, 0:3], mode='RGB') for im in copies]
            self.log(images, resp, row['success'], metadata, copy_images=copies)

        if self.step >= 3 and self.df['actions'].iloc[-3:].tolist().count(-1) >= 2:
            print("STOPPING EARLY, DONE")
            return None

        return self.annotatedSimulator.move_choices(row['actions'], points=list(real_actions.keys()))        


























class GOATBench(DynamicBench):

    task = 'GOAT_BENCH'

    def setup_experiment(self, split, num_scenes):
        self.goat_data = []
        self.split = split
        self.sim_kwargs['scene_config'] =  f"datasets/hm3d/hm3d_annotated_{self.split}_basis.scene_dataset_config.json"
        self.sim_kwargs['goal_image_agent'] = True
        if self.split == 'train':
            dir = 'train'
        else:
            dir = 'val_unseen'
        
        for f in os.listdir(f'datasets/goatBench/{dir}/content')[0:num_scenes]:
            with gzip.open(f'datasets/goatBench/{dir}/content/{f}', 'rt') as gz:
                self.goat_data.append(json.load(gz))
        
        random.shuffle(self.goat_data)


    def setup_run(self, history=7, mask_thinking=True, add_timesteps_prompt=True, draw_arrows=True,
            points=None, consistency=1, max_steps_per_goal=5, priv_actions=False, use_map=False, uniform=False):
        while True:
            goat_scene = random.choice(self.goat_data)
            episode = random.choice(goat_scene['episodes'])
            f, glb = episode['scene_id'].split('/')[-2:]

            if os.path.exists(f'logs/{self.outer_run_name}/{episode["episode_id"]}_{f[2:5]}'):
                continue
            break

        self.sim_kwargs['scene_id'] = f[2:5]
        self.sim_kwargs['scene_path'] = f'datasets/hm3d/{self.split}/{f}/{glb}'
        self.annotatedSimulator = AnnotatedSimulator(**self.sim_kwargs)
        self.annotatedSimulator.priv_actions = priv_actions
        self.annotatedSimulator.do_draw_arrows = points if draw_arrows else None
        self.annotatedSimulator.do_annotate_image = False

        self.curr_episode = []
        # all_objects = {obj.id : obj for obj in self.annotatedSimulator.get_all_objects(filter = False)}
        # self.habitat_objects = []
        self.init_pos = np.array(episode['start_position'])
        # init_rot = quaternion.quaternion(*episode['start_rotation'][0:3], episode['start_rotation'][3])
        self.set_state(pos=self.init_pos, quat=episode['start_rotation'])
        for goal in episode['tasks']:
            name = goal[0]
            mode = goal[1]
            
            target = {'name': name, 'mode': mode, 'id': goal[2], 'objects': []}
            
            descriptions = goat_scene['goals'][f'{f[6:]}.basis.glb_{name}']

            for d in descriptions:

                if mode == 'object':
                    target['objects'].append({'object_id': d['object_id'], 'position': d['position']})
                else:
                    if d['object_id'] == goal[2]:
                        if mode == 'description':
                            target['lang_desc'] = d['lang_desc']
                            target['position'] = d['position']
                        if mode == 'image':
                            ndx = goal[3]
                            target['position'] = d['position']
                            target['image_position'] = d['image_goals'][ndx]['position']
                            target['image_rotation'] = d['image_goals'][ndx]['rotation']

            # self.habitat_objects.append(habitat_obj)
            self.curr_episode.append(target)
        print(f'Running episode with {len(self.curr_episode)} goals')
        # self.annotatedSimulator.objects_to_annotate = self.habitat_objects                   
        self.curr_goal_ndx = 0
        self.curr_run_name = f"{episode['episode_id']}_{self.annotatedSimulator.scene_id}"
        self.last_goal_reset = -1
        goal = self.curr_episode[self.curr_goal_ndx]

        if goal['mode'] == 'object':
            print('Current general object:', goal['name'], f'there are {len(goal["objects"])} instances')
        if goal['mode'] == 'description':
            print('Current desc:', goal['lang_desc'])
        if goal['mode'] == 'image':
            print('Current image:', goal['name'])
        self.run_metadata = {
            'task': self.curr_run_name,
            'history': history,
            'points': tuple(points) if points else 0,
            'arrows': draw_arrows,
            'consistency': consistency,
            'mask_thinking': mask_thinking,
            'add_timesteps_prompt': add_timesteps_prompt,
            'sensors': self.annotatedSimulator.sensors,
            'fov': self.annotatedSimulator.fov,
            'seed': self.random_seed,
            'scene_id': self.annotatedSimulator.scene_id,
            'init_pos': self.init_pos,
            'max_steps_per_goal': max_steps_per_goal,
            'use_map': use_map,
            'uniform': uniform

        }

        obs = self.annotatedSimulator.step([('forward', 0)])
        return obs

    def step_env(self, obs):
        goal = self.curr_episode[self.curr_goal_ndx]

        if goal['mode'] == 'object':
            inst = f'Find the nearest {goal["name"]} and navigate to it. '
            inst2 =  f'Tell me which room you would find this {goal["name"]} in? Do you see a {goal["name"]} in your current observations?' 
        if goal['mode'] == 'description':
            inst = f"Find and navigate to the {goal['lang_desc']} "
            inst2 =  f"Tell me which room you would find this {goal['name']} in, and in which direction you should go. "
        if goal['mode'] == 'image':
            inst = f"Observe the image labeled GOAL IMAGE. Find this specific {goal['name']} shown in the image and navigate to it. "
            inst2 =  f"Tell me which room you would find this {goal['name']} in, and which in which direction you should go ."

        agent_state = self.get_agent_state()
        points = []
        rnge = 1.5 if len(self.annotatedSimulator.sensors) == 1 else 2.2
        spacing = 0.35 if len(self.annotatedSimulator.sensors) == 1 else 0.29
        

        for sensor in self.annotatedSimulator.sensors:
            points += self.get_arrow_options(obs[f'depth_sensor_{sensor}'], agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], rnge)
        points = self.select_arrrows(points, spacing)
        real_actions = {}    
        for sensor in self.annotatedSimulator.sensors:
            real_actions = self.draw_arrows(points, obs[f'color_sensor_{sensor}']['image'], agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], real_actions=real_actions)
        
        zoomed_map = self.generate_topdown(real_actions, zoom=10)

        multi = len(self.annotatedSimulator.sensors) > 1
        prompt = (

        f"TASK: {inst} "
        # f"First, analyze your updated camera observation and tell me the spatial layout of what you see. "
        f"There are {len(real_actions)} red arrows superimposed onto your observation{'s' if multi else''}, which represent potential actions. " 
        f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'Note that special action 0 turns you around completely' if self.step - self.turned >= 3 else ''}"
        #f"Your task is to navigate to the {self.curr_target.upper()}. Think of a high level plan on how you can reach the {self.curr_target.upper()} from where you are now. If you have already reached the {self.curr_target.upper()} choose special action -1 (done). "
        f"First, tell me what you see in your sensor observations, and if you have any leads on finding the {goal['name']}. Second, {inst2}. If you have already reached the {goal['name']} choose special action -1 (done). "
        f"Think about how each action will hlp you achieve this goal. Lastly, select one action from the image and explain how it helps you reach your goal. Return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS"
        )

        images = self.get_sensor_images(obs, convert=False)
        if self.run_metadata['use_map']:
            prompt = (
            f"TASK: {inst}. First, analyze your updated camera observation and tell me the spatial layout of what you see. "
            # f"Your task is to navigate to the {self.curr_target.upper()}, and "you have {len(real_actions)} actions available to you. "
            f"There are {len(real_actions)} red arrows superimposed onto your observation{'s' if multi else''}, which represent potential actions. " 
            f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'Note that special action 0 turns you around completely' if self.step - self.turned >= 3 else ''}"
            "\nTo help you with exploration, you have a topdown map of the environment, with navigable area shown in LIGHT GREY and obstacles shown in BLACK. This map shows where you have been in the past, in GREEN. Your current location is shown by a RED dot. "
            "The actions are red arrows and white circles, and show the location you would move to if you took that action. Use the map to identify unexplored rooms (light grey) and strategically plan out actions that help you reach these unexplored areas. Note you will sometimes need to backtrack through green areas to reach new rooms. "
            # "If you have already reached the {self.curr_target.upper()} choose special action -1 (done). "
            "First, describe the map you see, as well as your sensor observations. Then, use your both your sensor observations and your map to strategically think and tell me which actions will help you achieve your goal the most, and remember the light grey areas are navigable and the black areas are not "    
            f"Return your action as {{'action': <action_number>}}"
            # Think of a high level plan on how you can reach a {self.curr_target.upper()} from where you are now. ")
            )
            # self.vlm = self.map_vlm
            images.append(zoomed_map)        # f"First, analyze your updated camera observation and tell me the spatial layout of what you see. "
# """
# {{
# 0: turn completely around, use this when you dont see any good arrows, or IF THERE ARE NO ARROWS labeled on the image. 
# -1: DONE, have already navigated to the the {goal['name']}!!{'. Make sure it matches the one in the description' if goal['mode'] in ['description', 'image'] else ''}
# }}
# Tell me the following: how you plan to reach {goal['name']} from where you are now? Do you need to move into a new room? Lastly, select one action from the image or the special actions and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS. 
# """
# Tell me the following: how you plan to reach {goal['name']} from where you are now? Do you need to move into a new room? Lastly, select one action from the image or the special actions and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS. 

        images = self.get_sensor_images(obs, convert=False) + [zoomed_map]


        if goal['mode'] == 'image':
            position = goal['image_position']
            rotation = goal['image_rotation']
            goal_im = self.annotatedSimulator.get_goal_image(position, rotation)
            put_text_on_image(goal_im, f"GOAL IMAGE: {goal['name']}", background_color=(255, 255, 255), location='top_center')
            images.append(goal_im)

        row = {'actions': -10, 'tokens_generated':0, 'success': 1, 'metadata': self.run_metadata,
        'speed': 0, 'scene_id': self.annotatedSimulator.scene_id, 'goal': goal['name'], 'goal_mode': goal['mode'], 'goal_index': self.curr_goal_ndx, 'curr_goal_steps': self.step - self.last_goal_reset,
        'model': self.vlm.name, 'input_tokens': 0, 'agent_location': agent_state.position}
        row, metadata, resp = self.agent_self_consitency(prompt, images, row, self.run_metadata['consistency'])

        if goal['mode'] == 'object':
            distances = [np.linalg.norm(agent_state.position - obj['position']) for obj in goal['objects']]
            min_dist = min(distances)
            row['distance_to_goal'] = min_dist
            metadata['DIST TO GOAL'] = row['distance_to_goal']
        if goal['mode'] in ['description', 'image']:
            row['distance_to_goal'] = np.linalg.norm(agent_state.position - goal['position'])
            metadata['DIST TO GOAL'] = row['distance_to_goal']
        print('distance to goal', row['distance_to_goal'])

        metadata['INST'] = inst
        done = False
        new_goal = False
        goal_reached = False
        if row['distance_to_goal'] < 2.5 and row['actions'] == -1:
            print(f"SUCESSFULLY FINISHED GOAL {self.curr_goal_ndx}")
            new_goal = True
            goal_reached = True
        elif self.step + 1 - self.last_goal_reset > self.run_metadata['max_steps_per_goal']:
            print('MAX STEPS PER GOAL REACHED')
            new_goal = True
            goal_reached = False
        elif row['distance_to_goal'] < 2.5:
            print('NEAR GOAL BUT MODEL DID NOT RETURN DONE')
        elif row['actions'] == -1:
            print('MODEL RETURNED DONE BUT NOT NEAR GOAL')
            
        copies = []
        for sensor in self.annotatedSimulator.sensors:
            copy = obs[f'color_sensor_{sensor}']['image'].copy()
            self.draw_arrows(real_actions, copy, agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], chosen_action=row['actions'], real_actions=real_actions)
            # Assuming `copy` is the image you want to modify
            if new_goal and goal_reached:
                background_color = (0, 100, 0)  # Green color
            elif new_goal:
                background_color = (100, 0, 0) # Red color
            else:
                background_color = (255, 255, 255)  # White color
            put_text_on_image(copy, f"{self.curr_goal_ndx}: {goal['name']}({goal['mode'][0]})", background_color=background_color, location='top_left', text_size=2.3)
            copies.append(copy)

        copies.append(self.topdown_map)


        if self.run_metadata['mask_thinking'] and row['success'] == 1 and self.run_metadata['history'] > 0:
            self.vlm.session.history[-1].parts[0].text = f'\n' + '{"action": ' + str(row["actions"]) + '}'
        row['goal_reached'] = goal_reached
        row['new_goal'] = new_goal
        if new_goal:
            self.curr_goal_ndx += 1
            self.last_goal_reset = self.step
            if self.curr_goal_ndx >= len(self.curr_episode):
                done = True
                print("FINISHING TRAJECTORY, NO MORE GOALS")
            else:
                print(f"Moving onto")
                goal = self.curr_episode[self.curr_goal_ndx]

                if goal['mode'] == 'object':
                    print('New general object:', goal['name'])
                if goal['mode'] == 'description':
                    print('New specific:', goal['lang_desc'])
                if goal['mode'] == 'image':
                    print('New image:', goal['name'])

        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        if self.step % self.log_freq == 0 or row['success'] == 0:
            images = [Image.fromarray(im[:, :, 0:3], mode='RGB') for im in images]
            copies = [Image.fromarray(im[:, :, 0:3], mode='RGB') for im in copies]
            self.log(images, resp, row['success'], metadata, copy_images=copies)
            
        if done:
            return None
        self.vlm.get_spend()
        return self.annotatedSimulator.move_choices(row['actions'], points=list(real_actions.keys()))   
    






















class EQABench(DynamicBench):


    task = 'EQA_BENCH'

    def setup_experiment(self, split, scene_ids):
        if scene_ids is None:
            scene_ids = range(515)
        file1 = 'datasets/EQA/questions.csv'
        file2 = 'datasets/EQA/scene_init_poses.csv'
        self.answerVLM = GeminiModel(sys_instruction='You made to assist an agent in an environemnt. The agent sends you images and queries, and you intelligently respond with an answer ', 
                                  model='gemini-1.5-pro')
        # self.answerVLM = GPTModel(sys_instruction='You are a helpful assistant that answers questions about the observations you see', 
        #                           model='gpt-4o')
        self.quids = scene_ids
        with open(file1) as f:

            self.questions_data = [
                {"qid": idx, **{k: v for k, v in row.items()}}
                for idx, row in enumerate(csv.DictReader(f, skipinitialspace=True)) if idx in scene_ids
            ]
        with open(file2) as f:
            self.init_pose_data = {}
            for row in csv.DictReader(f, skipinitialspace=True):
                self.init_pose_data[row["scene_floor"]] = {
                    "init_pts": [
                        float(row["init_x"]),
                        float(row["init_y"]),
                        float(row["init_z"]),
                    ],
                    "init_angle": float(row["init_angle"]),
                }
        random.shuffle(self.questions_data)
        self.q_index = -1

    def setup_run(self, history=7, mask_thinking=True, add_timesteps_prompt=True, draw_arrows=True,
            points=None, consistency=1, max_steps_per_goal=5, uniform=False, use_map=True):
            self.interesting_images = {'A': [], 'B': [], 'C': [], 'D': []}
            self.q_index += 1
            # self.q_index = self.q_index % len(self.questions_data)
            question_data = self.questions_data[self.q_index % len(self.questions_data)]
            self.quid = question_data["qid"]
            scene = question_data["scene"]
            floor = question_data["floor"]
            scene_floor = scene + "_" + floor
            init_pts = self.init_pose_data[scene_floor]["init_pts"]
            init_angle = self.init_pose_data[scene_floor]["init_angle"]

            scene_dir = 'datasets/hm3d/train'
            scene_mesh_dir = os.path.join(
                    scene_dir, scene, scene[6:] + ".basis" + ".glb"
                )
            self.sim_kwargs['scene_path'] = scene_mesh_dir
            self.sim_kwargs['scene_config'] = f"{scene_dir}/hm3d_train_basis.scene_dataset_config.json"
            self.sim_kwargs['scene_id'] = scene[2:5]
            rotation = quat_to_coeffs(
                quat_from_angle_axis(init_angle, np.array([0, 1, 0]))
            ).tolist()
            self.answer_counter = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
            self.annotatedSimulator = AnnotatedSimulator(**self.sim_kwargs)
            scene_bnds = self.annotatedSimulator.sim.pathfinder.get_bounds()
            scene_lower_bnds_normal = np.dot(scene_bnds[0], np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))
            scene_upper_bnds_normal = np.dot(scene_bnds[1], np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))
            size = np.sum((scene_upper_bnds_normal[:2] - scene_lower_bnds_normal[:2]) ** 2)
            num_step = int(math.sqrt(size) * 3)
            self.curr_run_steps = num_step
            print("\nQUESTION:", question_data["question"])
            print(f'BUDGET STEPS FOR THIS RUN: {num_step}')

            self.run_metadata = {
            'max_steps': num_step,
            'q_index': self.q_index,
            'quid': self.quid,
            'history': history,
            'points': tuple(points) if points else 0,
            'arrows': draw_arrows,
            'consistency': consistency,
            'mask_thinking': mask_thinking,
            'add_timesteps_prompt': add_timesteps_prompt,
            'sensors': self.annotatedSimulator.sensors,
            'fov': self.annotatedSimulator.fov,
            'seed': self.random_seed,
            'scene_id': self.annotatedSimulator.scene_id,
            'init_pos': self.init_pos,
            'uniform': uniform,
            'use_map': use_map,
            'question_type': question_data["label"]
            }  

            self.annotatedSimulator.priv_actions = False
            self.annotatedSimulator.do_draw_arrows = False #points if draw_arrows else None
            self.annotatedSimulator.do_annotate_image = False
            self.annotatedSimulator.objects_to_annotate = []
            
            self.set_state(init_pts, rotation)

            self.curr_run_name = f'{self.q_index}_{self.quid}_{self.annotatedSimulator.scene_id}'
            obs = self.annotatedSimulator.step([('forward', 0)])
            return obs


    def step_env(self, obs):
        if self.step >= self.curr_run_steps:
            print("MAX STEPS REACHED")
            return None
        agent_state = self.get_agent_state()
        question_data = self.questions_data[self.q_index % len(self.questions_data)]

        question = question_data["question"]
        choices = [c.split("'")[1] for c in question_data["choices"].split("',")]
        # choices.append('I need to explore the environment further')
        answer = question_data["answer"]
        vlm_question = question
        vlm_pred_candidates = ["A", "B", "C", "D"]
        for token, choice in zip(vlm_pred_candidates, choices):
            vlm_question += "\n" + token + "." + " " + choice
        multi = len(self.run_metadata['sensors']) > 1

        self.vlm_question = vlm_question

        raw_images = [obs[f'color_sensor_{i}']['image'].copy() for i in self.annotatedSimulator.sensors]

        
        def answer_thread():
            es = ["A", "B", "C", "D", "E"]
            extra = "\n" + es[len(choices)] + "." + " " + "I need the agent to move to a different location to answer this question"
            pred = 'E'
            answer_prompt = (f"The agent has sent you an image, and is asking you the following question: [QUESTION]: {vlm_question+extra}\nFirst, describe what you see in the image, and it there any notable objects that are relevant to the question. Pay attention to the details in the question. Then, explain what the best answer choice is and why. Lastly, return it as a JSON like {{'answer': <answer letter>}}")
            res = None
            for image in raw_images:
                r, p = self.answerVLM.call([image], answer_prompt, logprobs=5)
                dct = self.parse_response(r)
                if 'answer' in dct and dct['answer'] in ['A', 'B', 'C', 'D']:
                    print('ANSWRE MODEL:', r)
                    self.interesting_images[dct['answer']].append(image)
                    self.answer_counter[dct['answer']] += 1.01
                    pred = dct['answer']
                    res = r
            if res is None:
                res = r
            answer_mdata = {'ANSWER PROMPT': answer_prompt, 'ANSWER RESPONSE': res}
            # if type(self.answerVLM) == GeminiModel:
            # else:
            #     for i in p['logprobs']:
            #         prob = np.exp(i['logprob'])
            #         choice = i['token']
            #         if choice in ['A', 'B', 'C', 'D'] and prob > 0.2:
            #             self.answer_counter[choice] += prob
            #     max_token = max(p['logprobs'], key=lambda x: x['logprob'])['token']
            #     if max_token in ['A', 'B', 'C', 'D']:
            #         pred = max_token
            #     else:
            #         pred = max_token
            #     answer_mdata = {'ANSWER PROMPT': answer_prompt, 'ANSWER RESPONSE': max_token, 'ANSWER LOGPROBS': p['logprobs']}

            return pred, answer_mdata

  

        def action_thread():
            row = {'actions': -10, 'tokens_generated':0, 'success': 1, 'metadata': self.run_metadata, 'cum_answer': None,
            'speed': 0, 'scene_id': self.annotatedSimulator.scene_id, 'question': question, 'choices': choices, 'answer': None, 'ground_truth': answer,
            'model': self.vlm.name, 'input_tokens': 0, 'agent_location': agent_state.position, 'actions': -10, 'prediction': None}
            
            points = []
            rnge = 1.5 if len(self.annotatedSimulator.sensors) == 1 else 2.2
            spacing = 0.34 if len(self.annotatedSimulator.sensors) == 1 else 0.29
            for sensor in self.annotatedSimulator.sensors:
                points += self.get_arrow_options(obs[f'depth_sensor_{sensor}'], agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], rnge)
            points = self.select_arrrows(points, spacing)
            real_actions = {}

            for sensor in self.annotatedSimulator.sensors:
                real_actions = self.draw_arrows(points, obs[f'color_sensor_{sensor}']['image'], agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], real_actions=real_actions)
            
            images = self.get_sensor_images(obs, convert=False)
            zoomed_map = self.generate_topdown(real_actions)

            prompt_question = (
                "Your task is to navigate through the environment and learn the answer to the following quesiton\n"
                f"[QUESTION]: {vlm_question}\n"
                f"There are {len(real_actions)} red arrows superimposed onto your observation{'s' if multi else''}, which represent potential actions. " 
                f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'Note that action 0 turns you around completely .' if self.step - self.turned >= 3 else ''}"
                "First, tell me what you see from your each of your current sensor observations, and if there are any notable objects that are relevant to the question. "
                "Second, tell me a room or location you should navigate to in order to answer the question. "
                "Lastly, choose the an action that will get you to that room or location, and return it in the format {'action': <action_number>}. Dont answer the question, just return an action"
                
                # "Note you CANNOT GO THROUGH CLOSED DOORS."
            )

            if self.run_metadata['use_map']:
                prompt_question = (
                "Your task is to navigate throughout the environment and learn the answer to the following quesiton\n"
                f"[QUESTION]: {vlm_question}\n"
                f"There are {len(real_actions)} red arrows superimposed onto your observation{'s' if multi else''}, which represent potential actions. " 
                f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'Note that action 0 turns you around completely .' if self.step - self.turned >= 3 else ''}"
    
                "\nYou have a topdown map of the environment, with navigable area shown in LIGHT GREY and obstacles shown in BLACK. This map shows you where you have been in the past, shown in GREEN. Your current location is shown by a RED dot. "
                "The same actions you see superimposed on the RGB image are also shown on the top-down map. These actions also represented by red arrows and white circles, and show the location you would move to if you took that action. "
                "Use this map to help you explore new areas (light grey). "
                
                "First, tell me what you see from your current sensor observations. "
                "Second, tell me what room or location you should navigate to in order to answer the question, and which direction you should go to reach that. "
                "Third. cross reference these actions with the top-down map to make sure they are are good choices."
                "Lastly, return an action in the format {'action': <action_number>}. Don't answer the question, just return an action"
                # "Note you CANNOT GO THROUGH CLOSED DOORS."
                )

                images.append(zoomed_map)

            return *self.agent_self_consitency(prompt_question, images, row, self.run_metadata['consistency']), zoomed_map, real_actions
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future1 = executor.submit(answer_thread)
            future2 = executor.submit(action_thread)
            
            row, metadata, resp, zoomed_map, real_actions = future2.result()
            pred, answer_mdata = future1.result()
        
        row['answer'] = pred    
        row['cum_answer'] = max(self.answer_counter, key=self.answer_counter.get)

        images = self.get_sensor_images(obs) + [zoomed_map]
        print(f'action {row["actions"]}, pred {row["answer"]}, ground {answer}')
        if row["actions"] <= len(list(real_actions.keys())) and row["actions"] > 0:
            mag, theta = list(real_actions.keys())[row["actions"]-1]
            self.update_topdown(mag, theta, agent_state, mag)
        metadata['PREDICTION'] = row['answer']
        metadata['GROUND TRUTH'] = answer


        metadata.update(answer_mdata)
        # self.answer_counter.update([row['answer']])
        copies = []
        for i, sensor in enumerate(self.annotatedSimulator.sensors):
            copy = images[i].copy()

            self.draw_arrows(real_actions, copy, agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], chosen_action=row['actions'], real_actions=real_actions)
            put_text_on_image(images[i], f"QUESTION: {question}", background_color=(255, 255, 255), location='top_left', text_size=1.5, text_thickness=2)
            put_text_on_image(copy, f"QUESTION: {question}", background_color=(255, 255, 255), location='top_left', text_size=1.5, text_thickness=2)
            if row['answer'] and not row['answer'] == 'E':
                ans = choices[vlm_pred_candidates.index(row['answer'])]
                color = (0, 255, 0) if row['answer'] == answer else (255, 0, 0)
                put_text_on_image(copy, f"{ans}", background_color=color, location='top_right', text_size=2, text_thickness=2)
                put_text_on_image(images[i], f"QUESTION: {question}", background_color=(255, 255, 255), location='top_left', text_size=1.5, text_thickness=2)

            copies.append(copy)
        copies.append(self.topdown_map)

        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        if self.run_metadata['mask_thinking'] and row['success'] == 1 and self.run_metadata['history'] > 0:
            self.vlm.session.history[-1].parts[0].text = f'\n' + '{"action": ' + str(row["actions"]) + '}'

        if self.step % self.log_freq == 0 or row['success'] == 0:
            images = [Image.fromarray(im[:, :, 0:3], mode='RGB') for im in images]
            copies = [Image.fromarray(im[:, :, 0:3], mode='RGB') for im in copies]
            self.log(images, resp, row['success'], metadata, copy_images=copies)

        # Sort self.answer_counter by values in descending order
        sorted_counter = list(sorted(self.answer_counter.items(), key=lambda x: x[1], reverse=True))
        
        # Get the keys and values of the top two entries
        print('answers counter', sorted_counter)
        if(sorted_counter[0][1] > 3 and sorted_counter[0][1]/(sorted_counter[1][1]+0.001) > 2.5 and self.step > 7) or (sorted_counter[0][1] > 5 and self.step > 7):
            print("STOPPING EARLY, DONE")
            return None

        self.answerVLM.get_spend()
        self.vlm.get_spend()
        return self.annotatedSimulator.move_choices(row['actions'], points=list(real_actions.keys()))        

    def post_run(self):
        self.final_answer()
        return super().post_run()

    def final_answer(self):

        self.run_metadata['final_answer'] = random.choice(['A', 'B', 'C', 'D'])
        images = []
        for k, v in self.interesting_images.items():
            if len(v) > 1:
                images += random.choices(v, k=2)
            elif len(v) == 1:
                images.append(random.choice(v))
        answer_prompt = (f"Your task is to analyze {len(images)} images from the same environment, and then answer a question about the environment [QUESTION]: {self.vlm_question}\n First, tell me what you see in each image, and if there any notable objects that are relevant to the question. Then, do your best to answer the quesiton, even if you are unsure, and return it as a JSON like {{'answer': <answer letter>}}")
        if len(images) > 0:
            r, p = self.answerVLM.call(images, answer_prompt)
            print('FINAL ANSWER MODEL:', r)
            dct = self.parse_response(r)
            if 'answer' in dct and dct['answer'] in ['A', 'B', 'C', 'D']:
                self.run_metadata['final_answer'] = dct['answer']
