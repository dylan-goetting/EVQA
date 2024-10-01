import logging
import math
import os
import random
from re import I
from sqlite3 import DatabaseError
import pdb
from typing import Optional, Sequence, Union
from habitat.utils.visualizations import maps
import numpy as np
import cv2
import ast
import pandas as pd
from PIL import Image
import requests
from torch import Value, clip_, real
from src.annoatedSimulator import AnnotatedSimulator
from src.agent import VLMNav
from src.utils import *
from src.vlm import VLM, GPTModel, GeminiModel
import habitat_sim
import cv2
from src.pivot import PIVOT
import traceback
import wandb

class DynamicBench: 

    task = 'Not defined'

    def __init__(self, sim_kwargs=None, agent=None, exp_kwargs=None, outer_run_name=None, catch=False, log_file=None, port=5000):
        

        self.missed = []
        self.sim_kwargs = sim_kwargs
        self.path = habitat_sim.MultiGoalShortestPath()

        self.agent: VLMNav = agent
        self.exp_kwargs = exp_kwargs
        self.port = port
        self.df = pd.DataFrame({})
        self.random_seed = sim_kwargs['random_seed']
        self.outer_run_name = self.task + '_' + outer_run_name
        self.log_file = log_file
        self.inner_run_name = f'{exp_kwargs["part"]}_of_{exp_kwargs["parts"]}'
        self.parts=exp_kwargs['parts']
        self.part=exp_kwargs['part']
        self.curr_run_name = "Not started"
        self.annotatedSimulator: AnnotatedSimulator = None
        self.exception_type = DatabaseError
        self.data_len = 0
        if catch:
            self.exception_type = Exception

        self.setup_experiment(**self.exp_kwargs)

    def setup_experiment(self, **exp_kwargs):
        raise NotImplementedError

    def run_experiment(self, outer_loop, inner_loop, log_freq, **run_kwargs):
        
        for i in range(outer_loop):
            if self.data_len > 0:
                part_size = math.ceil(self.data_len/self.parts)
                start_ndx = self.part * part_size
                if start_ndx + i >= self.data_len:
                    return
                run_kwargs['data_ndx'] = start_ndx + i
                self.log_data = {'data_ndx': run_kwargs['data_ndx'], 'instance': self.inner_run_name, 'total_episodes': self.parts*outer_loop, 'task': self.task, 'task_data': {}}

            try:
                self.run_trajectory(inner_loop, log_freq, **run_kwargs)
            except self.exception_type as e:
                tb = traceback.extract_tb(e.__traceback__)
                for frame in tb:
                    print(f"Exception {frame.filename} at line {frame.lineno}")
                    logging.error(f"frame {frame.filename} line {frame.lineno}")
                logging.error(f'Error {e}, moving on to next iteration {i+1}')
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
        self.inner_loop = inner_loop
        topdown_map = maps.get_topdown_map_from_sim(self.annotatedSimulator.sim, map_resolution=2048)
        self.agent.annotatedSimulator = self.annotatedSimulator
        self.explored_color = (200, 200, 200)
        self.unexplored_color = (0, 255, 0)
        recolor_map = np.array(
        [[40, 40, 40], self.unexplored_color, [0, 0, 0]], dtype=np.uint8)
        topdown_map = recolor_map[topdown_map]

        self.topdown_map = topdown_map
        self.agent.topdown_map = self.topdown_map
        x1, y1 = self.agent.toGrid(self.init_pos)
        x2, y2 = self.agent.toGrid([self.init_pos[0]+1, self.init_pos[1], self.init_pos[2]+1])
        pixels_per_sqm = abs(x1-x2)*abs(y1-y2)
        self.navigable_area_sqm = np.all(self.topdown_map == self.unexplored_color, axis=-1).sum() / pixels_per_sqm
        print(f'Navigable area: {self.navigable_area_sqm} sqm, pixels per sqm: {pixels_per_sqm}')

        print(f'\n===================STARTING RUN: {self.curr_run_name} ===================\n')
        for _ in range(inner_loop):
            rng_state = random.getstate()
            try:
                logging.info(f'Step {self.step}')
                actions = self.step_env(obs)
                if actions is None:
                    break
                # if self.add_noise and actions[0][1] > 0.01:
                #     theta = random.uniform(-0.2, 0.2)
                #     mag = random.uniform(0.1, 0.25)
                #     actions = list(actions) + [('rotate', theta), ('forward', -mag)]
                obs = self.annotatedSimulator.step(actions)

            except self.exception_type as e:
                tb = traceback.extract_tb(e.__traceback__)
                for frame in tb:
                    logging.error(f"Exception occurred in {frame.filename} at line {frame.lineno}")
                actions = [('forward', 0.2)]
                print(e)
                logging.error(f'Error {e}')
                print('\n\n\n\n\nERROR OCCURRED')
            finally:
                self.step += 1
                random.setstate(rng_state)
        self.post_run()
    
    def setup_run(self, **run_kwargs):
        raise NotImplementedError

    def step_env(self, obs):
        raise NotImplementedError

    def post_run_log(self, items=None):
        pass
        # if 'agent_location' in self.df:
        #     s = self.df['agent_location']
        #     pairs = list(zip(s[:-1], s[1:]))

        #     unpriv_map = self.unpriv_map.copy()
        #     mask = np.all(self.explored_map == self.explored_color, axis=-1)
        #     unpriv_map[mask] = self.explored_color

        #     for loc1, loc2 in pairs:
        #         c1 = self.toGrid(loc1)
        #         c2 = self.toGrid(loc2)
        #         cv2.arrowedLine(self.topdown_map, c1, c2, (0, 150, 0), 10)

        #         c1 = self.toGrid2(loc1)
        #         c2 = self.toGrid2(loc2)
        #         cv2.arrowedLine(unpriv_map, c1, c2, (0, 150, 0), 10)

        # path = f'logs/{self.outer_run_name}/{self.curr_run_name}/step_FINAL'
        # os.makedirs(path, exist_ok=True)
        # im = Image.fromarray(self.topdown_map, mode='RGB')
        # im.save(f'{path}/final_map.png')
        
        # im = Image.fromarray(unpriv_map, mode='RGB')
        # im.save(f'{path}/final_map_unpriv.png')

        # topdown_map = maps.get_topdown_map_from_sim(self.annotatedSimulator.sim, map_resolution=2048)
        # recolor_map = np.array(
        # [[40, 40, 40], self.unexplored_color, [0, 0, 0]], dtype=np.uint8)
        # topdown_map = recolor_map[topdown_map]
        # im = Image.fromarray(topdown_map, mode='RGB')
        # im.save(f'{path}/true_topdown.png')

        # im = Image.fromarray(self.unpriv_map, mode='RGB')
        # im.save(f'{path}/SLAM_topdown.png')

    def post_run(self):
        self.post_run_log()
        self.df.to_pickle(f'parallel/{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}/df_results.pkl')
        self.annotatedSimulator.sim.close()
        self.agent.reset()
        self.log_data['spend'] = self.agent.get_spend()
        self.log_data['default_rate'] = len(self.df[self.df['actions'] == -10])/len(self.df)
        self.log_data['episode_len'] = len(self.df)
        if self.exp_kwargs['parallel']:
            try:
                response = requests.post(f'http://localhost:{self.port}/log', json=self.log_data)
                if response.status_code != 200:
                    logging.error(f"Failed to send metrics: {response.text}")
            except self.exception_type as e:
                tb = traceback.extract_tb(e.__traceback__)
                for frame in tb:
                    logging.error(f"Frame {frame.filename} line {frame.lineno}")
                    logging.error(e)
                print(f"Error sending metrics: {e}")   

        logging.info('\n===================RUN COMPLETE===================\n')
        if self.log_freq == 1:         
            gif(f'parallel/{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}', multi=len(self.annotatedSimulator.sensors) > 1)
            print('saved gif')

    def log(self, images, success, metadata, copy_images=[]):

        path = f'parallel/{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}/step{self.step}'
        if success == 0:
            path += '_ERROR'
        os.makedirs(path)
        for ndx, im in enumerate(images):
            im.save(f'{path}/image{ndx}.png')
        for ndx, im in enumerate(copy_images):
            im.save(f'{path}/copy_image{ndx}.png')
        with open(f'{path}/details.txt', 'w') as file:     
            if success:
                for k, v in metadata.items():
                    file.write(f'{k}\n{v}\n\n')

    def set_state(self, pos=None, quat=None, agent_id=0):
        if pos is None:
            pos = self.init_pos 
        if quat is None:
            quat = self.default_quat

        init_state = habitat_sim.AgentState()
        init_state.position = pos
        init_state.rotation = quat
        self.annotatedSimulator.sim.get_agent(agent_id).set_state(init_state)
    

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


    def dist2d(self, p1, p2):
        return np.linalg.norm(np.array([p1[0], p1[2]]) - np.array([p2[0], p2[2]]))


    def generate_topdown(self, real_actions, agent_id=0, goal=None, zoom=12):

        agent_state = self.get_agent_state(agent_id)
        agent_coords = self.toGrid(agent_state.position)

        # if goal is not None:
        #     goal_coords = self.toGrid(goal)
        #     cv2.circle(self.topdown_map, goal_coords, radius=25, color=(255, 255, 0), thickness=-1)
            # cv2.putText(self.topdown_map, 'GOAL', (goal_coords[0] + 10, goal_coords[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  
        
        topdown_map = self.topdown_map.copy()

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
            #cv2.line(self.topdown_map, agent_coords, act_coords, self.explored_color, 50)
            text = str(action) 
            (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
            circle_center = (act_coords[0], act_coords[1])
            circle_radius = max(text_width, text_height) // 2 + 15
            cv2.circle(topdown_map, circle_center, circle_radius, (255, 255, 255), -1)
            text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
            # put_text_on_image(self.topdown_map, '', background_color=(255, 255, 255), location='top_left', text_size=text_size, text_thickness=text_thickness+1)
            cv2.putText(topdown_map, text, text_position, font, text_size, (0, 0, 0), text_thickness+1)

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
        
        return zoomed_map
