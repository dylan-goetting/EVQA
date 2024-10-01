
from collections import Counter
import csv
import gzip
import json
import logging
import math
import os
import pdb
import random
from sqlite3 import DatabaseError
import time
from tkinter import Y
from turtle import update
from habitat.core import agent
from habitat.datasets.rearrange.samplers import scene_sampler
from habitat_baselines.il import data
import habitat_sim
from networkx import shortest_path
import numpy as np
import pandas as pd
from PIL import Image
from regex import D
from src.utils import *
from src.vlm import VLM, GPTModel, GeminiModel
from src.annoatedSimulator import AnnotatedSimulator
from src.dynamicBench import DynamicBench
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis
import concurrent.futures
import wandb

class NavBench(DynamicBench):

    task = 'objnav'
    default_quat = quaternion.quaternion(0.70536607503891, 0, 0.708843231201172, 0)

    def setup_experiment(self, split, scene_ids, **kwargs):

        self.split = split
        self.sim_kwargs['scene_config'] =  f"datasets/hm3d/hm3d_annotated_{self.split}_basis.scene_dataset_config.json"
        files = [f for f in os.listdir(f'datasets/hm3d/{self.split}/')]

        if scene_ids:
            files = [f for f in files if int(f[2:5]) in scene_ids]
        files.sort()
        self.files = files        
        self.data_len = len(files)
    
    def setup_run(self, data_ndx, **kwargs):
        
        self.goal_loc = [0, 0, 0]
        while True:
            try:
                f = self.files[data_ndx]
                hsh = f[6:]
                self.sim_kwargs['scene_id'] = f[2:5]
                self.sim_kwargs['scene_path'] = f'datasets/hm3d/{self.split}/00{f[2:5]}-{hsh}/{hsh}.basis.glb'
                self.annotatedSimulator = AnnotatedSimulator(**self.sim_kwargs)
                self.curr_target = 'WASHER AND DRYER'
                self.curr_related_objects = []
                y_counter = Counter()
                y_counter.update([-1000])
                mapping = {}
                while max(y_counter.values()) < 3:
                    point = self.annotatedSimulator.sim.pathfinder.get_random_navigable_point()
                    y_counter.update([round(point[1], 2)])
                    mapping[round(point[1], 2)] = point

                print('took', len(mapping), 'tries')               
                self.init_pos = mapping[max(y_counter, key=y_counter.get)]

                break
            except Exception as e:
                print(e)
                print('\n\n\n')
                continue    

        self.run_metadata = {
            'task': self.curr_run_name,
            'sensors': self.agent.sensors,
            'fov': self.agent.fov,
            'seed': self.random_seed,
            'scene_id': self.annotatedSimulator.scene_id,
            'init_pos': self.init_pos,
            'data_ndx': data_ndx,
        }

        self.set_state()
        self.curr_run_name = f'{data_ndx}_{self.annotatedSimulator.scene_id}'
        obs = self.annotatedSimulator.step([('forward', 0)])
        return obs


    def step_env(self, obs):

        actions, row, metadata, real_actions, zoomed_map = self.agent.step_env(obs)
        row['goal_object'] = self.curr_target

        images = self.get_sensor_images(obs, convert=False) + [zoomed_map]
        copies = []
        for sensor in self.annotatedSimulator.sensors:

            copy = obs[f'color_sensor_{sensor}']['image'].copy()
            self.agent.draw_arrows(real_actions.keys(), copy, obs['agent_state'], obs['agent_state'].sensor_states[f'color_sensor_{sensor}'], chosen_action=row['actions'], real_actions=real_actions)
            copies.append(copy)
        copies.append(self.agent.topdown_map)

        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)

        if self.step % self.log_freq == 0 or row['success'] == 0:
            images = [Image.fromarray(im[:, :, 0:3], mode='RGB') for im in images]
            copies = [Image.fromarray(im[:, :, 0:3], mode='RGB') for im in copies]
            self.log(images, row['success'], metadata, copy_images=copies)

        max_steps = self.navigable_area_sqm**(0.4) * 12
        if self.step >= max_steps:
            print("STOPPING EARLY, MAX STEPS")
            return None
        
        return actions    

    def post_run_log(self):
        self.log_data['navigable_area'] = self.navigable_area_sqm
        self.log_data['task_data']['explore_curve'] = list(self.df['explored'])
        self.log_data['explored'] = self.df['explored'].iloc[-1]
        super().post_run_log()





class GOATBench(DynamicBench):

    task = 'goat'

    def setup_experiment(self, split,  **kwargs):
        self.split_name = split
        self.split = 'val' if 'val' in split else 'train'
        self.sim_kwargs['scene_config'] =  f"datasets/hm3d/hm3d_annotated_{self.split}_basis.scene_dataset_config.json"
        self.sim_kwargs['goal_image_agent'] = True
        dir = split
        self.all_episodes = []
        self.goals = {}
        for f in os.listdir(f'datasets/goatBench/{dir}/content'):
            with gzip.open(f'datasets/goatBench/{dir}/content/{f}', 'rt') as gz:
                loaded = json.load(gz)
                hsh = f.split('.')[0]
                self.goals[hsh] = loaded['goals']
                self.all_episodes += loaded['episodes']
        self.data_len = len(self.all_episodes)
        
        logging.info(f'Loaded {len(self.all_episodes)} episodes from {len(self.all_episodes)} files')


    def setup_run(self, max_steps_per_goal=5, success_thresh=2.5, data_ndx=0, **kwargs):

        episode = self.all_episodes[data_ndx]
        f, glb = episode['scene_id'].split('/')[-2:]
        hsh = f[6:]
        goals = self.goals[hsh]
        self.sim_kwargs['scene_id'] = f[2:5]
        self.sim_kwargs['scene_path'] = f'datasets/hm3d/{self.split}/{f}/{glb}'
        self.annotatedSimulator = AnnotatedSimulator(**self.sim_kwargs)
        self.curr_episode = []

        self.init_pos = np.array(episode['start_position'])
        self.set_state(pos=self.init_pos, quat=episode['start_rotation'])
        for goal in episode['tasks']:
            name = goal[0]
            mode = goal[1]
            
            target = {'name': name, 'mode': mode, 'id': goal[2], 'view_points': []}
            
            descriptions = goals[f'{f[6:]}.basis.glb_{name}']
            for d in descriptions:
                if mode == 'object':
                    target['view_points'] += [a['agent_state']['position'] for a in d['view_points']]
                else:
                    if d['object_id'] == goal[2]:
                        target['view_points'] = [a['agent_state']['position'] for a in d['view_points']]
                        if mode == 'description':
                            target['lang_desc'] = d['lang_desc']
                        if mode == 'image':
                            ndx = goal[3]
                            target['image_position'] = d['image_goals'][ndx]['position']
                            target['image_rotation'] = d['image_goals'][ndx]['rotation']

            self.curr_episode.append(target)

        logging.info(f'\n\nSTARTING EPISODE {data_ndx}, SCENE: {self.annotatedSimulator.scene_id}')

        print(f'STARTING EPISODE {data_ndx}')
        print(f'Running episode with {len(self.curr_episode)} goals')
        for i, obj in enumerate(self.curr_episode):
            print(f'Goal {i}: {obj["name"]}, {obj["mode"]}')

        self.curr_goal_ndx = 0
        self.curr_run_name = f"{data_ndx}_{self.annotatedSimulator.scene_id}"
        self.last_goal_reset = -1
        goal = self.curr_episode[self.curr_goal_ndx]
        self.run_metadata = {
            'task': self.curr_run_name,
            'sensors': self.agent.sensors,
            'fov': self.agent.fov,
            'seed': self.random_seed,
            'scene_id': self.annotatedSimulator.scene_id,
            'init_pos': self.init_pos,
            'max_steps_per_goal': max_steps_per_goal,
            'success_thresh': success_thresh,
            'split': self.split_name,
            'data_ndx': data_ndx,
        }
        self.agent.reset_goal()
        self.path = habitat_sim.MultiGoalShortestPath()
        self.path.requested_ends = np.array(goal['view_points'], dtype=np.float32)
        self.path.requested_start = self.init_pos
        if self.annotatedSimulator.sim.pathfinder.find_path(self.path):
            self.curr_shortest_path = self.path.geodesic_distance
        else:
            print('NO PATH FOUND')
            self.curr_shortest_path = 1000
        print(f'Current {goal["mode"]}: {goal["name"]}, GEODESIC: {self.curr_shortest_path}, num_view_points: {len(goal["view_points"])}')

        obs = self.annotatedSimulator.step([('forward', 0)])
        return obs

    def step_env(self, obs):
        goal = self.curr_episode[self.curr_goal_ndx]

        obs['goal'] = goal

        goal_ims = []
        if goal['mode'] == 'image':
            position = goal['image_position']
            rotation = goal['image_rotation']
            goal_im = self.annotatedSimulator.get_goal_image(position, rotation)
            put_text_on_image(goal_im, f"GOAL IMAGE: {goal['name']}", background_color=(255, 255, 255), location='top_center')
            goal_ims.append(goal_im)
            obs['goal_image'] = goal_im

        actions, row, metadata, real_actions, zoomed_map = self.agent.step_env(obs)
        images = self.get_sensor_images(obs, convert=False) + [zoomed_map]

        self.path.requested_start = obs['agent_state'].position
        if self.annotatedSimulator.sim.pathfinder.find_path(self.path):
            min_euclidian = self.path.geodesic_distance
        else:
            logging.info('NO PATH FOUND')
            min_euclidian = 1000
        min_dist = min_euclidian
        row['distance_to_goal'] = min_dist
        row['shortest_path_for_goal'] = self.curr_shortest_path
        row['curr_goal_steps'] = self.step - self.last_goal_reset
        row['curr_shortest_path'] = self.curr_shortest_path
        metadata['DIST TO GOAL'] = row['distance_to_goal']
       
        print('distance to goal', round(row['distance_to_goal'], 2), 'min euclidian', round(min_euclidian, 2))
        row['spl'] = 0

        done = False
        new_goal = False
        goal_reached = False

        if min_dist < self.run_metadata['success_thresh'] and (row['actions'] == -1 or self.step + 1 - self.last_goal_reset > self.run_metadata['max_steps_per_goal']):
            print(f"SUCESSFULLY FINISHED GOAL {self.curr_goal_ndx} in {round(self.agent.distance_traveled, 3)} meters")
            new_goal = True
            print('THE OPTIMAL PATH WAS:', round(self.curr_shortest_path, 3))
            row['spl'] = (self.curr_shortest_path)/max(self.curr_shortest_path, self.agent.distance_traveled)
            if 'goal_data' not in self.log_data['task_data']:
                self.log_data['task_data']['goal_data'] = []
            self.log_data['task_data']['goal_data'].append({
                'goal': goal['name'],
                'goal_mode': goal['mode'],
                'goal_index': self.curr_goal_ndx,
                'goal_reached': 1,
                'spl': row['spl'],
            })
            goal_reached = True
            row['finish_status'] = 'success'

        elif row['actions'] == -1 or self.step + 1 - self.last_goal_reset > self.run_metadata['max_steps_per_goal']:
            if row['actions'] == -1:
                row['finish_status'] = 'fp'
                print('FALSE POSITIVE')
            else:
                row['finish_status'] = 'max_steps'
                print('REACHED MAX STEPS')
            new_goal = True
            goal_reached = False
            if 'goal_data' not in self.log_data['task_data']:
                self.log_data['task_data']['goal_data'] = []

            self.log_data['task_data']['goal_data'].append({
                'goal': goal['name'],
                'goal_mode': goal['mode'],
                'goal_index': self.curr_goal_ndx,
                'goal_reached': 0,
                'spl': 0,
            })

        copies = []
        for sensor in self.annotatedSimulator.sensors:
            copy = obs[f'color_sensor_{sensor}']['image'].copy()
            self.agent.draw_arrows(real_actions, copy, obs['agent_state'], obs['agent_state'].sensor_states[f'color_sensor_{sensor}'], chosen_action=row['actions'], real_actions=real_actions)
            if new_goal and goal_reached:
                background_color = (0, 100, 0) 
            elif new_goal:
                background_color = (100, 0, 0) 
            else:
                background_color = (255, 255, 255)  
            put_text_on_image(copy, f"{self.curr_goal_ndx}: {goal['name']}({goal['mode'][0]})_{round(min_dist, 2)}", background_color=background_color, location='top_left', text_size=2.3)
            copies.append(copy)

        copies += goal_ims
        copies.append(self.topdown_map)

        row['goal_reached'] = goal_reached
        row['new_goal'] = new_goal
        if new_goal:
            self.curr_goal_ndx += 1
            self.last_goal_reset = self.step
            self.agent.reset_goal()
            if self.curr_goal_ndx >= len(self.curr_episode):
                done = True
                print("FINISHING TRAJECTORY, NO MORE GOALS")
            else:
                actions = [('forward', 0)]
                goal = self.curr_episode[self.curr_goal_ndx]
                self.path.requested_ends = np.array(goal['view_points'], dtype=np.float32)
                self.path.requested_start = obs['agent_state'].position
                if self.annotatedSimulator.sim.pathfinder.find_path(self.path):
                    self.curr_shortest_path = self.path.geodesic_distance
                else:
                    print('NO PATH FOUND')
                    self.curr_shortest_path = 1000
                print(f'New goal {goal["mode"]}: {goal["name"]}, GEODESIC: {self.curr_shortest_path}')


        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        if self.step % self.log_freq == 0 or row['success'] == 0:
            images = [Image.fromarray(im[:, :, 0:3], mode='RGB') for im in images]
            copies = [Image.fromarray(im[:, :, 0:3], mode='RGB') for im in copies]
            self.log(images, row['success'], metadata, copy_images=copies)
            
        if done:
            return None
        return actions  



class HMONBench(DynamicBench):

    task = 'hmon'

    def setup_experiment(self, split, version=2, **kwargs):
        self.all_episodes = []
        self.version = version
        self.split = split

        if self.version == 2:
            data_path = 'hm3d'
        else:
            self.split = 'val'
            data_path = 'hm3d_v0.1'
        self.sim_kwargs['scene_config'] =  f"datasets/{data_path}/hm3d_annotated_{self.split}_basis.scene_dataset_config.json"

        if self.split == 'train':
            dir = 'train'
        else:
            dir = 'val'
        if version == 2:
            nm = 'hmon2023'
        else:
            dir = 'val'
            nm = 'hmon2022'
        self.goals = {}
        for f in os.listdir(f'datasets/{nm}/{dir}/content'):
            with gzip.open(f'datasets/{nm}/{dir}/content/{f}', 'rt') as gz:
                js = json.load(gz)
                hsh = f.split('.')[0]
                self.goals[hsh] = js['goals_by_category']
                self.all_episodes += js['episodes']
        self.data_len = len(self.all_episodes)
        # self.missed = [945, 946, 947,  948,   949,  950, 951,  952,  953, 954,  955,  956,  957,  958, 959,  960,  961,  962, 963,  964,  965,  966,  967,  968,  969,  970,  971]
        # m = [self.all_episodes[i] for i in self.missed]
        # self.all_episodes = m 

    def setup_run(self, success_thresh=2.5, data_ndx=0, **kwargs):
        
        episode =self.all_episodes[data_ndx]
        f = episode['scene_id'].split('/')[1:]
        # pdb.set_trace()
        self.sim_kwargs['scene_id'] = f[1][2:5]
        if self.version == 2:
            data_path = 'hm3d'
        else:
            data_path = 'hm3d_v0.1'

        self.sim_kwargs['scene_path'] = f'datasets/{data_path}/{self.split}/{f[1]}/{f[2]}'
        self.annotatedSimulator = AnnotatedSimulator(**self.sim_kwargs)
        self.annotatedSimulator.do_annotate_image = False
        self.false_positives = 0
        self.false_negatives = 0
        goals = self.goals[f[1][6:]]
        all_objects = goals[f'{f[-1]}_{episode["object_category"]}']
        view_positions = []
        for obj in all_objects:
            for vp in obj['view_points']:
                view_positions.append(vp['agent_state']['position'])
        self.path = habitat_sim.MultiGoalShortestPath()
        self.path.requested_ends = np.array(view_positions, dtype=np.float32)
        logging.info(f'RUNNING EPISODE {data_ndx} with {episode["object_category"]} and {len(all_objects)} instances. GEODESIC DISTANCE: {episode["info"]["geodesic_distance"]}, NUM VIEWPOINTS: {len(view_positions)}')
        self.log_data.update(
                        {'spl': 0, 'distance_traveled': 0, 
                        'goal_reached': 0})
        if episode['object_category'] == 'tv_monitor':
            episode['object_category'] = 'tv screen'
        self.curr_episode = {'object': episode['object_category'], 'shortest_path': episode['info']['geodesic_distance'], 'object_positions': [a['position'] for a in all_objects], 'view_positions': view_positions}
        self.init_pos = np.array(episode['start_position'])
        self.set_state(pos=self.init_pos, quat=episode['start_rotation'])
        self.curr_run_name = f"{data_ndx}_{f[1][2:5]}"
        self.last_goal_reset = -1
        self.run_metadata = {
            'task': self.curr_run_name,
            'sensors': self.annotatedSimulator.sensors,
            'fov': self.annotatedSimulator.fov,
            'seed': self.random_seed,
            'scene_id': self.annotatedSimulator.scene_id,
            'success_thresh': success_thresh, **self.curr_episode,
            'data_ndx': data_ndx,
        }

        obs = self.annotatedSimulator.step([('forward', 0)])
        return obs

    def step_env(self, obs):
        obs['goal'] = self.curr_episode['object']
        agent_state = obs['agent_state']
        actions, row, metadata, real_actions, zoomed_map = self.agent.step_env(obs)
        images = self.get_sensor_images(obs, convert=False) + [zoomed_map]    
        self.path.requested_start = obs['agent_state'].position
        if self.annotatedSimulator.sim.pathfinder.find_path(self.path):
            min_euclidian = self.path.geodesic_distance
        else:
            logging.info('NO PATH FOUND')
            min_euclidian = 1000
        
        row['distance_to_goal'] = min_euclidian
        metadata['DIST TO GOAL'] = row['distance_to_goal']
        row['spl'] = 0
        row['goal_reached'] = False

        done = False
        print('distance to goal', round(row['distance_to_goal'], 2), 'min euclidian', round(min_euclidian, 2))
        if min_euclidian < self.run_metadata['success_thresh'] and (row['actions'] == -1 or self.step + 1 > self.inner_loop):
            print(f"SUCESSFULLY FINISHED GOAL in {round(self.agent.distance_traveled, 2)} meters of distance. Shortest path was {round(self.curr_episode['shortest_path'], 2)}")
            row['goal_reached'] = True 
            row['spl'] = (self.curr_episode['shortest_path'])/max(self.curr_episode['shortest_path'], self.agent.distance_traveled)
            done = True
            self.log_data.update(
                {'spl': row['spl'], 'distance_traveled': row['distance_traveled'], 
                'goal_reached': row['goal_reached']})
            row['finish_status'] = 'success'

        elif row['actions'] == -1 or self.step + 1 >= self.inner_loop:
            done = True
            if row['actions'] == -1:
                row['finish_status'] = 'fp'
                print('FALSE POSITIVE')
            else:
                row['finish_status'] = 'max_steps'
                print('REACHED MAX STEPS')

        copies = []
        for sensor in self.annotatedSimulator.sensors:
            copy = obs[f'color_sensor_{sensor}']['image'].copy()
            self.agent.draw_arrows(real_actions, copy, agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], chosen_action=row['actions'], real_actions=real_actions)
            if row['goal_reached']:
                background_color = (0, 100, 0) 
            elif self.step == self.inner_loop - 1:
                background_color = (100, 0, 0) 
            else:
                background_color = (255, 255, 255)  
            put_text_on_image(copy, f"{self.curr_episode['object']}_{np.round(row['distance_to_goal'], 2)}", background_color=background_color, location='top_left', text_size=2.3)
            copies.append(copy)

        copies.append(self.topdown_map)

        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        if self.step % self.log_freq == 0 or row['success'] == 0:
            images = [Image.fromarray(im[:, :, 0:3], mode='RGB') for im in images]
            copies = [Image.fromarray(im[:, :, 0:3], mode='RGB') for im in copies]
            self.log(images, row['success'], metadata, copy_images=copies)
            
        if done:
            self.agent.reset()
            return None

        return actions


class EQABench(DynamicBench):


    task = 'eqa'

    def setup_experiment(self, scene_ids, **kwargs):
        if scene_ids is None:
            scene_ids = range(515)
        file1 = 'datasets/EQA/questions.csv'
        file2 = 'datasets/EQA/scene_init_poses.csv'
        self.answerVLM = GeminiModel(sys_instruction='You are a 5 time world champion question answerer. An agent sends you images and questions, and you intelligently respond with the correct answer ', 
                                  model=self.vlm.name)

        with open(file1) as f:
            self.questions_data = [
                {"qid": idx, **{k: v for k, v in row.items()}}
                for idx, row in enumerate(csv.DictReader(f, skipinitialspace=True)) if idx in scene_ids and row['label'] == 'location'
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
        self.questions_data.sort(key=lambda x: x["qid"])
        self.data_len = len(self.questions_data)

    def setup_run(self, data_ndx=0, **kwargs):
        self.interesting_images = {'A': [], 'B': [], 'C': [], 'D': []}
        self.question_data = self.questions_data[data_ndx]
        self.quid = self.question_data["qid"]
        scene = self.question_data["scene"]
        floor = self.question_data["floor"]
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
        logging.info(f'STARTING RUN {data_ndx} out of {self.data_len} \nQUESTION: {self.question_data["question"]} It is type {self.question_data["label"]} BUDGET STEPS FOR THIS RUN: {num_step}')
        

        self.run_metadata = {
        'max_steps': num_step,
        'data_ndx': data_ndx,
        'quid': self.quid,
        'sensors': self.annotatedSimulator.sensors,
        'fov': self.annotatedSimulator.fov,
        'seed': self.random_seed,
        'scene_id': self.annotatedSimulator.scene_id,
        'init_pos': self.init_pos,
        'question_type': self.question_data["label"],
        }  

        self.annotatedSimulator.priv_actions = False
        self.annotatedSimulator.do_draw_arrows = False #points if draw_arrows else None
        self.annotatedSimulator.do_annotate_image = False
        self.annotatedSimulator.objects_to_annotate = []
        self.init_pos = np.array(init_pts)
        self.set_state(init_pts, rotation)

        self.curr_run_name = f'{data_ndx}_{self.annotatedSimulator.scene_id}'
        obs = self.annotatedSimulator.step([('forward', 0)])
        return obs


    def step_env(self, obs):
        if self.step >= self.curr_run_steps:
            print("MAX STEPS REACHED")
            return None
        obs['question_data'] = self.question_data
        agent_state = obs['agent_state']
        actions, row, metadata, real_actions, zoomed_map = self.agent.step_env(obs)
        agent_state = self.get_agent_state()
        
        question = self.question_data["question"]
        choices = [c.split("'")[1] for c in self.question_data["choices"].split("',")]
        # choices.append('I need to explore the environment further')
        answer = self.question_data["answer"]
        vlm_question = question
        vlm_pred_candidates = ["A", "B", "C", "D"]
        for token, choice in zip(vlm_pred_candidates, choices):
            vlm_question += "\n" + token + "." + " " + choice
        
        images = self.get_sensor_images(obs) + zoomed_map

        metadata['PREDICTION'] = row['answer']
        metadata['GROUND TRUTH'] = answer

        copies = []
        for i, sensor in enumerate(self.annotatedSimulator.sensors):
            copy = images[i].copy()

            self.draw_arrows(real_actions, copy, agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], chosen_action=row['actions'], real_actions=real_actions)
            put_text_on_image(images[i], f"QUESTION: {question}", background_color=(255, 255, 255), location='top_left', text_size=1.5, text_thickness=2)
            put_text_on_image(copy, f"QUESTION: {question}", background_color=(255, 255, 255), location='top_left', text_size=1.5, text_thickness=2)
            if row['answer'] and not row['answer'] == 'E' and row['answer'] in vlm_pred_candidates:
                ndx = vlm_pred_candidates.index(row['answer'])
                if ndx < len(choices):
                    ans = choices[ndx]
                else:
                    ans = 'error'
                    print(f'Error: model answered: {row["answer"]} but there are only the choices are {choices} and the candiates are {vlm_pred_candidates}')
                color = (0, 255, 0) if row['answer'] == answer else (255, 0, 0)
                put_text_on_image(copy, f"{ans}", background_color=color, location='top_right', text_size=2, text_thickness=2)
                put_text_on_image(images[i], f"QUESTION: {question}", background_color=(255, 255, 255), location='top_left', text_size=1.5, text_thickness=2)

            copies.append(copy)

        copies.append(self.topdown_map)

        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)

        if self.step % self.log_freq == 0 or row['success'] == 0:
            images = [Image.fromarray(im[:, :, 0:3], mode='RGB') for im in images]
            copies = [Image.fromarray(im[:, :, 0:3], mode='RGB') for im in copies]
            self.log(images, row['success'], metadata, copy_images=copies)

        sorted_counter = list(sorted(self.answer_counter.items(), key=lambda x: x[1], reverse=True))
        print('answers counter', sorted_counter)
        t1 = 2 if len(self.annotatedSimulator.sensors) > 1 else 3
        t2 = 10 if len(self.annotatedSimulator.sensors) > 1 else 8
        if self.step >= self.run_metadata['max_steps']:
            print("STOPPING EARLY, MAX STEPS REACHED")
            return None

        return actions     

    def post_run(self):

        self.run_metadata['final_answer1'] = 'E' #random.choice(['A', 'B', 'C', 'D'])
        path = f'logs/{self.outer_run_name}/{self.inner_run_name}{self.curr_run_name}/step_FINAL'
        os.makedirs(path, exist_ok=True)
        final_counter = self.agent.final_answer()
        if sum([len(v) for k, v in final_counter.items()]) == 0:
            print('NO FINAL ANSWER')
            self.log_data['did_not_answer'] = True
            return
        v = max(final_counter.items(), key=lambda x: len(x[1]))
        images, r = v[1][0]
        self.run_metadata['final_answer1'] = v[0]
        self.finalAnswerVlm.get_spend()
        self.log_data['success'] = max(self.answer_counter, key=self.answer_counter.get) == self.question_data['answer']
        self.log_data['sucess1'] = self.run_metadata['final_answer1'] == self.question_data['answer']

        if len(images) > 0:
            for i, im in enumerate(images):
                im = Image.fromarray(im[:, :, 0:3], mode='RGB')
                im.save(f'{path}/final_image_{i}.png')
                
            with open(f'{path}/final_answer.txt', 'w') as f:
                f.write(r)
                f.write(f'\n Answer counter {self.answer_counter}')
                f.write(f'\n final answer counters, {[(k, len(v)) for k, v in final_counter.items()]}')
                
        return super().post_run()


            

    
# class VLNCE(DynamicBench):

#     task = 'vlnce'

#     def setup_experiment(self, split, **kwargs):
#         self.all_episodes = []
#         # self.version = version
#         self.split = split

#         self.sim_kwargs['scene_config'] =  f"datasets/mp3d/scene_dataset_config.json"

#         self.goals = {}
#         with gzip.open(f'datasets/vlnce/{split}/{split}.json.gz', 'rt') as gz:
#             js = json.load(gz)
#             self.all_episodes += js['episodes']
#         self.all_episodes.sort(key=lambda x: x['episode_id'])
#         self.data_len = len(self.all_episodes)

#     def setup_run(self, history=7, mask_thinking=True, add_timesteps_prompt=True, draw_arrows=True,
#             points=None, consistency=1, use_map=0, uniform=False, 
#             explore_factor=0, map_type='priv', success_thresh=2.5, data_ndx=0, **kwargs):
        
#         episode =self.all_episodes[data_ndx]
#         f = episode['scene_id'].split('/')[1]
#         # pdb.set_trace()
#         self.sim_kwargs['scene_id'] = f[1][0:2]

#         self.sim_kwargs['scene_path'] = f'datasets/{episode["scene_id"]}'
#         self.annotatedSimulator = AnnotatedSimulator(**self.sim_kwargs)
#         self.annotatedSimulator.do_annotate_image = False
#         self.false_positives = 0
#         self.false_negatives = 0
        
#         goal_position = episode['goals'][0]['position']
#         self.path = habitat_sim.ShortestPath()
#         self.path.requested_end = np.array(goal_position, dtype=np.float32)

#         logging.info(f'RUNNING EPISODE {data_ndx}, instruction {episode["instruction"]["instruction_text"]}')
#         self.log_data.update(
#                         {'spl': 0, 'spl_1.5': 0, 'spl_xz': 0, 'distance_traveled': 0, 
#                         'goal_reached': 0, 'goal_reached_1.5': 0, 'goal_reached_xz': 0})

#         self.curr_episode = {'instruction': episode['instruction']['instruction_text'], 'goal_position': goal_position, 'shortest_path': episode['info']['geodesic_distance']}
#         self.init_pos = np.array(episode['start_position'])
#         self.set_state(pos=self.init_pos, quat=episode['start_rotation'])

#         self.curr_run_name = f"{data_ndx}_{f[1][0:2]}"
#         self.last_goal_reset = -1
#         self.run_metadata = {
#             'task': self.curr_run_name,
#             'history': history,
#             'points': tuple(points) if points else 0,
#             'arrows': draw_arrows,
#             'consistency': consistency,
#             'mask_thinking': mask_thinking,
#             'add_timesteps_prompt': add_timesteps_prompt,
#             'sensors': self.annotatedSimulator.sensors,
#             'fov': self.annotatedSimulator.fov,
#             'seed': self.random_seed,
#             'scene_id': self.annotatedSimulator.scene_id,
#             'init_pos': self.init_pos,
#             'use_map': use_map,
#             'uniform': uniform,
#             'explore_factor': explore_factor,
#             'map_type': map_type,
#             'success_thresh': 3, **self.curr_episode,
#             'use_euclidian': kwargs.get('euclid', True),
#             'data_ndx': data_ndx,
#         }


#         obs = self.annotatedSimulator.step([('forward', 0)])
#         self.history = {}
#         return obs

#     def step_env(self, obs):
#         rng_state = random.getstate()
#         agent_state = obs['agent_state']
#         raw_images = [a.copy() for a in self.get_sensor_images(obs, convert=False)]
#         # def done_thread():
#         #     if self.step - self.called_done < 2:
#         #         return False, None
#         #     answer_prompt = (f"The agent has has been tasked with navigating to a {self.curr_episode['object'].upper()}. The agent has sent you an image taken from its current location. "
#         #                      f'Your job is to determine whether the agent is VERY CLOSE to a {self.curr_episode["object"]} Note a chair is distinct from a sofa which is distinct from a bed. '
#         #                      f"First, tell me what you see in the image, and tell me if there is a {self.curr_episode['object']} (You must be confident). Return 1 if the agent is CLOSE ENOUGH TO INTERACT with the {self.curr_episode['object']} and make sure the object you see is ACTUALLY a {self.curr_episode['object']}, Return 0 if if there is no {self.curr_episode['object']}, or if it is far away, or if you are not sure. Format your answer in the json {{'done': <1 or 0>}}")

#         #     def process_image(image):
#         #         r, p = self.answerVLM.call([image], answer_prompt, logprobs=5)
#         #         dct = self.parse_response(r)
#         #         if 'done' in dct and int(dct['done']) == 1:
#         #             return 1, r
#         #         return 0, r

#         #     with concurrent.futures.ThreadPoolExecutor() as executor:
#         #         futures = {executor.submit(process_image, image): image for image in raw_images}

#         #         for future in concurrent.futures.as_completed(futures):
#         #             isDone, r = future.result()
#         #             if isDone:
#         #                 print('ANSWER MODEL THINKS DONE:', r)
#         #                 self.called_done = self.step
#         #                 return True, r

#         #     return False, r
        
#         def preprocessing_thread():
#             # self.turned = self.step - 1
#             images = []
#             real_actions = self.annotate_image(agent_state, obs)
#             images += self.get_sensor_images(obs, convert=False)

#             # zoomed_map = self.generate_unpriv(real_actions, zoom=9) if self.run_metadata['map_type'] == 'unpriv' else self.generate_topdown(real_actions, zoom=9)
#             # if self.run_metadata['use_map'] and self.step - self.called_done > 1:
#             #     images.append(zoomed_map)

#             self.path.requested_start = agent_state.position
#             if self.annotatedSimulator.sim.pathfinder.find_path(self.path):
#                 min_euclidian = self.path.geodesic_distance
#             else:
#                 print('NO PATH FOUND')
#                 min_euclidian = 1000
            
#             return real_actions, images, None, min_euclidian
        
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             # future1 = executor.submit(done_thread)
#             future2 = executor.submit(preprocessing_thread)
            
#             real_actions, images, zoomed_map, min_euclidian = future2.result()
#             # done, r = future1.result()
#             done = False
#             r = ''
    
#         # multi = len(self.annotatedSimulator.sensors) > 1
#         # prompt = (

#         # f"TASK: NAVIGATE TO THE NEAREST {self.curr_episode['object'].upper()}, and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
#         # f"There are {len(real_actions) - 1} red arrows superimposed onto your observation{'s' if multi else''}, which represent potential actions. " 
#         # f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step - self.turned >= 3 else ''}"
#         # f"First, tell me what you see in your sensor observations, and if you have any leads on finding the {self.curr_episode['object'].upper()}. {'Second, tell me which sensor looks the most promising. ' if multi else 'Second, tell me which general direction you should go in. '}"
#         # f"Lastly, explain which action acheives that best, and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
#         # )

#         # if self.run_metadata['use_map'] and self.step - self.called_done > 1:
#         #     prompt = (
#         # f"TASK: NAVIGATE TO THE NEAREST {self.curr_episode['object'].upper()} and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
#         # f"There are {len(real_actions) - 1} red arrows superimposed onto your observation{'s' if multi else''}, which represent potential actions. " 
#         # f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step - self.turned >= 3 else ''}"
#         # "\nYou also have a topdown map of the environment, with unexplored area shown in GREEN. This map shows you where you have been in the past, shown in GRAY. "
#         # "The same actions you see superimposed on the RGB image are also shown on the top-down map. "
#         # f"First, tell me what you see in your sensor observations, and if you have any leads on finding the {self.curr_episode['object'].upper()}. {'Second, tell me which sensor looks the most promising. ' if multi else 'Second, tell me which general direction you should go in. '} {' Remember you can always turn around to search in a different area' if self.step - self.turned >= 3 else ''}"
#         # "If you are not sure where to go, analyze map to help you plan actions that lead towards the GREEN AREAS. "
#         # f"Lastly, combine both sources of informaiton and explain which action is the best and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
#         #     )
#         #  Your task is considered complete once you reach the final location described to you (regardless of how you got there). If you have reached that location choose special action -1
#         if self.step == 0:
#             prompt = f'You are a 5 time world champion navigating agent. TASK: \n {self.curr_episode["instruction"]}\nYou are currently at the start of the trajectory. Each numbered circle you see represents an action you can take. {"You also have action 0 which turns you around. " if self.step - self.turned >= 3 else ""} Tell me what the first step is following the trajectory, and tell me which action if best for that, and return it as the json {{"action": <action_key>}}'
#         else:
#             prompt = f'You are a 5 time world champion navigating agent. TASK: \n {self.curr_episode["instruction"]}\n You are successful once you are close to the final location described in the task. You have already started this task, and for context you are shown an image of where you started. There are {len(real_actions)} arrowed actions superimposed onto the image, which represent where you would move to if you chose such action number. {"You also have action 0 which turns you around. " if self.step - self.turned >= 3 else ""}. First tell me what you see and where you are currently. Which way should you go to navigate the final location described in the task? Lastly, tell me which action is best to complete this task, and return it as the json {{"action": <action_key>}} Note you CANNOT GO THROUGH CLOSED DOORS.'
        
#         history = self.make_history()
#         # pdb.set_trace()
#         self.vlm.session.history = history
#         # prompt = "TASK:\n" + self.curr_episode['instruction'] + '\nNote you may currently be paritally through completing this task. Look at your sensor observation and try to figure out where in the instruction trajectory you are. Each numbered circle you see represents an action you can take. Explain which action number is best to complete this goal, and return it as the json {"action": <action_key>}'

#         row = {'actions': -10, 'tokens_generated':0, 'success': 1, 'metadata': self.run_metadata, 'false_positives': self.false_positives, 'false_negatives': self.false_negatives,
#         'speed': 0, 'distance_traveled': self.distance_traveled, 'pivot': 1 if self.pivot is not None else 0,
#         'model': self.vlm.name, 'input_tokens': 0, 'agent_location': agent_state.position}
#         logs = []
#         if done:
#             row['actions'] = -1
#             rgb = self.topdown_map
#             green = np.sum(rgb == self.explored_color)
#             light_grey = np.sum(rgb == self.unexplored_color)
#             row['explored'] = green / (green + light_grey) 
#             row['distance_traveled'] = self.distance_traveled
#             resp = r
#             metadata = {}
#         else:
#             if self.pivot is not None:
#                 instruction = self.curr_episode['instruction']
#                 pivot_actions, log_images = self.pivot.run(raw_images[0], obs['depth_sensor_0'], instruction, agent_state, agent_state.sensor_states['color_sensor_0'])
#                 metadata = {'pivot_actions': pivot_actions}
#                 logs += log_images
#                 resp = ""
#                 row['actions'] = -10
#             else:
#                 row, metadata, resp = self.agent_self_consitency(prompt, images, row, self.run_metadata['consistency'])
#                 metadata['DONE RESP'] = r

        
#         images = self.get_sensor_images(obs, convert=False) 
#         if row["actions"] <= len(list(real_actions.keys())) and row["actions"] > 0:
#             mag, theta = list(real_actions.keys())[row["actions"]-1]
#             self.update_unpriv(mag, theta, agent_state, mag, clip_frac=1)
            
#         row['distance_to_goal'] = min_euclidian
#         metadata['DIST TO GOAL'] = row['distance_to_goal']
#         row['spl'] = 0
#         row['goal_reached'] = False

#         print('distance to goal', round(row['distance_to_goal'], 2), 'min euclidian', round(min_euclidian, 2))

#         if (min_euclidian < self.run_metadata['success_thresh']):
#             print(f"SUCESSFULLY FINISHED GOAL in {round(self.distance_traveled, 2)} meters of distance. Shortest path was {round(self.curr_episode['shortest_path'], 2)}")
#             row['goal_reached'] = True 
#             row['spl'] = (self.curr_episode['shortest_path'])/max(self.curr_episode['shortest_path'], self.distance_traveled)

#         elif row['distance_to_goal'] < self.run_metadata['success_thresh']:
#             self.false_negatives += 1
#             print('NEAR GOAL BUT MODEL DID NOT RETURN DONE')
#         elif row['actions'] == -1:
#             print(f'MODEL RETURNED DONE BUT STILL {round(min_euclidian, 2)} METERS AWAY')
#             self.false_positives += 1

#         copies = []
#         for sensor in self.annotatedSimulator.sensors:
#             copy = obs[f'color_sensor_{sensor}']['image'].copy()
#             self.draw_arrows(real_actions, copy, agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], chosen_action=row['actions'], real_actions=real_actions)
#             copies.append(copy)

#         copies.append(self.topdown_map)
#         self.history[self.step] = raw_images[0]

#         # if self.run_metadata['mask_thinking'] and row['success'] == 1 and self.run_metadata['history'] > 0:
#         #     pdb.set_trace()
#         #     self.vlm.session.history[-1].parts[0].text = f'\n' + '{"action": ' + str(row["actions"]) + '}'

#         self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
#         if self.step % self.log_freq == 0 or row['success'] == 0:
#             images = [Image.fromarray(im[:, :, 0:3], mode='RGB') for im in images+logs]
#             copies = [Image.fromarray(im[:, :, 0:3], mode='RGB') for im in copies]
#             self.log(images, resp, row['success'], metadata, copy_images=copies)
            
#         if row['goal_reached']:
#             self.log_data.update(
#                 {'spl': row['spl'], 'distance_traveled': row['distance_traveled'], 
#                 'goal_reached': row['goal_reached']})

#             return None
#         self.vlm.get_spend()
#         random.setstate(rng_state)
        
#         if self.pivot is not None and row['actions'] != -1:
#             return pivot_actions
#         actions = self.annotatedSimulator.move_choices(row['actions'], points= [(min(mag, 1), theta) for mag, theta in real_actions.keys()]) 
#         return actions

#     def make_history(self):
#         if self.step == 0:
#             return []
#         hist = []
#         for key, image in sorted(self.history.items(), key=lambda x: x[0]):
#             # put_text_on_image(image, f"{key}", background_color=(255, 255, 255), location='top_left', text_size=2.3)
#             if key%2 == 0:
#                 if key == 0:
#                     prompt = 'STARTING OBSERVATION TIME 0'
#                 # else:
#                 #     prompt = "PREVIOUS OBSERVATION AT TIME " + str(key)
#                     hist.append(    {
#                     "role": "user",
#                     "parts": [
#                         Image.fromarray(image[:, :, 0:3], mode='RGB'),
#                         prompt,
#                     ],
#                     })
#                 # put_text_on_image(image, f"START", background_color=(255, 255, 255), location='top_left', text_size=2.3
#         return hist
    