import gzip
from hmac import new
import json
from math import e
import os
from sqlite3 import DatabaseError
import sys
import pdb
import pickle
import random
from turtle import distance
from matplotlib import pyplot as plt
import numpy as np
import datetime
import cv2
import ast
import pandas as pd
from PIL import Image
from src.utils import *
from src.vlm import VLM
from src.annoatedSimulator import AnnotatedSimulator
import habitat_sim
import cv2

class DynamicBench: 

    task = 'Not defined'

    def __init__(self, sim_kwargs, vlm_agent: VLM, exp_kwargs, outer_run_name):

        self.sim_kwargs = sim_kwargs
        self.vlm = vlm_agent
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


            self.run_trajectory(inner_loop, log_freq, **run_kwargs)

    def run_trajectory(self, inner_loop, log_freq, **run_kwargs):
        self.step = 0
        self.init_pos = None
        self.df = pd.DataFrame({})
        self.log_freq = log_freq
        obs = self.setup_run(**run_kwargs)
        print(f'\n===================STARTING RUN: {self.curr_run_name} ===================\n')
        for _ in range(inner_loop):
            actions = self.step_env(obs)
            if actions is None:
                break
            obs = self.annotatedSimulator.step(actions)
            self.step += 1

        self.post_run()
    
    def setup_run(self, **run_kwargs):
        raise NotImplementedError

    def step_env(self, obs):
        raise NotImplementedError

    def post_run(self):
        self.df.to_pickle(f'logs/{self.outer_run_name}/{self.curr_run_name}/df_results.pkl')
        self.vlm.reset()
        self.annotatedSimulator.sim.close()
        self.get_costs()
        print('\n===================RUN COMPLETE===================\n')
        gif(f'logs/{self.outer_run_name}/{self.curr_run_name}')

    def log(self, prompt, images, response, success, metadata, copy_images=[]):
        path = f'logs/{self.outer_run_name}/{self.curr_run_name}/step{self.step}'
        if success == 0:
            path += '_ERROR'
        os.makedirs(path)
        for ndx, im in enumerate(images):
            im.save(f'{path}/image{ndx}.png')
        for ndx, im in enumerate(copy_images):
            im.save(f'{path}/copy_image{ndx}.png')
        with open(f'{path}/details.txt', 'w') as file:
            file.write(f'[PROMPT]\n{prompt}\n\n')
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
            return {}


    def get_agent_state(self):
        return self.annotatedSimulator.sim.get_agent(0).get_state()
    
    def set_state(self, pos=None, quat=None):
        if pos is None:
            pos = self.init_pos 
        if quat is None:
            quat = self.default_quat

        init_state = habitat_sim.AgentState()
        init_state.position = pos
        init_state.rotation = quat
        self.annotatedSimulator.sim.get_agent(0).set_state(init_state)
    
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
            except (IndexError, KeyError, TypeError) as e:
                print(e)
                row['success'] = 0
            finally:
                metadata['ACTIONS'] = row['actions']

            if row['actions'] in action_counter:
                action_counter[row['actions']]+= 1
            else:
                action_counter[row['actions']] = 1
            
            if action_counter[row['actions']] == consistency:
                print(f'Stepping, took {num_calls} calls')
                break
            else:
                if row['success']==1:
                    self.vlm.session.rewind()
        row['num_calls'] = num_calls

        return row, metadata, resp

    def get_sensor_images(self, obs):
        ims = [obs[f'color_sensor_{sensor}']['image'] for sensor in self.annotatedSimulator.sensors]
        images = []
        for im in ims:
            if im.shape[-1] == 4:
                im = im[:, :, 0:3]
            images.append(Image.fromarray(im, mode='RGB'))
        return images

    def get_costs(self):
        print('\n')
        print(f'GPT Mini would cost: {np.round(self.total_input_tokens*0.15/1000000 + self.total_output_tokens*0.6/1000000, 2)}')
        print(f'GPT 4o would cost: {np.round(self.total_input_tokens*5/1000000 + self.total_output_tokens*15/1000000, 2)}')
        print(f'Gemini 1.5pro would cost: {np.round(self.total_input_tokens*3.5/1000000 + self.total_output_tokens*10.50/1000000, 2)}')
        print(f'Gemini flash would cost: {np.round(self.total_input_tokens*0.35/1000000 + self.total_output_tokens*0.150/1000000, 2)}')
        




class NavBench(DynamicBench):

    task = 'obj_nav'
    default_quat = quaternion.quaternion(0.70536607503891, 0, 0.708843231201172, 0)

    def setup_experiment(self, split, scene_ids):

        self.split = split
        self.sim_kwargs['scene_config'] =  f"scenes/hm3d/hm3d_annotated_{self.split}_basis.scene_dataset_config.json"
        if self.split == 'train':
            json_file =  "scenes/hm3d/hm3d_annotated_train_basis.scene_dataset_config.json"
            with open(json_file, 'r') as f:
                data = json.load(f)
                scenes = data['stages']['paths']['.glb']
                scene_ids = set(int(s[2:5]) for s in scenes)
            files = [f for f in os.listdir('scenes/hm3d/train/') if int(f[2:5]) in scene_ids]
        else:
            files = os.listdir('scenes/hm3d/val/')
        if scene_ids:
            files = [f for f in files if int(f[2:5]) in scene_ids]
        files.sort()
        random.shuffle(files)
        self.files = files        
        
    
    def setup_run(self, history=7, mask_thinking=True, add_timesteps_prompt=True, draw_arrows=True,
            points=None, consistency=1, goals = []):

        while True:
            try:
                f = random.choice(self.files)
                hsh = f[6:]
                self.sim_kwargs['scene_id'] = f[2:5]
                self.sim_kwargs['scene_path'] = f'scenes/hm3d/{self.split}/00{f[2:5]}-{hsh}/{hsh}.basis.glb'
                self.annotatedSimulator = AnnotatedSimulator(**self.sim_kwargs)

                random.shuffle(goals)            
                for target, related in goals:
                    tries = 0
                    if os.path.exists(f'logs/{self.outer_run_name}/{target}_{self.annotatedSimulator.scene_id}'):
                        print(f'{target}_{self.annotatedSimulator.scene_id} ALREADY EXISTS')
                        continue
                    self.curr_target = target
                    self.curr_related_objects = []
                    for word in related + [target]:
                        self.curr_related_objects += self.annotatedSimulator.search_objects(word, exact=False)
                    print(f'Targeting object: {target}')
                    print(f'related objects: {len([obj.category.name() for obj in self.curr_related_objects])}')
                    if len(self.curr_related_objects) == 0:
                        continue
                    for _ in range(200):
                        point = self.annotatedSimulator.sim.pathfinder.get_random_navigable_point()
                        for idx, floor_height in enumerate(self.annotatedSimulator.floors):
                            tries += 1
                            if abs(point[1] - floor_height) < 0.1:
                                floor = idx
                                distances = [np.linalg.norm(point - obj.aabb.center) for obj in self.curr_related_objects if obj.aabb.center[1] < self.annotatedSimulator.floors[floor+1] and obj.aabb.center[1] > floor_height]
                                min_dist = 7 if target in ['kitchen', 'living room'] else 5.5
                                if len(distances) > 0 and min(distances) > min_dist and min(distances) < min_dist + 10:
                                    # print('found point, min_dist', min(distances), f'thresh: {min_dist}')
                                    self.init_pos = point
                                    break
                        if self.init_pos is not None:
                            break
                    if self.init_pos is not None:
                        break
                    print('sampling again')
                if self.init_pos is not None:
                        break
                print(f'Scene id {self.annotatedSimulator.scene_id} Could not find a valid starting position')

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
        }

        self.annotatedSimulator.do_draw_arrows = points if draw_arrows else None
        self.annotatedSimulator.do_annotate_image = False
        self.annotatedSimulator.objects_to_annotate = self.curr_related_objects
        self.set_agent_state()

        self.curr_run_name = f'{self.curr_target}_{self.annotatedSimulator.scene_id}'
        obs = self.annotatedSimulator.step([('forward', 0)])
        return obs

    def step_env(self, obs):
        agent_state = self.get_agent_state()
            
        prompt = f"It is now timestep {self.step}. You have moved to a new location within the environment. "
        prompt += (
        f"First, analyze your updated camera observation and tell me the spatial layout of what you see. "
        f"There may be some arrows superimposed onto the image, which represent valid actions. "
        f"Your task is to navigate to a {self.curr_target.upper()}. Think of a high level plan on how you can reach a {self.curr_target.upper()} from where you are now. ")
        prompt += f"""In addition to any actions labeled on the image, you have the following special actions.
{{
0: turn completely around, use this when you are at a DEAD END, or IF THERE ARE NO ARROWS labeled on the image. 
-1: DONE, you think you have reached the {self.curr_target.upper()}!!
}}
Think about how each action will move you. Then, select one action from the image or the special actions and explain how it helps you reach your goal. Return it as {{'action': <action_key>}}. Note you CANNOT go through CLOSED DOORS or through obstacles. 
"""
        if len(self.run_metadata['sensors']) > 1:
            num_sensors = len(self.annotatedSimulator.sensors)
            prompt = f"It is now timestep {self.step}. You have moved to a new location within the environment. "
            prompt += (
            f"First, tell me what you see in each of your sensor observations. Think about how the observations overlap, some objects will be visible in multiple sensors. "  
            f"There are arrows superimposed onto the {num_sensors} different images, which represent valid actions. "
            f"Your task to navigate to a {self.curr_target.upper()}. Think of a high level plan on how you can reach a {self.curr_target.upper()} from where you are now. ")
            prompt += f"""In addition to any actions labeled on the images, you have the following special actions.
{{
0: turn completely around, use this when you DONT SEE ANY GOOD ACTIONS, and want fresh observations. 
-1: DONE, you think you have reached the {self.curr_target.upper()}!!
}}
Think about how each action will move you. Then, select one action from the images or the special actions and explain how it helps you reach your goal. Return it as {{'action': <action_key>}}. Note you cannot go through closed doors or through obstacles. 
"""
        images = self.get_sensor_images(obs)
        row = {'actions': -10, 'tokens_generated':0, 'success': 1, 'metadata': self.run_metadata,
        'speed': 0, 'scene_id': self.annotatedSimulator.scene_id,
        'model': self.vlm.name, 'input_tokens': 0, 'agent_location': agent_state.position}
        row, metadata, resp = self.agent_self_consitency(prompt, images, row, self.run_metadata['consistency'])

        row['goal_object'] = self.curr_target
            
        min_dist = 1000
        closest_object = None

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
            self.annotatedSimulator.draw_arrows(copy, agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], self.run_metadata['points'], chosen_action=row['actions'])
            if copy.shape[-1] == 4:
                copy = copy[:, :, 0:3]
            copies.append(Image.fromarray(copy, mode='RGB'))
        row['closest_object'] = closest_object
        row['distance_to_goal'] = min_dist
        metadata['DIST TO GOAL'] = row['distance_to_goal']
    #if last three actions are -1, break
        # distances = []
        # for i in range(len(self.df)-4, len(self.df)):
        #     agent_state = self.df['agent_location'].iloc[i]
        #     distance = np.linalg.norm(agent_state - self.df['agent_location'].iloc[i-1])
        #     distances.append(distance)
        # average_distance = sum(distances) / len(distances)
        # print(f"Average distance between agent states in the last 4 steps: {average_distance}")

        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        if self.run_metadata['mask_thinking'] and row['success'] == 1:
            self.vlm.session.history[-1].parts[0].text = f'\n' + '{"action": ' + str(row["actions"]) + '}'
        if self.step % self.log_freq == 0 or row['success'] == 0:
            self.log(prompt, images, resp, row['success'], metadata, copy_images=copies)

        if self.step >= 3 and self.df['actions'].iloc[-3:].tolist().count(-1) >= 2:
            print("STOPPING EARLY, DONE")
            return None

        actions = self.annotatedSimulator.move_choices(row['actions'], points=self.run_metadata['points'] if self.run_metadata['arrows'] else None)        
        return actions

























class GOATBench(DynamicBench):

    task = 'GOAT_BENCH'

    def setup_experiment(self, split, num_scenes):
        self.goat_data = []
        self.split = split
        self.sim_kwargs['scene_config'] =  f"scenes/hm3d/hm3d_annotated_{self.split}_basis.scene_dataset_config.json"
        if self.split == 'train':
            dir = 'train'
        else:
            dir = 'val_unseen'
        
        for f in os.listdir(f'goatBench/{dir}/content')[0:num_scenes]:
            with gzip.open(f'goatBench/{dir}/content/{f}', 'rt') as gz:
                self.goat_data.append(json.load(gz))

        random.shuffle(self.goat_data)


    def setup_run(self, history=7, mask_thinking=True, add_timesteps_prompt=True, draw_arrows=True,
            points=None, consistency=1, max_steps_per_goal=5):
        while True:
            goat_scene = random.choice(self.goat_data)
            episode = random.choice(goat_scene['episodes'])
            f, glb = episode['scene_id'].split('/')[-2:]
            
            if os.path.exists(f'logs/{self.outer_run_name}/{episode["episode_id"]}_{f[2:5]}'):
                continue
            break


        self.sim_kwargs['scene_id'] = f[2:5]
        self.sim_kwargs['scene_path'] = f'scenes/hm3d/{self.split}/{f}/{glb}'
        self.annotatedSimulator = AnnotatedSimulator(**self.sim_kwargs)

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
            target = {'name': name, 'mode': mode, 'objects': [], 'lang_desc': None}
            
            if mode == 'image':
                continue
            
            descriptions = goat_scene['goals'][f'{f[6:]}.basis.glb_{name}']

            for d in descriptions:

                if mode == 'object':
                    target['objects'].append({'object_id': d['object_id'], 'position': d['position']})
                else:
                    if d['object_id'] == goal[2]:
                        target['lang_desc'] = d['lang_desc']
                        target['position'] = d['position']

            # self.habitat_objects.append(habitat_obj)
            self.curr_episode.append(target)
        print(f'Running episode with {len(self.curr_episode)} goals')
        # self.annotatedSimulator.objects_to_annotate = self.habitat_objects                   
        self.curr_goal_ndx = 0
        self.curr_run_name = f"{episode['episode_id']}_{self.annotatedSimulator.scene_id}"
        self.last_goal_reset = -1
        goal = self.curr_episode[self.curr_goal_ndx]

        if goal['mode'] == 'object':
            print('Current goal:', goal['name'])
        if goal['mode'] == 'description':
            print('Current goal:', goal['lang_desc'])
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
            'max_steps_per_goal': max_steps_per_goal

        }

        obs = self.annotatedSimulator.step([('forward', 0)])
        return obs

    def step_env(self, obs):
        agent_state = self.get_agent_state()
        goal = self.curr_episode[self.curr_goal_ndx]

        if goal['mode'] == 'object':
            inst = f'Find the nearest {goal["name"]} and navigate to it.' 
        if goal['mode'] == 'description':
            inst = f"Find and navigate to the {goal['lang_desc']}. Think about which room you would find this {goal['name']} in."

        prompt = f"It is now timestep {self.step}. You have moved to a new location within the environment. "
        prompt += (
        # f"First, analyze your updated camera observation and tell me the spatial layout of what you see. "
        f"TASK: {inst}. "
        f"There may be some arrows superimposed onto the image, which represent potential actions. ")
        prompt += f"""In addition to any actions labeled on the image, you have the following special actions.
{{
0: turn completely around, use this when you dont see any good arrows, or IF THERE ARE NO ARROWS labeled on the image. 
-1: DONE, you have already navigated to the {goal['name']}!!
}}
Think which way you should go to reagh the {goal['name']} Then, select one action from the image or the special actions and explain how it helps you reach your goal. Return it as {{'action': <action_key>}}. Note you CANNOT go through CLOSED DOORS or through obstacles. 
"""
        if len(self.run_metadata['sensors']) > 1:
            num_sensors = len(self.annotatedSimulator.sensors)
            prompt = f"It is now timestep {self.step}. You have moved to a new location within the environment. "
            prompt += (
            f"First, tell me what you see in each of your sensor observations. Think about how the observations overlap, some objects will be visible in multiple sensors. "  
            f"There are arrows superimposed onto the {num_sensors} different images, which represent valid actions. "
            f"TASK: {inst} ")
            prompt += f"""In addition to any actions labeled on the images, you have the following special actions.
{{
0: turn completely around, use this when you DONT SEE ANY GOOD ACTIONS, and want fresh observations. 
-1: DONE, you are within 1 meter of the {goal['name']}{'. Make sure it matches the one in the description' if goal['mode'] == 'description' else ''}!!
}}
Think about how each action will move you. Then, select one action from the images or the special actions and explain how it helps you reach your goal. Return it as {{'action': <action_key>}}. Note you cannot go through closed doors or through obstacles"""
            
        images = self.get_sensor_images(obs)
        row = {'actions': -10, 'tokens_generated':0, 'success': 1, 'metadata': self.run_metadata,
        'speed': 0, 'scene_id': self.annotatedSimulator.scene_id, 'goal': goal['name'], 'goal_mode': goal['mode'], 'goal_index': self.curr_goal_ndx, 'curr_goal_steps': self.step - self.last_goal_reset,
        'model': self.vlm.name, 'input_tokens': 0, 'agent_location': agent_state.position}
        row, metadata, resp = self.agent_self_consitency(prompt, images, row, self.run_metadata['consistency'])

        if goal['mode'] == 'object':
            distances = [np.linalg.norm(agent_state.position - obj['position']) for obj in goal['objects']]
            min_dist = min(distances)
            row['distance_to_goal'] = min_dist
            metadata['DIST TO GOAL'] = row['distance_to_goal']
        if goal['mode'] == 'description':
            row['distance_to_goal'] = np.linalg.norm(agent_state.position - goal['position'])
            metadata['DIST TO GOAL'] = row['distance_to_goal']
        print('distance to goal', row['distance_to_goal'])
        copies = []
        for sensor in self.annotatedSimulator.sensors:
            copy = obs[f'color_sensor_{sensor}']['image'].copy()
            self.annotatedSimulator.draw_arrows(copy, agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], self.run_metadata['points'], chosen_action=row['actions'])
            # Assuming `copy` is the image you want to modify
            text = goal['name']
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.7
            thickness = 3
            text_color = (0, 0, 0)  # Black color
            background_color = (255, 255, 255)  # White color
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = 10
            text_y = 10 + text_size[1]  # Add the height of the text to the y position
            cv2.rectangle(copy, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), background_color, -1)
            cv2.putText(copy, text, (text_x, text_y), font, font_scale, text_color, thickness)

            if copy.shape[-1] == 4:
                copy = copy[:, :, 0:3]
            copies.append(Image.fromarray(copy, mode='RGB'))

        metadata['INST'] = inst
        done = False
        new_goal = False
        goal_reached = False
        if row['distance_to_goal'] < 3 and row['actions'] == -1:
            print(f"SUCESSFULLY FINISHED GOAL {self.curr_goal_ndx}")
            new_goal = True
            goal_reached = True
        elif self.step + 1 - self.last_goal_reset > self.run_metadata['max_steps_per_goal']:
            print('MAX STEPS PER GOAL REACHED')
            new_goal = True
            goal_reached = False
        elif row['distance_to_goal'] < 3:
            print('NEAR GOAL BUT MODEL DID NOT RETURN DONE')
        elif row['actions'] == -1:
            print('MODEL RETURNED DONE BUT NOT NEAR GOAL')
            
        if self.run_metadata['mask_thinking'] and row['success'] == 1:
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
                    print('New object:', goal['name'])
                if goal['mode'] == 'description':
                    print('New specific:', goal['lang_desc'])

        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        if self.step % self.log_freq == 0 or row['success'] == 0:
            self.log(prompt, images, resp, row['success'], metadata, copy_images=copies)
            
            
        if done:
            return None
        actions = self.annotatedSimulator.move_choices(row['actions'], points=self.run_metadata['points'] if self.run_metadata['arrows'] else None)        
        return actions
