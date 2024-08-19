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
from src.vlmAgent import VLMAgent
from src.annoatedSimulator import AnnotatedSimulator
import habitat_sim

class DynamicBench: 

    task = 'Not defined'

    def __init__(self, sim_kwargs, vlm_agent: VLMAgent, task_kwargs, outer_run_name):

        self.annotatedSimulator = AnnotatedSimulator(**sim_kwargs)

        self.vlmAgent = vlm_agent
        self.init_pos = None
        self.task_kwargs = task_kwargs

        self.df = pd.DataFrame({})
        self.random_seed = sim_kwargs['random_seed']
        self.outer_run_name = self.task + '_' + outer_run_name
        self.step = 0
        self.curr_run_name = "Not started"
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.setup_task()

    def setup_task(self):
        raise NotImplementedError

    def parse_response(self, response):
        try:
            response_dict = ast.literal_eval(response[response.rindex('{'):response.rindex('}')+1])
        except (ValueError, SyntaxError):
            response_dict = {}
        return response_dict
    
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

    def get_agent_state(self):
        return self.annotatedSimulator.sim.get_agent(0).get_state()
    
    def set_agent_state(self, pos=None, quat=None):
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
            resp, performance = self.vlmAgent.call_chat(self.run_metadata['history'], images, prompt, add_timesteps_prompt=self.run_metadata['add_timesteps_prompt'], step=self.step)
            self.total_input_tokens += performance['input_tokens']
            self.total_output_tokens += performance['tokens_generated']

            metadata = {}
            row = original_row.copy()
            try:
                resp_dict = self.parse_response(resp)
                row['actions'] = resp_dict['action']

            except (IndexError, KeyError) as e:
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
                    self.vlmAgent.session.rewind()
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

    def run(self, inner_loop, log_freq, **run_kwargs):
        self.step = 0
        self.df = pd.DataFrame({})
        self.log_freq = log_freq
        obs = self.setup_run(**run_kwargs)
        print(f'\n===================STARTING RUN: {self.curr_run_name}===================\n')

        for _ in range(inner_loop):
            actions = self.step_env(obs)
            if actions is None:
                break
            obs = self.annotatedSimulator.step(actions)
            self.step += 1

        self.post_run()
    
    def step_env(self, obs):
        raise NotImplementedError

    def setup_run(self, run_kwargs):
        raise NotImplementedError

    def post_run(self):
        print('\n===================RUN COMPLETE===================\n')
        self.df.to_pickle(f'logs/{self.outer_run_name}/{self.curr_run_name}/df_results.pkl')
        gif(f'logs/{self.outer_run_name}/{self.curr_run_name}')
        self.vlmAgent.reset()

    def get_costs(self):
        print(f'GPT Mini would cost: {np.round(self.total_input_tokens*0.15/1000000 + self.total_output_tokens*0.6/1000000, 2)}')
        print(f'GPT 4o would cost: {np.round(self.total_input_tokens*5/1000000 + self.total_output_tokens*15/1000000, 2)}')
        print(f'Gemini 1.5pro would cost: {np.round(self.total_input_tokens*3.5/1000000 + self.total_output_tokens*10.50/1000000, 2)}')
        print(f'Gemini flash would cost: {np.round(self.total_input_tokens*0.35/1000000 + self.total_output_tokens*0.150/1000000, 2)}')
        






class NavBench(DynamicBench):

    task = 'obj_nav'
    default_quat = quaternion.quaternion(0.70536607503891, 0, 0.708843231201172, 0)

    def setup_task(self):

        random.shuffle(self.task_kwargs['goals'])             
        for i in range(4):
            target, related = self.task_kwargs['goals'][i]
            if os.path.exists(f'logs/{self.outer_run_name}/{target}_{self.annotatedSimulator.scene_id}'):
                print(f'{target}_{self.annotatedSimulator.scene_id} already exists')
                continue
            self.task_kwargs['obj_name'] = target
            self.related_objects = []
            for word in related + [target]:
                self.related_objects += self.annotatedSimulator.search_objects(word, exact=False)
            print(f'Targeting object: {target}')
            print(f'related objects: {len([obj.category.name() for obj in self.related_objects])}')
            if len(self.related_objects) == 0:
                continue
            for _ in range(200):
                point = self.annotatedSimulator.sim.pathfinder.get_random_navigable_point()
                for idx, floor_height in enumerate(self.annotatedSimulator.floors):
                    if abs(point[1] - floor_height) < 0.1:
                        floor = idx
                        distances = [np.linalg.norm(point - obj.aabb.center) for obj in self.related_objects if obj.aabb.center[1] < self.annotatedSimulator.floors[floor+1] and obj.aabb.center[1] > floor_height]
                        min_dist = 7 if target in ['kitchen', 'living room'] else 5.5
                        if len(distances) > 0 and min(distances) > min_dist and min(distances) < min_dist + 8:
                            self.init_pos = point
                            break
                if self.init_pos is not None:
                    break
            if self.init_pos is not None:
                break
            print('sampling again')

        if self.init_pos is None:
            self.annotatedSimulator.sim.close()
            raise SystemError(f'Scene id {self.annotatedSimulator.scene_id} Could not find a valid starting position')
    
    def setup_run(self, history=7, 
            mask_thinking=True, add_timesteps_prompt=True, draw_arrows=True,
            points=None, consistency=1):

        self.run_metadata = {
            'task': self.task_kwargs['obj_name'],
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
            'init_pos': self.init_pos
        }

        self.annotatedSimulator.do_draw_arrows = points if draw_arrows else None
        self.annotatedSimulator.do_annotate_image = False
        self.annotatedSimulator.objects_to_annotate = self.related_objects
        self.step = 0
        self.set_agent_state() 

        self.curr_run_name = f'{self.task_kwargs["obj_name"]}_{self.annotatedSimulator.scene_id}'
        obs = self.annotatedSimulator.step([('forward', 0)])
        return obs

    def step_env(self, obs):
        agent_state = self.get_agent_state()
            
        prompt = f"It is now timestep {self.step}. You have moved to a new location within the environment. "
        prompt += (
        f"First, analyze your updated camera observation and tell me the spatial layout of what you see. "
        f"There may be some arrows superimposed onto the image, which represent valid actions. "
        f"Your task is to navigate to a {self.task_kwargs['obj_name'].upper()}. Think of a high level plan on how you can reach a {self.task_kwargs['obj_name'].upper()} from where you are now. ")
        prompt += f"""In addition to any actions labeled on the image, you have the following special actions.
{{
0: turn completely around, use this when you are in a dead end, or IF THERE ARE NO ARROWS labeled on the image. 
-1: DONE, you think you have reached the {self.task_kwargs['obj_name'].upper()}!!
}}
Think about how each action will move you. Then, select one action from the image or the special actions and explain how it helps you reach your goal. Return it as {{'action': <action_key>}}. Note you cannot go through closed doors or through obstacles. 
"""
        if len(self.run_metadata['sensors']) > 1:
            num_sensors = len(self.annotatedSimulator.sensors)
            prompt = f"It is now timestep {self.step}. You have moved to a new location within the environment. "
            prompt += (
            f"First, tell me what you see in each of your sensor observations. Think about how the observations overlap, some objects will be visible in multiple sensors. "  
            f"There are arrows superimposed onto the {num_sensors} different images, which represent valid actions. "
            f"Your task to navigate to a {self.task_kwargs['obj_name'].upper()}. Think of a high level plan on how you can reach a {self.task_kwargs['obj_name'].upper()} from where you are now. ")
            prompt += f"""In addition to any actions labeled on the images, you have the following special actions.
{{
0: turn completely around, use this when you DONT SEE ANY GOOD ACTIONS, and want fresh observations. 
-1: DONE, you think you have reached the {self.task_kwargs['obj_name'].upper()}!!
}}
Think about how each action will move you. Then, select one action from the images or the special actions and explain how it helps you reach your goal. Return it as {{'action': <action_key>}}. Note you cannot go through closed doors or through obstacles. 
"""
        images = self.get_sensor_images(obs)
        row = {'actions': -10, 'tokens_generated':0, 'success': 1, 'metadata': self.run_metadata,
        'speed': 0, 'scene_id': self.annotatedSimulator.scene_id,
        'model': self.vlmAgent.name, 'input_tokens': 0, 'agent_location': agent_state.position}
        row, metadata, resp = self.agent_self_consitency(prompt, images, row, self.run_metadata['consistency'])

        row['goal_object'] = self.task_kwargs['obj_name']
            
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
            self.annotatedSimulator.draw_arrows(copy, agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], 
                                                        self.run_metadata['points'], chosen_action=row['actions'])
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
            self.vlmAgent.session.history[-1].parts[0].text = f'\n' + '{"action": ' + str(row["actions"]) + '}'
        if self.step % self.log_freq == 0 or row['success'] == 0:
            self.log(prompt, images, resp, row['success'], metadata, copy_images=copies)

        if self.step >= 3 and self.df['actions'].iloc[-3:].tolist().count(-1) >= 2:
            print("STOPPING EARLY, DONE")
            return None

        actions = self.annotatedSimulator.move_choices(row['actions'], points=self.run_metadata['points'] if self.run_metadata['arrows'] else None)        
        return actions

#     def run(self, inner_loop, log_freq, history=7, 
#             mask_thinking=True, add_timesteps_prompt=True, draw_arrows=True,
#             points=None, consistency=1):
        
#         total_input_tokens = 0
#         total_output_tokens = 0

#         self.step = 0
#         self.df = pd.DataFrame({})
#         self.run_metadata = {
#             'task': self.task_kwargs['obj_name'],
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
#             'init_pos': self.init_pos
#         }

#         self.set_agent_state() 

#         self.curr_run_name = f'{self.task_kwargs["obj_name"]}_{self.annotatedSimulator.scene_id}'
#         print(f'\nSTARTING RUN: {self.curr_run_name}\n')
#         obs = self.annotatedSimulator.step([('forward', 0)], draw_arrows=points if draw_arrows else [], objects_to_annotate=[])
#         multi = len(self.annotatedSimulator.sensors) > 1
# # 1: strong left
# # 2: slight left
# # 3: straight ahead
# # 4: slight right
# # 5: strong right
#         for _ in range(inner_loop):
#             agent_state = self.get_agent_state()
            
#             prompt = f"It is now timestep {self.step}. You have moved to a new location within the environment. "
#             prompt += (
#             f"First, analyze your updated camera observation and tell me the spatial layout of what you see. "
#             f"There may be some arrows superimposed onto the image, which represent valid actions. "
#             f"Your task is to navigate to a {self.task_kwargs['obj_name'].upper()}. Think of a high level plan on how you can reach a {self.task_kwargs['obj_name'].upper()} from where you are now. ")
#             prompt += f"""In addition to any actions labeled on the image, you have the following special actions.
# {{
# 0: turn completely around, use this when you are in a dead end, or IF THERE ARE NO ARROWS labeled on the image. 
# -1: DONE, you think you have reached the {self.task_kwargs['obj_name'].upper()}!!
# }}
# Think about how each action will move you. Then, select one action from the image or the special actions and explain how it helps you reach your goal. Return it as {{'action': <action_key>}}. Note you cannot go through closed doors or through obstacles. 
# """
#             if multi:
#                 num_sensors = len(self.annotatedSimulator.sensors)
#                 prompt = f"It is now timestep {self.step}. You have moved to a new location within the environment. "
#                 prompt += (
#                 f"First, tell me what you see in each of your sensor observations. Think about how the observations overlap, some objects will be visible in multiple sensors. "  
#                 f"There are arrows superimposed onto the {num_sensors} different images, which represent valid actions. "
#                 f"Your task to navigate to a {self.task_kwargs['obj_name'].upper()}. Think of a high level plan on how you can reach a {self.task_kwargs['obj_name'].upper()} from where you are now. ")
#                 prompt += f"""In addition to any actions labeled on the images, you have the following special actions.
# {{
# 0: turn completely around, use this when you DONT SEE ANY GOOD ACTIONS, and want fresh observations. 
# -1: DONE, you think you have reached the {self.task_kwargs['obj_name'].upper()}!!
# }}
# Think about how each action will move you. Then, select one action from the images or the special actions and explain how it helps you reach your goal. Return it as {{'action': <action_key>}}. Note you cannot go through closed doors or through obstacles. 
# """
#             images = self.get_sensor_images(obs)

#             action_counter = {}
#             num_calls = 0
#             while True:
#                 num_calls += 1
#                 resp, performance = self.vlmAgent.call_chat(history, images, prompt, add_timesteps_prompt=add_timesteps_prompt, step=step)
#                 total_input_tokens += performance['input_tokens']
#                 total_output_tokens += performance['tokens_generated']

#                 metadata = {}
#                 row = {'actions': -10, 'tokens_generated':performance['tokens_generated'], 'success': 1, 'metadata': self.run_metadata,
#                         'speed': performance['tokens_generated']/performance['duration'], 'scene_id': self.annotatedSimulator.scene_id,
#                         'model': self.vlmAgent.name, 'input_tokens': performance['input_tokens'], 'agent_location': agent_state.position,
#                         'num_calls': num_calls}
#                 try:
#                     resp_dict = self.parse_response(resp)
#                     row['actions'] = resp_dict['action']

#                 except (IndexError, KeyError) as e:
#                     print(e)
#                     row['success'] = 0
                
#                 finally:
#                     metadata['ACTIONS'] = row['actions']


#                 if row['actions'] in action_counter:
#                     action_counter[row['actions']]+= 1
#                 else:
#                     action_counter[row['actions']] = 1
                
#                 if action_counter[row['actions']] == consistency:
#                     print(f'Stepping, took {num_calls} calls')
#                     break
#                 else:
#                     if row['success']==1:
#                         self.vlmAgent.session.rewind()

#             objects_to_annotate = []
#             if self.task == 'obj_nav':
#                 row['goal_object'] = self.task_kwargs['obj_name']
#                 objects_to_annotate = self.related_objects
                
#             min_dist = 1000
#             closest_object = None

#             copies = []
#             for sensor in self.annotatedSimulator.sensors:
#                 annotations = obs[f'color_sensor_{sensor}']['annotations']
#                 for obj in annotations:
#                     dist = np.linalg.norm(obj['curr_local_coords'])
#                     print('object', obj['obj'], 'distance', dist)
#                     if dist < min_dist:
#                         min_dist = dist
#                         closest_object = obj['obj']

#                 copy = obs[f'color_sensor_{sensor}']['image'].copy()
#                 self.annotatedSimulator.draw_arrows(copy, agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], 
#                                                             points, chosen_action=row['actions'])
#                 if copy.shape[-1] == 4:
#                     copy = copy[:, :, 0:3]
#                 copies.append(Image.fromarray(copy, mode='RGB'))
#             row['closest_object'] = closest_object
#             row['distance_to_goal'] = min_dist
#             metadata['DIST TO GOAL'] = row['distance_to_goal']
#         #if last three actions are -1, break
#             # distances = []
#             # for i in range(len(self.df)-4, len(self.df)):
#             #     agent_state = self.df['agent_location'].iloc[i]
#             #     distance = np.linalg.norm(agent_state - self.df['agent_location'].iloc[i-1])
#             #     distances.append(distance)
#             # average_distance = sum(distances) / len(distances)
#             # print(f"Average distance between agent states in the last 4 steps: {average_distance}")

#             self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
#             if mask_thinking and row['success'] == 1:
#                 self.vlmAgent.session.history[-1].parts[0].text = f'\n' + '{"action": ' + str(row["actions"]) + '}'
#             if self.step % log_freq == 0 or row['success'] == 0:
#                 self.log(prompt, images, resp, row['success'], metadata, copy_images=copies)

#             if self.step >= 3 and self.df['actions'].iloc[-3:].tolist().count(-1) >= 2:
#                 print("STOPPING EARLY, DONE")
#                 break

#             actions = self.annotatedSimulator.move_choices(metadata['actions'], points=points if draw_arrows else None)
#             obs = self.annotatedSimulator.step(actions, draw_arrows=points if draw_arrows else [], 
#                                                annotate_image=False, objects_to_annotate=objects_to_annotate)


#         self.post_run() 