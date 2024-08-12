import os
import sys
import pdb
import pickle
import random
from matplotlib import pyplot as plt
from networkx import draw
import numpy as np
import datetime
import cv2
import ast
import pandas as pd
from PIL import Image
from sympy import re
from src.utils import *
from src.vlmAgent import VLMAgent
from src.annoatedSimulator import AnnotatedSimulator
import seaborn as sns
import habitat_sim

class DynamicBench:

    def __init__(self, sim_kwargs, vlm_agent: VLMAgent, task_kwargs):

        self.annotatedSimulator = AnnotatedSimulator(**sim_kwargs)
        self.vlmAgent = vlm_agent

        self.df = pd.DataFrame({})
        self.task_kwargs = task_kwargs
        self.task = task_kwargs['task']
        assert self.task in ['obj_nav', 'image_nav', 'obj_distances', 'obj_height', 'obj_sizes']
        self.step = 0

    def parse_response(self, response):
        try:
            response_dict = ast.literal_eval(response[response.rindex('{'):response.rindex('}')+1])
        except (ValueError, SyntaxError):
            response_dict = {}
        return response_dict
    
    def simple_move(self, key, action, noise=True):

        if 'forward' in key:
            return ['forward', action]
        if 'backward' in key:
            return ['forward', -action]
        if 'left' in key:
            return ['rotate', np.pi*min(action, 180)/180]
        if 'right' in key:
            return ['rotate', -np.pi*min(action, 180)/180]


        action = random.sample(['left', 'right', 'forward', 'backward'], 1)[0]

        if 'left' in action:
            return ['rotate', np.pi/8]
        if 'right' in action:
            return ['rotate', -np.pi/8]
        if action == 'forward':
            return ['forward', 0.5]
        if action == 'backward':
            return ['forward', 0.5]
        if action == 'turn around':
            return ['rotate', np.pi]
        return ['forward', 0]


    # def setup_task(self):
    #     if self.task == 'obj_nav':
    #         all_objects = self.annotatedSimulator.get_all_objects(unique=True)
    #         bad_categories = ['floor', 'wall', 'ceiling', 'Unknown', 'unknown', 'surface', 'mirror']
    #         sa_max = 5
    #         sa_min = 0.1
    #         filtered = []
    #         for obj in all_objects:
    #             if obj.category.name() in bad_categories:
    #                 continue
            
    #             width = obj.aabb.sizes[0]
    #             height = obj.aabb.sizes[1]
    #             depth = obj.aabb.sizes[2]

    #             surface_area = 2 * (width * height + height * depth + depth * width)

    #             if surface_area > sa_max or surface_area < sa_min:
    #                 continue
    #             filtered.append(obj)

    #         target_obj = random.sample(filtered, 1)[0]
    #     return target_obj

    def log(self, prompt, obs, response, metadata):
        path = f'logs/{self.run_name}/step{self.step}'
        if metadata['success'] == 0:
            path += '_ERROR'
        os.makedirs(path)
        im_file = Image.fromarray(obs['image'][:, :, 0:3].astype('uint8'))
    
        im_file.save(f'{path}/image.png')
        with open(f'{path}/details.txt', 'w') as file:
            file.write(f'[PROMPT]\n{prompt}\n\n')
            file.write(f'[MODEL OUTPUT]\n{response}\n\n')

            if metadata['success']:
                for k, v in metadata['to_txt'].items():
                    file.write(f'{k}\n{v}\n\n')
    
    def _get_agent_state(self):
        return self.annotatedSimulator.sim.get_agent(0).get_state()
    
    def move_choices(self, action, points=None):
        if points is not None:
            action = int(action)
            if action == -1:
                print("DEFAULTING ACTION")
                return (['forward' , 0.2],)
            if action <= len(points):
                mag, theta = points[action-1]
                return (['rotate', -theta], ['forward', 1],)
            if action == 6:
                return (['rotate', np.pi],)
            if action == 7:
                return (['forward', -1.5],)
            if action == 8:
                print('MODEL THINKS DONE')
                return (['forward' , 0.2],)

        if action == 'small_forward':
            return (['forward', 1],)
        if action == 'big_forward':
            return (['forward', 2],)
        if action == 'backward':
            return (['forward', -2],)
        if action == 'turn_around':
            return (['rotate', np.pi],)
        if action == 'slight_right':
            return (['rotate', -np.pi/6], ['forward', 0.5])
        if action == 'hard_right':
            return (['rotate', -np.pi/3], ['forward', 0.5])
        if action == 'slight_left':
            return (['rotate', np.pi/6],['forward', 0.5])
        if action == 'hard_left':
            return (['rotate', np.pi/3],['forward', 0.5])
        
        print(f'action {action} not recognized')
        return (['forward', 0.2],)
    
    def run(self, inner_loop=40, log_freq = 10, history=7, 
            mask_thinking=True, add_timesteps_prompt=True, draw_arrows=True, 
            points=None, font_size=1.9, font_thickness=3, consistency=1, random_spawn=False, task='UPSTAIRS BEDROOM'):
        
        total_input_tokens = 0
        total_output_tokens = 0
        self.step = 0
        self.df = pd.DataFrame({})
        self.task_kwargs['obj'] = task
        self.metadata = {
            'task': task,
            'history': history,
            'points': points[0][0] if points else 0,
            'fontsize': font_size,
            'arrows': draw_arrows,
            'consistency': consistency,
            'mask_thinking': mask_thinking,
            'add_timesteps_prompt': add_timesteps_prompt
        }
            
        self.step = 0
        # context = self.setup_task()
        #context = {'category': lambda x: "bedroom"}
        if random_spawn:
            while True:
                obs = self.annotatedSimulator.step('r')
                init_state = self.annotatedSimulator.sim.get_agent(0).get_state().position
                print(init_state)
                if init_state[1] > -0.2:
                    self.task_kwargs['obj'] = 'DOWNSTAIRS KITCHEN'  
                    break
                elif init_state[1] < -2.8:
                    self.task_kwargs['obj'] = 'UPSTAIRS BEDROOM'
                    break
        else:
            init_state = habitat_sim.AgentState()
            init_state.position = np.array([ 0.7598896, -2.9753249, -0.6088147])
            init_state.rotation = quaternion.quaternion(0.70536607503891, 0, 0.708843231201172, 0)
            self.annotatedSimulator.sim.get_agent(0).set_state(init_state)

        self.metadata['task'] = self.task_kwargs['obj']
        self.run_name = "EXP_" + datetime.datetime.now().strftime("%m%d-%H%M%S") + "_" + self.vlmAgent.name
        print(f'\nSTARTING RUN: {self.run_name}\n')
        obs = {'image': self.annotatedSimulator.sim.get_sensor_observations(0)['color_sensor']}
        for step in range(inner_loop):
    
            if draw_arrows:
                self.annotatedSimulator.draw_arrows(obs['image'], font_scale=font_size, font_thickness=font_thickness, points=points)
            prompt = f"It is now timestep {step}. You have moved to a new location within the environment. "
            prompt += (
            f"First, analyze your updated camera observation and tell me the spatial layout of what you see. "
            f"{'There are five arrows superimposed onto the image, which represent your possible actions. ' if draw_arrows else ''}"
            f"Your task is to navigate to the {self.task_kwargs['obj']}. Think of a high level plan on how you can reach the {self.task_kwargs['obj']} from where you are now. ")

            if draw_arrows:
                prompt += f"""The following describes your 8 possible actions in more detail, 5 of which are already labeled in the image.
{{
1: strong left
2: slight left
3: straight ahead
4: slight right
5: strong right
6: turn completely around, use this when you are facing a wall and cannot see the room
7: move backward, still facing in the same direction
8: done, you have completed your task
}}
Think about how each action will move you. Then, select one action and explain how it helps you reach your goal. Return it as {{'action': <action_key>}}. Note you cannot go through closed doors or through obstacles. 
"""
            else:
                prompt += f"""The following describes your possible actions: \n{{
'small_forward': Move forward by 1 meter,
'big_forward': Move forward by 2 meters
'backward': Move backward by 2 meter
'slight_right': turn right by 30 degrees, and then forward 0.5 meters
'hard_right': turn right by 60 degrees, and then forward 0.5 meters
'slight_left': turn left by 30 degrees, and then forward 0.5 meters
'hard_left': turn left by 60 degrees, and then forward 0.5 meters
'turn_around': turns a full 180 degrees. USE THIS WHEN YOU ARE FACING A WALL AND CANNOT SEE THE ROOM
'done': you have completed your task
}} \nSelect one of the nine action keys to help you continue your task to navigate to the {self.task_kwargs['obj']}. Return it as {{'action': <action_key>}}"""
            action_counter = {}
            num_calls = 0
            while True:
                num_calls += 1
                resp, performance = self.vlmAgent.call_chat(history, obs['image'], prompt, add_timesteps_prompt=add_timesteps_prompt, step=step)
                total_input_tokens += performance['input_tokens']
                #print(f'input_tokens: {perf["input_tokens"]}')
                total_output_tokens += performance['tokens_generated']
                # resp, perf = self.vlmAgent.call(obs['image'], prompt)
                # if len(self.history) > 0:
                    # resp, perf = self.vlmAgent.call_multi_image(convo, [first_obs, self.history[0][0], obs['image']])
                # else: 
                #resp, perf = self.vlmAgent.call_multi_image(conversation, [first_obs, obs['image']])
                metadata = {'success': 1, 'to_txt': {}}
                row = {'actions': -1, 'tokens_generated':performance['tokens_generated'], 'success': 1, 'metadata': self.metadata,
                        'speed': performance['tokens_generated']/performance['duration'], 'scene_id': self.annotatedSimulator.scene_id,
                        'model': self.vlmAgent.name, 'input_tokens': performance['input_tokens'], 'agent_location': self._get_agent_state().position,
                        'num_calls': num_calls}
                try:
                    resp_dict = self.parse_response(resp)
                    row['actions'] = resp_dict['action']
                    print(f'chose action {row["actions"]}')

                except (IndexError, KeyError) as e:
                    print(e)
                    row['success'] = 0
                    metadata['success'] = 0
                
                finally:
                    metadata['to_txt']['ACTIONS'] = row['actions']
                    metadata['actions'] = row['actions']

                    if self.task == 'obj_nav':
                        row['goal_object'] = self.task_kwargs['obj']
                        if row['goal_object'] == 'FIREPLACE':
                            row['distance_to_goal'] = np.linalg.norm(self._get_agent_state().position - np.array([-7.681808,  -2.9753249,  0.2911849]))
                        
                        if row['goal_object'] == 'DOWNSTAIRS KITCHEN':
                            row['distance_to_goal'] = abs(self._get_agent_state().position[1]- -2.97)
                        elif row['goal_object'] == 'UPSTAIRS BEDROOM':
                            row['distance_to_goal'] = abs(self._get_agent_state().position[1]- 0.02)
                    
                        metadata['to_txt']['DIST TO GOAL'] = row['distance_to_goal']
                if row['actions'] in action_counter:
                    action_counter[row['actions']]+= 1
                else:
                    action_counter[row['actions']] = 1
                
                if action_counter[row['actions']] == consistency:
                    break
                else:
                    self.vlmAgent.session.rewind()

            
            self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
            
            if mask_thinking:
                self.vlmAgent.session.history[-1].parts[0].text = f'\n' + '{"action": ' + str(row["actions"]) + '}'
            if step % log_freq == 0 or metadata['success'] == 0:
                self.log(prompt, obs, resp, metadata)
            # if metadata['actions'] == 'done':
            #     break

            for a1, a2 in self.move_choices(metadata['actions'], points=points if draw_arrows else None):
                obs = self.annotatedSimulator.move(a1, a2)
                #   obs = self.annotatedSimulator.move(*self.simple_move(k, metadata['actions'][k], noise=True))
                # self.annotatedSimulator.move(k, metadata['actions'][k])
            self.step += 1

        self.df.to_pickle(f'logs/{self.run_name}/df_results.pkl')
        self.vlmAgent.reset()
        
        print('complete')
        print(f'GPT Mini would cost: {total_input_tokens*0.15/1000000 + total_output_tokens*0.6/1000000}')
        print(f'GPT 4o would cost: {total_input_tokens*5/1000000 + total_output_tokens*15/1000000}')
        print(f'Gemini 1.5pro would cost: {total_input_tokens*3.5/1000000 + total_output_tokens*10.50/1000000}')
        print(f'Gemini flash would cost: {total_input_tokens*0.35/1000000 + total_output_tokens*0.150/1000000}')

        #create_stop_motion_video(f'logs/{self.run_name}', f'logs/{self.run_name}/video.mp4')