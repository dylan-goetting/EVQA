
import csv
import gzip
import json
import math
import os
import pdb
import random
from turtle import update
import numpy as np
import pandas as pd
from PIL import Image
from src.utils import *
from src.vlm import VLM, GPTModel, GeminiModel
from src.annoatedSimulator import AnnotatedSimulator
from src.dynamicBench import DynamicBench
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis
import concurrent.futures

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
            points=None, consistency=1, goals = [], priv_actions=False, uniform=True, use_map=True, map_type='unpriv',
            explore_factor = 0):
        
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
            'map_model': self.map_vlm.name,
            'model': self.vlm.name,
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
            'explore_factor': explore_factor,
            'map_type': map_type
        }
        self.annotatedSimulator.priv_actions = False
        self.annotatedSimulator.do_annotate_image = False
        self.annotatedSimulator.objects_to_annotate = self.curr_related_objects
        self.set_state()

        self.curr_run_name = f'{self.curr_target}_{self.annotatedSimulator.scene_id}_{random.randint(0, 1000)}'
        obs = self.annotatedSimulator.step([('forward', 0)])
        return obs


    def step_env(self, obs):
        agent_state = self.get_agent_state()
        real_actions = self.annotate_image(agent_state, obs)
        zoomed_map = self.generate_unpriv(real_actions, zoom=9) if self.run_metadata['map_type'] == 'unpriv' else self.generate_topdown(real_actions, zoom=9)


        multi = len(self.annotatedSimulator.sensors) > 1
        prompt = (
        # f"First, analyze your updated camera observation and tell me the spatial layout of what you see. "
        f"I have lost my diamond ring! Your task is to search every inch of this floor to help me find it. It is a huge, bright diamond that is unmistakable. "
        f"There are {len(real_actions) - 1} red arrows superimposed onto your observation{'s' if multi else''}, which represent potential actions. " 
        f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'Note that special action 0 turns you around completely' if self.step - self.turned >= 3 else ''}"
        #f"Your task is to navigate to the {self.curr_target.upper()}. Think of a high level plan on how you can reach the {self.curr_target.upper()} from where you are now. If you have already reached the {self.curr_target.upper()} choose special action -1 (done). "
        f"First, tell me what you see in your sensor observations. Then, tell me a high level plan on how you will find the ring and where you will go next. Recall your past observations so that you dont waste time exploring the same locations"
        f"Think about how each action will move you. Lastly, select one action from the image and explain how it helps you reach your goal. Return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS"
        )

        images = self.get_sensor_images(obs, convert=False)
        if self.run_metadata['use_map'] > 0:
            prompt = (

            # f"Your task is to navigate to a {self.curr_target.upper()}, and you have {len(real_actions) - 1} actions available to you. "
            # "You have two different sources of information that show these actions: \n1. your RGB sensors show your current view of the environment, and the actions are superimposed onto the images as red arrows to white circles. The white cicles show the exactly where the action will move you. "
            # "\n2. you have a topdown map of the environment, with navigable area shown in light grey and obstacles shown in black. This map shows the trajectory of where you have been in the past, shown GREEN. Your current location is shown by a RED dot. "
            # "The same actions you see superimposed on the RGB image are also shown on the top-down map. These actions also represented by red arrows and white circles, and show the location you would move to if you took that action. "
            # f"Carefully analyze this map and make sure you understand what it means. {'Remember you have action 0 for when you are in a dead end or want to turn around' if self.step - self.turned >= 3 else ''}"
            # f"\n\nFirst, tell me what you see in your sensor observations, and which way you should go to reach your goal. Second, describe the map and think about which actions leave your curent room. Combine both sources of information to make an informed decision on what action to take. "
            # f"If you have already comleted your goal choose special action -1. Return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS "
            # # Think of a high level plan on how you can reach a {self.curr_target.upper()} from where you are now. ")
            # )
            f"I have lost my diamond ring! Your task is to search every inch of this floor to help me find it. It is a huge, bright diamond that is unmistakable. "

            # f"Your task is to navigate to the {self.curr_target.upper()}, and "you have {len(real_actions) - 1} actions available to you. "
            # f"There are {len(real_actions) - 1} red arrows superimposed onto your observation{'s' if multi else''}, which show where you would move to if you chose that action number ." 
            "\nTo help you with exploration, you have a topdown map of the environment, with navigable area shown in LIGHT GREY and obstacles shown in BLACK. This map shows where you have been in the past, in GREEN. Your current location is shown by a RED dot. "
            f"You have {len(real_actions) - 1} actions, which are red arrows and white circles, and show the location you would move to if you took that action number. Use the map to identify unexplored rooms (light grey) and plan out actions that help you reach these unexplored areas. Note you will sometimes need to backtrack through green areas to reach new rooms. "
            # "If you have already reached the {self.curr_target.upper()} choose special action -1 (done). "
            f'{"First, describe the map you see, as well as your sensor observations. Tell me a high level plan on how you will find the ring and where you will go next. " if self.run_metadata["use_map"] != 2 else ""}'
            # "Use your map to strategically think about which actions will you get to new territory, and remember the light grey areas are navigable and the black areas are not "    
            f'Lastly, use {"both the map and your sensor observations" if self.run_metadata["use_map"] != 2 else "the map"} to select the best action and explain how it helps you reach your goal. '
            f"Return it as {{'action': <action_number>}}"
            # Think of a high level plan on how you can reach a {self.curr_target.upper()} from where you are now. ")
            )
            images.append(zoomed_map)
            if self.run_metadata['use_map'] == 2:
                self.vlm = self.map_vlm
                images = [zoomed_map]
        
        row = {'actions': -10, 'tokens_generated':0, 'success': 1, 'metadata': self.run_metadata,
        'speed': 0, 'scene_id': self.annotatedSimulator.scene_id,
        'model': self.vlm.name, 'input_tokens': 0, 'agent_location': agent_state.position}
        row, metadata, resp = self.agent_self_consitency(prompt, images, row, self.run_metadata['consistency'])
        row['goal_object'] = self.curr_target

        rgb = self.topdown_map
        green = np.sum(rgb == self.explored_color)
        light_grey = np.sum(rgb == self.unexplored_color)
        row['explored'] = green / (green + light_grey) 

        min_dist = 1000
        closest_object = None

        if row["actions"] <= len(list(real_actions.keys())) and row["actions"] > 0:
            mag, theta = list(real_actions.keys())[row["actions"]-1]
            self.update_unpriv(mag, theta, agent_state, mag, clip_frac=1)
            self.update_topdown(mag, theta, agent_state, mag, clip_frac=1)
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
            self.draw_arrows(real_actions.keys(), copy, agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], chosen_action=row['actions'], real_actions=real_actions)
            copies.append(copy)
        copies.append(self.topdown_map)
        row['closest_object'] = closest_object
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
            points=None, consistency=1, max_steps_per_goal=5, priv_actions=False, use_map=0, uniform=False, 
            explore_factor=0, map_type='priv', success_thresh=2.5):
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

        self.init_pos = np.array(episode['start_position'])
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
        for i, obj in enumerate(self.curr_episode):
            print(f'Goal {i}: {obj["name"]}, {obj["mode"]}')
        # self.annotatedSimulator.objects_to_annotate = self.habitat_objects                   
        self.curr_goal_ndx = 0
        self.curr_run_name = f"{episode['episode_id']}_{self.annotatedSimulator.scene_id}"
        self.last_goal_reset = -1
        goal = self.curr_episode[self.curr_goal_ndx]

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
            'uniform': uniform,
            'explore_factor': explore_factor,
            'map_type': map_type,
            'success_thresh': success_thresh
        }
        if goal['mode'] == 'object':
            print('Current general object:', goal['name'], f'there are {len(goal["objects"])} instances')

        if goal['mode'] == 'description':
            print('Current desc:', goal['lang_desc'])
        if goal['mode'] == 'image':
            print('Current image:', goal['name'])

        obs = self.annotatedSimulator.step([('forward', 0)])
        return obs

    def step_env(self, obs):
        goal = self.curr_episode[self.curr_goal_ndx]
        if self.step == 0 and goal['mode'] != 'object' and self.run_metadata['map_type'] == 'unpriv':
            self.update_topdown(0, 0, None, 0, clip_frac=1, goal=goal['position'], goal_name=goal['name'])


        if goal['mode'] == 'object':
            inst = f'Find the nearest {goal["name"]} and navigate to it. '
            inst2 =  f'Tell me which room you would find this {goal["name"]} in? Do you see any {goal["name"]} in your current observations?' 
        if goal['mode'] == 'description':
            inst = f"Find and navigate to the {goal['lang_desc']} "
            inst2 =  f"Tell me which room you would find this specifc {goal['name']} in, and in which direction you should go. "
        if goal['mode'] == 'image':
            inst = f"Observe the image labeled GOAL IMAGE. Find this specific {goal['name']} shown in the image and navigate to it. "
            inst2 =  f"Tell me which room you would find this {goal['name']} in, and which in which direction you should go ."

        agent_state = self.get_agent_state()
        real_actions = self.annotate_image(agent_state, obs)

        multi = len(self.annotatedSimulator.sensors) > 1

        zoomed_map = self.generate_unpriv(real_actions, zoom=9) if self.run_metadata['map_type'] == 'unpriv' else self.generate_topdown(real_actions, zoom=9)
        prompt = (

        f"TASK: {inst} "
        f"There are {len(real_actions) - 1} red arrows superimposed onto your observation{'s' if multi else''}, which represent potential actions. " 
        f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or are at a dead end .' if self.step - self.turned >= 3 else ''}"
        f"First, tell me what you see in your sensor observations, and if you have any leads on finding the {goal['name']}. Second, {inst2}. If you have already reached the {goal['name']} choose special action -1 (YOU HAVE SUCEEDED). "
        f"Lastly, explain which action is the best and Return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS"
        )

        images = self.get_sensor_images(obs, convert=False)
        if self.run_metadata['use_map']:
            prompt = (
        f"TASK: {inst} "
        f"There are {len(real_actions) - 1} red arrows superimposed onto your observation{'s' if multi else''}, which represent potential actions. " 
        f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or at a dead end .' if self.step - self.turned >= 3 else ''}"
        "\nYou ALSO have a topdown map of the environment, with unexplored area shown in LIGHT GREY and obstacles shown in BLACK. This map shows you where you have been in the past, shown in GREEN. Your current location is shown by a RED dot. "
        "The same actions you see superimposed on the RGB image are also shown on the top-down map. These actions also represented by red arrows and white circles, and show the location you would move to if you took that action. "
        "Use this map to help you explore new areas (light grey). "
        f"First, tell me what you see in your sensor observations, what the map is telling you, and if you have any leads on finding the {goal['name']}. Second, {inst2}. If you have already reached the {goal['name']} choose special action -1 (YOU HAVE SUCEEDED). "
        "If you are not sure where to go, use the map to help you explore. "
        f"Lastly, explain which action is the best and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS"
            )
            # self.vlm = self.map_vlm
            images.append(zoomed_map)       

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

        if row["actions"] <= len(list(real_actions.keys())) and row["actions"] > 0:
            mag, theta = list(real_actions.keys())[row["actions"]-1]
            self.update_unpriv(mag, theta, agent_state, mag, clip_frac=1)
            self.update_topdown(mag, theta, agent_state, mag, clip_frac=1)

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
        if row['distance_to_goal'] < self.run_metadata['success_thresh'] and row['actions'] == -1:
            print(f"SUCESSFULLY FINISHED GOAL {self.curr_goal_ndx}")
            new_goal = True
            goal_reached = True
        elif self.step + 1 - self.last_goal_reset > self.run_metadata['max_steps_per_goal']:
            print('MAX STEPS PER GOAL REACHED')
            new_goal = True
            goal_reached = False
        elif row['distance_to_goal'] < self.run_metadata['success_thresh']:
            print('NEAR GOAL BUT MODEL DID NOT RETURN DONE')
        elif row['actions'] == -1:
            print('MODEL RETURNED DONE BUT NOT NEAR GOAL')
            
        copies = []
        for sensor in self.annotatedSimulator.sensors:
            copy = obs[f'color_sensor_{sensor}']['image'].copy()
            self.draw_arrows(real_actions, copy, agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], chosen_action=row['actions'], real_actions=real_actions)
            if new_goal and goal_reached:
                background_color = (0, 100, 0) 
            elif new_goal:
                background_color = (100, 0, 0) 
            else:
                background_color = (255, 255, 255)  
            put_text_on_image(copy, f"{self.curr_goal_ndx}: {goal['name']}({goal['mode'][0]})", background_color=background_color, location='top_left', text_size=2.3)
            copies.append(copy)

        copies.append(self.topdown_map)

        if self.run_metadata['mask_thinking'] and row['success'] == 1 and self.run_metadata['history'] > 0:
            self.vlm.session.history[-1].parts[0].text = f'\n' + '{"action": ' + str(row["actions"]) + '}'
        row['goal_reached'] = goal_reached
        row['new_goal'] = new_goal
        if new_goal:
            self.explored_map = np.zeros_like(self.explored_map)
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
                elif self.run_metadata['map_type'] == 'unpriv':
                    self.update_topdown(0, 0, agent_state, 0, clip_frac=1, goal=goal['position'], goal_name=goal['name'])
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
            points=None, consistency=1, max_steps_per_goal=5, uniform=False, use_map=True, explore_factor=1.5):
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
            'question_type': question_data["label"],
            'explore_factor': explore_factor,
            }  

            self.annotatedSimulator.priv_actions = False
            self.annotatedSimulator.do_draw_arrows = False #points if draw_arrows else None
            self.annotatedSimulator.do_annotate_image = False
            self.annotatedSimulator.objects_to_annotate = []
            self.init_pos = np.array(init_pts)
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
            answer_prompt = (f"The agent has sent you an image, and is asking you the following question: [QUESTION]: {vlm_question+extra}\n\nPay close attention to the specific details in the question, note the agent may not be in the right location. First, tell me where you are in the environmemt, and if there are any notable objects that are relevant to the question. Then, explain what the best answer choice is and why. Lastly, return it as a JSON like {{'answer': <answer letter>}}")
            res = None

            def process_image(image):
                r, p = self.answerVLM.call([image], answer_prompt, logprobs=5)
                dct = self.parse_response(r)
                if 'answer' in dct and dct['answer'] in ['A', 'B', 'C', 'D']:
                    return dct['answer'], r
                return 'E', r

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(process_image, image): image for image in raw_images}

                for future in concurrent.futures.as_completed(futures):
                    answer, r = future.result()
                    if answer in ['A', 'B', 'C', 'D']:
                        print('ANSWER MODEL:', r)
                        self.interesting_images[answer].append(futures[future])
                        self.answer_counter[answer] += 1.01
                        pred = answer
                        res = r

            answer_mdata = {'ANSWER PROMPT': answer_prompt, 'ANSWER RESPONSE': res}
            if res is None:
                res = r


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
            # # 
    
            real_actions = self.annotate_image(agent_state, obs) 
            
            images = self.get_sensor_images(obs, convert=False)
            # zoomed_map = self.generate_topdown(real_actions)
            zoomed_map = self.generate_unpriv(real_actions, zoom=10)
            prompt_question = (
                "Your task is to navigate through the environment and learn the answer to the following quesiton\n"
                f"[QUESTION]: {vlm_question}\n"
                f"There are {len(real_actions) - 1} red arrows superimposed onto your observation{'s' if multi else''}, which represent potential actions. " 
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
                f"There are {len(real_actions) - 1} red arrows superimposed onto your observation{'s' if multi else''}, which represent potential actions. " 
                f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'Note that action 0 turns you around completely .' if self.step - self.turned >= 3 else ''}"
    
                "\nYou ALSO have a topdown map of the environment, with unexplored area shown in LIGHT GREY and obstacles shown in BLACK. This map shows you where you have been in the past, shown in GREEN. Your current location is shown by a RED dot. "
                "The same actions you see superimposed on the RGB image are also shown on the top-down map. These actions also represented by red arrows and white circles, and show the location you would move to if you took that action. "
                "Use this map to help you explore new areas (light grey). "
                
                "First, tell me what you see from your current sensor observations. "
                "Second, tell me what room or location you should navigate to in order to answer the question, and which direction you should go to reach that. "
                "If you are not sure where to go, use the map to help you explore. "
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
        rgb = self.topdown_map
        green = np.sum(rgb == self.explored_color)
        light_grey = np.sum(rgb == self.unexplored_color)
        row['explored'] = green / (green + light_grey) 
        
        images = self.get_sensor_images(obs) + [zoomed_map]
        print(f'action {row["actions"]}, pred {row["answer"]}, ground {answer}')
        if row["actions"] <= len(list(real_actions.keys())) and row["actions"] > 0:
            mag, theta = list(real_actions.keys())[row["actions"]-1]
            self.update_unpriv(mag, theta, agent_state, mag, clip_frac=1)
            self.update_topdown(mag, theta, agent_state, mag, clip_frac=1)
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

        self.run_metadata['final_answer1'] = 'E' #random.choice(['A', 'B', 'C', 'D'])
        self.run_metadata['final_answer2'] = 'E' #random.choice(['A', 'B', 'C', 'D'])
        path = f'logs/{self.outer_run_name}/{self.curr_run_name}/step_FINAL'
        os.makedirs(path, exist_ok=True)

        for j in range(2):
            images = []
            for k, v in self.interesting_images.items():
                if len(v) > 0:
                    images += random.choices(v, k=min(j+1, len(v)))

            answer_prompt = (f"The agent has sent you {len(images)} images from the same environment, and is asking you the following question about the environment [QUESTION]: {self.vlm_question}\n First, tell me what you see in each image, and if there any notable objects that are relevant to the question. Then, explain what the best answer choice is and why. Lastly, return it as a JSON like {{'answer': <answer letter>}}")

            if len(images) > 0:
                r, p = self.answerVLM.call(images, answer_prompt)
                print('FINAL ANSWER MODEL:', r)
                dct = self.parse_response(r)
                if 'answer' in dct and dct['answer'] in ['A', 'B', 'C', 'D']:
                    self.run_metadata[f'final_answer{j+1}'] = dct['answer']

                for i, im in enumerate(images):
                    im = Image.fromarray(im[:, :, 0:3], mode='RGB')
                    im.save(f'{path}/{j+1}_image_{i}.png')
                if len(images) > 0:
                    with open(f'{path}/{j+1}_final_answer.txt', 'w') as f:
                        f.write(r)
                        f.write(f'\n Answer counter {self.answer_counter}')
            