
from bisect import insort_right
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
from regex import D
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
            explore_factor = 0, **kwargs):
        
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
                self.curr_target = 'WASHER AND DRYER'
                self.curr_related_objects = []
                while self.init_pos is None:
                    point = self.annotatedSimulator.sim.pathfinder.get_random_navigable_point()
                    
                    # self.init_pos = point
                    self.init_pos = self.goal_loc
                    break
                    # break

                    if abs(point[1] - self.goal_loc[1]) < 0.1 and np.linalg.norm(self.goal_loc-point) > 0:
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
        path = self.get_shortest_path(self.init_pos, self.goal_loc)
        print('PATH TO GOAL:', path)
        self.curr_run_name = f'{self.curr_target}_{self.annotatedSimulator.scene_id}_{random.randint(0, 1000)}'
        obs = self.annotatedSimulator.step([('forward', 0)])
        return obs


    def step_env(self, obs):
        agent_state = self.get_agent_state()
        raw_images = [obs[f'color_sensor_{i}']['image'][:, :, 0:3].copy() for i in self.annotatedSimulator.sensors]
        real_actions = self.annotate_image(agent_state, obs)
        unpriv_map = self.unpriv_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        unpriv_map[mask] = self.explored_color

        zoomed_map = self.generate_unpriv(real_actions, zoom=9) if self.run_metadata['map_type'] == 'unpriv' else self.generate_topdown(real_actions, zoom=9)
        unpriv_map2 = self.unpriv_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        unpriv_map2[mask] = self.explored_color

        multi = len(self.annotatedSimulator.sensors) > 1
        prompt = (
        # f"First, analyze your updated camera observation and tell me the spatial layout of what you see. "
        f"I have lost my diamond ring! Your task is to search every inch of this floor to help me find it. It is a huge, bright diamond that is unmistakable. "
        f"There are {len(real_actions) - 1} red arrows superimposed onto your observation{'s' if multi else''}, which represent potential actions. " 
        f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or at a dead end .' if self.step - self.turned >= 3 else ''}"
        #f"Your task is to navigate to the {self.curr_target.upper()}. Think of a high level plan on how you can reach the {self.curr_target.upper()} from where you are now. If you have already reached the {self.curr_target.upper()} choose special action -1 (done). "
        f"First, tell me what you see in your sensor observations. Then, tell me a high level plan on how you will find the ring and where you will go next. {'Recall your past observations so that you dont waste time exploring the same locations. ' if self.run_metadata['history'] > 0 else ''}"
        f"Think about how each action will move you. Lastly, select one action from the image and explain how it helps you reach your goal. Return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS"
        )

        images = self.get_sensor_images(obs, convert=False)
        if self.run_metadata['use_map'] > 0:
            prompt = (

            # f"Your task is to navigate to a {self.curr_target.upper()}, and you have {len(real_actions) - 1} actions available to you. "
            # "You have two different sources of information that show these actions: \n1. your RGB sensors show your current view of the environment, and the actions are superimposed onto the images as red arrows to white circles. The white cicles show the exactly where the action will move you. "
            # "\n2. you have a topdown map of the environment, with navigable area shown in GREEN and obstacles shown in black. This map shows the trajectory of where you have been in the past, shown GRAY. Your current location is shown by a RED dot. "
            # "The same actions you see superimposed on the RGB image are also shown on the top-down map. These actions also represented by red arrows and white circles, and show the location you would move to if you took that action. "
            # f"Carefully analyze this map and make sure you understand what it means. {'Remember you have action 0 for when you are in a dead end or want to turn around' if self.step - self.turned >= 3 else ''}"
            # f"\n\nFirst, tell me what you see in your sensor observations, and which way you should go to reach your goal. Second, describe the map and think about which actions leave your curent room. Combine both sources of information to make an informed decision on what action to take. "
            # f"If you have already comleted your goal choose special action -1. Return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS "
            # # Think of a high level plan on how you can reach a {self.curr_target.upper()} from where you are now. ")
            # )
            f"I have lost my diamond ring! Your task is to search every inch of this floor to help me find it. It is a huge, bright diamond that is unmistakable. "

            # f"Your task is to navigate to the {self.curr_target.upper()}, and "you have {len(real_actions) - 1} actions available to you. "
            # f"There are {len(real_actions) - 1} red arrows superimposed onto your observation{'s' if multi else''}, which show where you would move to if you chose that action number ." 
            "\nTo help you with exploration, you have a topdown map of the environment, with unexplored area shown in GREEN and obstacles shown in BLACK. This map shows where you have been in the past, in GRAY. Your current location is shown by a RED dot. "
            f"You have {len(real_actions) - 1} actions, which are red arrows and white circles, and show the location you would move to if you took that action number. Use the map to identify unexplored rooms (GREEN) and plan out actions that help you reach these unexplored areas. Note you will sometimes need to backtrack through gray areas to reach new rooms. "
            f"{'NOTE: choose action 0 if you want to TURN AROUND or at a dead end .' if self.step - self.turned >= 3 else ''}"            # "If you have already reached the {self.curr_target.upper()} choose special action -1 (done). "
            f'{"First, describe the map you see, as well as your sensor observations. Tell me a high level plan on how you will find the ring and where you will go next. " if self.run_metadata["use_map"] != 2 else ""}'
            # "Use your map to strategically think about which actions will you get to new territory, and remember the gray areas are navigable and the black areas are not "    
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
        logs = []
        if self.pivot is not None:
            instruction = f'Navigate to the {self.curr_target.upper()}'
            pivot_actions, log_images = self.pivot.run(raw_images[0], obs['depth_sensor_0'], instruction, agent_state, agent_state.sensor_states['color_sensor_0'])
            metadata = {'pivot_actions': pivot_actions}
            logs += log_images
            resp = ""
            row['actions'] = -10
        else:
            row, metadata, resp = self.agent_self_consitency(prompt, images, row, self.run_metadata['consistency'])
    
        row['goal_object'] = self.curr_target
        row['dist_traveled'] = self.distance_traveled
        min_dist = 1000
        closest_object = None

        if row["actions"] <= len(list(real_actions.keys())) and row["actions"] > 0:
            mag, theta = list(real_actions.keys())[row["actions"]-1]
            self.update_unpriv(mag, theta, agent_state, mag, clip_frac=1)
        images = self.get_sensor_images(obs, convert=False) + [zoomed_map]
        copies = [unpriv_map]
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
        row['distance_to_goal'] = np.linalg.norm(agent_state.sensor_states['color_sensor_0'].position - self.goal_loc)
        metadata['DIST TO GOAL'] = row['distance_to_goal']

        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        if self.run_metadata['mask_thinking'] and row['success'] == 1 and self.run_metadata['history'] > 0:
            self.vlm.session.history[-1].parts[0].text = f'\n' + '{"action": ' + str(row["actions"]) + '}'

        if self.step % self.log_freq == 0 or row['success'] == 0:
            images = [Image.fromarray(im[:, :, 0:3], mode='RGB') for im in images+logs]
            copies = [Image.fromarray(im[:, :, 0:3], mode='RGB') for im in copies]
            self.log(images, resp, row['success'], metadata, copy_images=copies)

        if self.step >= 3 and self.df['actions'].iloc[-3:].tolist().count(-1) >= 2:
            print("STOPPING EARLY, DONE")
            return None
        if self.pivot is not None:
            return pivot_actions
        return self.annotatedSimulator.move_choices(row['actions'], points=list(real_actions.keys()))        










class GOATBench(DynamicBench):

    task = 'GOAT_BENCH'

    def setup_experiment(self, split, num_scenes):
        self.goat_data = []
        self.split_name = split
        self.split = 'val' if 'val' in split else 'train'
        self.sim_kwargs['scene_config'] =  f"datasets/hm3d/hm3d_annotated_{self.split}_basis.scene_dataset_config.json"
        self.sim_kwargs['goal_image_agent'] = True
        self.goat_ndx = -1
        dir = split

        for f in os.listdir(f'datasets/goatBench/{dir}/content')[0:num_scenes]:
            with gzip.open(f'datasets/goatBench/{dir}/content/{f}', 'rt') as gz:
                loaded = json.load(gz)
                self.goat_data.append(loaded)

        random.shuffle(self.goat_data)


    def setup_run(self, history=7, mask_thinking=True, add_timesteps_prompt=True, draw_arrows=True,
            points=None, consistency=1, max_steps_per_goal=5, priv_actions=False, use_map=0, uniform=False, 
            explore_factor=0, map_type='priv', success_thresh=2.5, **kwargs):
        self.goat_ndx += 1
        goat_scene = self.goat_data[self.goat_ndx]
        while True:
            episode = random.choice(goat_scene['episodes'])
            f, glb = episode['scene_id'].split('/')[-2:]

            if os.path.exists(f'logs/{self.outer_run_name}/{self.goat_ndx}_{episode["episode_id"]}_{episode["scene_id"]}'):
                continue
            break

        self.sim_kwargs['scene_id'] = f[2:5]
        self.sim_kwargs['scene_path'] = f'datasets/hm3d/{self.split}/{f}/{glb}'
        self.annotatedSimulator = AnnotatedSimulator(**self.sim_kwargs)
        self.annotatedSimulator.priv_actions = priv_actions
        self.annotatedSimulator.do_draw_arrows = points if draw_arrows else None
        self.annotatedSimulator.do_annotate_image = False
        self.false_positives = 0
        self.false_negatives = 0
        self.curr_episode = []

        self.init_pos = np.array(episode['start_position'])
        self.set_state(pos=self.init_pos, quat=episode['start_rotation'])
        for goal in episode['tasks']:
            name = goal[0]
            mode = goal[1]
            
            target = {'name': name, 'mode': mode, 'id': goal[2], 'view_points': []}
            
            descriptions = goat_scene['goals'][f'{f[6:]}.basis.glb_{name}']
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

        print(f'Running episode with {len(self.curr_episode)} goals')
        for i, obj in enumerate(self.curr_episode):
            print(f'Goal {i}: {obj["name"]}, {obj["mode"]}')

        self.curr_goal_ndx = 0
        self.curr_run_name = f"{self.goat_ndx}_{episode['episode_id']}_{self.annotatedSimulator.scene_id}"
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
            'success_thresh': success_thresh,
            'split': self.split_name
        }

        self.curr_shortest_path = self.get_shortest_path(self.init_pos, goal['view_points'], multi=True)
        print(f'Current {goal["mode"]}: {goal["name"]}, GEODESIC: {self.curr_shortest_path}, num_view_points: {len(goal["view_points"])}')
        self.reached_early = False
        self.early_spl = 0
        obs = self.annotatedSimulator.step([('forward', 0)])
        return obs

    def step_env(self, obs):
        agent_state = self.get_agent_state()
        goal = self.curr_episode[self.curr_goal_ndx]
        multi = len(self.annotatedSimulator.sensors) > 1
        if goal['mode'] == 'object':
            t = f'Navigate to the nearest {goal["name"]}'
            inst = f'Find the nearest {goal["name"]} and navigate as close as you can to it. '
            inst2 =  f'Tell me which room you would find this {goal["name"]} in? Do you see any {goal["name"]} in your current observations?' 
        if goal['mode'] == 'description':
            inst = f"Find and navigate to the {goal['lang_desc']}. Navigate as close as you can to it "
            t = inst
            inst2 =  f"Tell me which room you would find this specifc {goal['name']} in, {'and which sensor looks the most promising. ' if multi else 'and which general direction you should go in. '}"
        if goal['mode'] == 'image':
            t = f'Navigate to the specific {goal["name"]} shown in the image labeled GOAL IMAGE. Pay close attention to the details. Navigate as close as you can to it '
            inst = f"Observe the image labeled GOAL IMAGE. Find this specific {goal['name']} shown in the image and navigate to it. "
            inst2 =  f"Tell me which room you would find this {goal['name']} in, {'and which sensor looks the most promising. ' if multi else 'and which general direction you should go in. '}"

        goal_ims = []
        raw_images = [a.copy() for a in self.get_sensor_images(obs, convert=False)]
        if goal['mode'] == 'image':
            position = goal['image_position']
            rotation = goal['image_rotation']
            goal_im = self.annotatedSimulator.get_goal_image(position, rotation)
            put_text_on_image(goal_im, f"GOAL IMAGE: {goal['name']}", background_color=(255, 255, 255), location='top_center')
            goal_ims.append(goal_im)

        def done_thread():
            if self.step - self.called_done < 2:
                return False, None
            answer_prompt = (f"The agent has the following navigation task: {t}. The agent has sent you an image taken from its current location. "
                             f'Your job is to determine whether the agent is within 1 meter of the specified {goal["name"]}'
                             f"First, tell me what you see in the image, and tell me if there is a {goal['name']} that matches the description. Then, return 1 if the agent is within 1 meter of the goal, and 0 if it isnt. Format your answer in the json {{'done': <1 or 0>}}")

            def process_image(image):
                r, p = self.answerVLM.call([image] + goal_ims, answer_prompt, logprobs=5)
                dct = self.parse_response(r)
                if 'done' in dct and int(dct['done']) == 1:
                    return 1, r
                return 0, r

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(process_image, image): image for image in raw_images}

                for future in concurrent.futures.as_completed(futures):
                    isDone, r = future.result()
                    if isDone:
                        print('ANSWER MODEL THINKS DONE:', r)
                        self.called_done = self.step
                        return True, r
            return False, r


        def preprocessing_thread():
            images = []
            real_actions = self.annotate_image(agent_state, obs)
            images += self.get_sensor_images(obs, convert=False)
            zoomed_map = self.generate_unpriv(real_actions, zoom=9) if self.run_metadata['map_type'] == 'unpriv' else self.generate_topdown(real_actions, zoom=9)
            if self.run_metadata['use_map'] or goal['mode'] in ['description', 'object']:
                images.append(zoomed_map)       

            min_euclidian = min([np.linalg.norm(agent_state.position - vp) for vp in goal['view_points']])

            return real_actions, images, zoomed_map, min_euclidian
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future1 = executor.submit(preprocessing_thread)
            future2 = executor.submit(done_thread)

            real_actions, images, zoomed_map, min_euclidian = future1.result() 
            done, r = future2.result()   

        multi = len(self.annotatedSimulator.sensors) > 1

        prompt = (

        f"TASK: {inst} use your prior knowledge about where items are typically located within a home"
        f"There are {len(real_actions) - 1} red arrows superimposed onto your observation{'s' if multi else''}, which represent potential actions. " 
        f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS.' if self.step - self.turned >= 3 else ''}"
        f"First, tell me what you see in each of your sensor observations, and if you have any leads on finding the {goal['name']}. Second, {inst2}. "
        f"Lastly, explain which action is the best and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
        )

        if self.run_metadata['use_map'] and goal['mode'] != 'image':
            prompt = (
        f"TASK: {inst} use your prior knowledge about where items are typically located within a home"
        f"There are {len(real_actions) - 1} red arrows superimposed onto your observation{'s' if multi else''}, which represent potential actions. " 
        f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS .' if self.step - self.turned >= 3 else ''}"
        "\nYou also have a topdown map of the environment, with unexplored area shown in GREEN. This map shows you where you have been in the past, shown in GRAY "
        "The same actions you see superimposed on the RGB image are also shown on the top-down map. "
        f"First, tell me what you see in each of your sensor observations, and if you have any leads on finding the {goal['name']}. Second, {inst2}. "
        "If you are not sure where to go, analyze the map to help you explore (GREEN AREAS). "
        f"Lastly, explain which action is the best and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
            )
        
        row = {'actions': -10, 'tokens_generated':0, 'success': 1, 'metadata': self.run_metadata, 'false_positives': self.false_positives, 'false_negatives': self.false_negatives,
        'speed': 0, 'goal': goal['name'], 'goal_mode': goal['mode'], 'goal_index': self.curr_goal_ndx, 'curr_goal_steps': self.step - self.last_goal_reset,
        'model': self.vlm.name, 'input_tokens': 0, 'agent_location': agent_state.position, 'distance_traveled': self.distance_traveled, 'curr_shortest_path': self.curr_shortest_path}
        logs = []
        if done:
            row['actions'] = -1
            rgb = self.topdown_map
            green = np.sum(rgb == self.explored_color)
            light_grey = np.sum(rgb == self.unexplored_color)
            resp = r
            metadata = {'ANSWER PROMPT': t}
            row['explored'] = green / (green + light_grey) 

        else:
            if self.pivot is not None:
                instruction = inst
                pivot_actions, log_images = self.pivot.run(raw_images[0], obs['depth_sensor_0'], instruction, agent_state, agent_state.sensor_states['color_sensor_0'], goal_image = goal_ims[0] if len(goal_ims) > 0 else None)
                metadata = {'pivot_actions': pivot_actions}
                logs += log_images
                resp = ""
                row['actions'] = -10
            else:
                row, metadata, resp = self.agent_self_consitency(prompt, images + goal_ims, row, self.run_metadata['consistency'])
            metadata['DONE RESP'] = r
        images = self.get_sensor_images(obs, convert=False) + [zoomed_map] + goal_ims 
        if row["actions"] <= len(list(real_actions.keys())) and row["actions"] > 0:
            mag, theta = list(real_actions.keys())[row["actions"]-1]
            self.update_unpriv(mag, theta, agent_state, mag, clip_frac=1)
            # self.update_topdown(mag, theta, agent_state, mag, clip_frac=1)

        min_dist = min_euclidian
        row['distance_to_goal'] = min_dist
        row['shortest_path_for_goal'] = self.curr_shortest_path
        metadata['DIST TO GOAL'] = row['distance_to_goal']
       
        print('distance to goal', round(row['distance_to_goal'], 2), 'min euclidian', round(min_euclidian, 2))
        row['spl'] = 0
        row['early_spl'] = self.early_spl
        row['reached_1.5'] = self.reached_early
        metadata['INST'] = inst
        done = False
        new_goal = False
        goal_reached = False
        if min_dist < 1 and row['actions'] == -1: 
            self.reached_early = True
            row['reached_1.5'] = True
            if self.early_spl == 0:
                self.early_spl = self.curr_shortest_path / max(self.curr_shortest_path, self.distance_traveled)
            row['early_spl'] = self.early_spl

        if min_dist < self.run_metadata['success_thresh'] and row['actions'] == -1:
            print(f"SUCESSFULLY FINISHED GOAL {self.curr_goal_ndx} in {round(self.distance_traveled, 2)} meters")
            new_goal = True
            print('THE OPTIMAL PATH WAS:', round(self.curr_shortest_path, 2))
            row['spl'] = (self.curr_shortest_path)/max(self.curr_shortest_path, self.distance_traveled)
            goal_reached = True

        elif self.step + 1 - self.last_goal_reset > self.run_metadata['max_steps_per_goal']:
            print('MAX STEPS PER GOAL REACHED')
            new_goal = True
            goal_reached = False
        elif row['distance_to_goal'] < self.run_metadata['success_thresh']:
            self.false_negatives += 1
            print('NEAR GOAL BUT MODEL DID NOT RETURN DONE')
        elif row['actions'] == -1:
            print(f'MODEL RETURNED DONE BUT STILL {round(min_dist, 2)} METERS AWAY')
            self.false_positives += 1

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
            put_text_on_image(copy, f"{self.curr_goal_ndx}: {goal['name']}({goal['mode'][0]})_{round(min_dist, 2)}", background_color=background_color, location='top_left', text_size=2.3)
            copies.append(copy)

        copies.append(self.topdown_map)

        if self.run_metadata['mask_thinking'] and row['success'] == 1 and self.run_metadata['history'] > 0:
            self.vlm.session.history[-1].parts[0].text = f'\n' + '{"action": ' + str(row["actions"]) + '}'
        row['goal_reached'] = goal_reached
        row['new_goal'] = new_goal
        if new_goal:
            self.early_spl = 0
            self.reached_early = False
            self.false_positives = 0
            self.false_negatives = 0
            self.called_done = self.step - 1
            self.turned = self.step - 2
            self.explored_map = np.zeros_like(self.explored_map)
            self.curr_goal_ndx += 1
            self.last_goal_reset = self.step
            self.distance_traveled = 0
            if self.curr_goal_ndx >= len(self.curr_episode):
                done = True
                print("FINISHING TRAJECTORY, NO MORE GOALS")
            else:
                print(f"Moving onto")
                goal = self.curr_episode[self.curr_goal_ndx]
                self.curr_shortest_path = self.get_shortest_path(self.init_pos, goal['view_points'], multi=True)
                print(f'Current {goal["mode"]}: {goal["name"]}, GEODESIC: {self.curr_shortest_path}')


        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        if self.step % self.log_freq == 0 or row['success'] == 0:
            images = [Image.fromarray(im[:, :, 0:3], mode='RGB') for im in images + logs]
            copies = [Image.fromarray(im[:, :, 0:3], mode='RGB') for im in copies]
            self.log(images, resp, row['success'], metadata, copy_images=copies)
            
        if done:
            return None
        self.vlm.get_spend()
        self.answerVLM.get_spend()
        if self.pivot is not None and row['actions'] != -1:
            return pivot_actions
        return self.annotatedSimulator.move_choices(row['actions'], points=list(real_actions.keys()))   



class HMONBench(DynamicBench):

    task = 'HMON'

    def setup_experiment(self, split, num_scenes=50, num_episodes_per_scene=10):
        self.scene_data = []
        self.split = split
        self.sim_kwargs['scene_config'] =  f"datasets/hm3d/hm3d_annotated_{self.split}_basis.scene_dataset_config.json"
        self.ep_ndx = -1
        if self.split == 'train':
            dir = 'train'
        else:
            dir = 'val'
        
        for f in random.choices(os.listdir(f'datasets/hmon2023/{dir}/content'), k=num_scenes):
            with gzip.open(f'datasets/hmon2023/{dir}/content/{f}', 'rt') as gz:
                js = json.load(gz)
                random.shuffle(js['episodes'])
                self.scene_data.append(js)
        
        random.shuffle(self.scene_data)
        self.num_episodes_per_scene = num_episodes_per_scene
        self.answerVLM = GeminiModel('gemini-1.5-flash', 'You are an intelligent agent that excels at identifying objects in an image. ')

    def setup_run(self, history=7, mask_thinking=True, add_timesteps_prompt=True, draw_arrows=True,
            points=None, consistency=1, priv_actions=False, use_map=0, uniform=False, 
            explore_factor=0, map_type='priv', success_thresh=2.5, **kwargs):

        self.ep_ndx += 1
        scene = self.scene_data[(self.ep_ndx//self.num_episodes_per_scene) % len(self.scene_data)]
        
        episode = scene['episodes'][self.ep_ndx % self.num_episodes_per_scene]
        f = episode['scene_id'].split('/')[1:]
        # pdb.set_trace()
        self.sim_kwargs['scene_id'] = f[1][2:5]
        self.sim_kwargs['scene_path'] = f'datasets/hm3d/{self.split}/{f[1]}/{f[2]}'
        self.annotatedSimulator = AnnotatedSimulator(**self.sim_kwargs)
        self.annotatedSimulator.do_annotate_image = False
        self.false_positives = 0
        self.false_negatives = 0

        all_objects = scene['goals_by_category'][f'{f[-1]}_{episode["object_category"]}']
        view_positions = []
        for obj in all_objects:
            for vp in obj['view_points']:
                view_positions.append(vp['agent_state']['position'])

        # view_positions = [a['agent_state']['position'] for a in all_objects]
        print(f'RUNNING EPISODE {self.ep_ndx} with {episode["object_category"]} and {len(all_objects)} instances. GEODESIC DISTANCE: {episode["info"]["geodesic_distance"]}, NUM VIEWPOINTS: {len(view_positions)}')

        if episode['object_category'] == 'tv_monitor':
            episode['object_category'] = 'tv screen'
        self.curr_episode = {'object': episode['object_category'], 'shortest_path': episode['info']['geodesic_distance'], 'object_positions': [a['position'] for a in all_objects], 'view_positions': view_positions}
        self.init_pos = np.array(episode['start_position'])
        self.set_state(pos=self.init_pos, quat=episode['start_rotation'])
        self.reached_early = False
        self.early_spl = 0
        self.curr_run_name = f"{self.ep_ndx}_{f[1][2:5]}"
        self.last_goal_reset = -1
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
            'use_map': use_map,
            'uniform': uniform,
            'explore_factor': explore_factor,
            'map_type': map_type,
            'success_thresh': success_thresh, **self.curr_episode,
            'use_euclidian': kwargs.get('euclid', True)
        }


        obs = self.annotatedSimulator.step([('forward', 0)])
        return obs

    def step_env(self, obs):
        rng_state = random.getstate()
        agent_state = self.get_agent_state()
        for position in self.curr_episode['object_positions']:
            self.update_topdown(0, 0, agent_state, 0, clip_frac=1, goal=position, goal_name=self.curr_episode['object'])
        raw_images = [a.copy() for a in self.get_sensor_images(obs, convert=False)]

        def done_thread():
            if self.step - self.called_done < 2:
                return False, None
            answer_prompt = (f"The agent has has been tasked with navigating to a {self.curr_episode['object'].upper()}. The agent has sent you an image taken from its current location. "
                             f'Your job is to determine whether the agent is within 1 meter of the a {self.curr_episode["object"]}'
                             f"First, tell me what you see in the image, and tell me if there is a {self.curr_episode['object']}. Return 1 if the agent is within a meter of the goal, and 0 if it isnt. Format your answer in the json {{'done': <1 or 0>}}")

            def process_image(image):
                r, p = self.answerVLM.call([image], answer_prompt, logprobs=5)
                dct = self.parse_response(r)
                if 'done' in dct and int(dct['done']) == 1:
                    return 1, r
                return 0, r

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(process_image, image): image for image in raw_images}

                for future in concurrent.futures.as_completed(futures):
                    isDone, r = future.result()
                    if isDone:
                        print('ANSWER MODEL THINKS DONE:', r)
                        self.called_done = self.step
                        return True, r

            return False, r
        
        def preprocessing_thread():
            images = []
            real_actions = self.annotate_image(agent_state, obs)
            images += self.get_sensor_images(obs, convert=False)


            zoomed_map = self.generate_unpriv(real_actions, zoom=9) if self.run_metadata['map_type'] == 'unpriv' else self.generate_topdown(real_actions, zoom=9)
            if self.run_metadata['use_map']:
                images.append(zoomed_map)
            distances = [np.linalg.norm(position - agent_state.position) for position in self.curr_episode['view_positions']]
            min_euclidian = min(distances)
            return real_actions, images, zoomed_map, min_euclidian
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future1 = executor.submit(done_thread)
            future2 = executor.submit(preprocessing_thread)
            
            real_actions, images, zoomed_map, min_euclidian = future2.result()
            done, r = future1.result()

    
        multi = len(self.annotatedSimulator.sensors) > 1
        prompt = (

        f"TASK: NAVIGATE TO THE NEAREST {self.curr_episode['object'].upper()}, and get as close to it as possible. Use your prior knowledge about where items are typically located within a home "
        f"There are {len(real_actions) - 1} red arrows superimposed onto your observation{'s' if multi else''}, which represent potential actions. " 
        f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step - self.turned >= 3 else ''}"
        f"First, tell me what you see in your sensor observations, and if you have any leads on finding the {self.curr_episode['object'].upper()}. {'Second, tell me which sensor looks the most promising. ' if multi else 'Second, tell me which general direction you should go in. '}"
        f"Lastly, explain which action acheives that best, and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
        )

        if self.run_metadata['use_map']:
            prompt = (
        f"TASK: NAVIGATE TO THE NEAREST {self.curr_episode['object'].upper()} and get as close to it as possible. Use your prior knowledge about where items are typically located within a home"
        f"There are {len(real_actions) - 1} red arrows superimposed onto your observation{'s' if multi else''}, which represent potential actions. " 
        f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step - self.turned >= 3 else ''}"
        "\nYou also have a topdown map of the environment, with unexplored area shown in GREEN. This map shows you where you have been in the past, shown in GRAY. "
        "The same actions you see superimposed on the RGB image are also shown on the top-down map. "
        f"First, tell me what you see in your sensor observations, and if you have any leads on finding the {self.curr_episode['object'].upper()}. {'Second, tell me which sensor looks the most promising. ' if multi else 'Second, tell me which general direction you should go in. '}"
        "If you are not sure where to go, analyze map to help you explore (GREEN AREAS). "
        f"Lastly, explain which action is the best and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
            )
        
        row = {'actions': -10, 'tokens_generated':0, 'success': 1, 'metadata': self.run_metadata, 'false_positives': self.false_positives, 'false_negatives': self.false_negatives,
        'speed': 0, 'object': self.curr_episode['object'], 'distance_traveled': self.distance_traveled,
        'model': self.vlm.name, 'input_tokens': 0, 'agent_location': agent_state.position}
        logs = []
        if done:
            row['actions'] = -1
            rgb = self.topdown_map
            green = np.sum(rgb == self.explored_color)
            light_grey = np.sum(rgb == self.unexplored_color)
            resp = r
            metadata = {}
            row['explored'] = green / (green + light_grey) 
            row['distance_traveled'] = self.distance_traveled
        else:
            if self.pivot is not None:
                instruction = f"TASK: NAVIGATE TO THE NEAREST {self.curr_episode['object'].upper()} and get as close to it as possible. Use your prior knowledge about where items are typically located within a home"
                pivot_actions, log_images = self.pivot.run(raw_images[0], obs['depth_sensor_0'], instruction, agent_state, agent_state.sensor_states['color_sensor_0'])
                metadata = {'pivot_actions': pivot_actions}
                logs += log_images
                resp = ""
                row['actions'] = -10
            else:
                row, metadata, resp = self.agent_self_consitency(prompt, images, row, self.run_metadata['consistency'])
                metadata['DONE RESP'] = r

        images = self.get_sensor_images(obs, convert=False) + [zoomed_map]
        if row["actions"] <= len(list(real_actions.keys())) and row["actions"] > 0:
            mag, theta = list(real_actions.keys())[row["actions"]-1]
            self.update_unpriv(mag, theta, agent_state, mag, clip_frac=1)
            
        row['distance_to_goal'] = min_euclidian
        metadata['DIST TO GOAL'] = row['distance_to_goal']
        row['spl'] = 0
        row['goal_reached'] = False
        row['spl_1.5'] = self.early_spl
        row['goal_reached_1.5'] = self.reached_early

        print('distance to goal', round(row['distance_to_goal'], 2), 'min euclidian', round(min_euclidian, 2))
        if (min_euclidian < 1 and row['actions'] == -1):
            print('reached early')
            row['goal_reached_1.5'] = True 
            self.reached_early = True
            if row['spl_1.5'] == 0:
                row['spl_1.5'] = (self.curr_episode['shortest_path'])/max(self.curr_episode['shortest_path'], self.distance_traveled)
                self.early_spl = row['spl_1.5']                

        if (min_euclidian < self.run_metadata['success_thresh']) and row['actions'] == -1:
            print(f"SUCESSFULLY FINISHED GOAL in {round(self.distance_traveled, 2)} meters of distance. Shortest path was {round(self.curr_episode['shortest_path']), 2}")
            
            row['goal_reached'] = True 
            row['spl'] = (self.curr_episode['shortest_path'])/max(self.curr_episode['shortest_path'], self.distance_traveled)

        elif row['distance_to_goal'] < self.run_metadata['success_thresh']:
            self.false_negatives += 1
            print('NEAR GOAL BUT MODEL DID NOT RETURN DONE')
        elif row['actions'] == -1:
            print(f'MODEL RETURNED DONE BUT STILL {round(min_euclidian, 2)} METERS AWAY')
            self.false_positives += 1

        copies = []
        for sensor in self.annotatedSimulator.sensors:
            copy = obs[f'color_sensor_{sensor}']['image'].copy()
            self.draw_arrows(real_actions, copy, agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], chosen_action=row['actions'], real_actions=real_actions)
            if row['goal_reached']:
                background_color = (0, 100, 0) 
            elif self.step == self.inner_loop - 1:
                background_color = (100, 0, 0) 
            else:
                background_color = (255, 255, 255)  
            put_text_on_image(copy, f"{self.curr_episode['object']}_{np.round(row['distance_to_goal'], 2)}", background_color=background_color, location='top_left', text_size=2.3)
            copies.append(copy)

        copies.append(self.topdown_map)

        if self.run_metadata['mask_thinking'] and row['success'] == 1 and self.run_metadata['history'] > 0:
            self.vlm.session.history[-1].parts[0].text = f'\n' + '{"action": ' + str(row["actions"]) + '}'

        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        if self.step % self.log_freq == 0 or row['success'] == 0:
            images = [Image.fromarray(im[:, :, 0:3], mode='RGB') for im in images+logs]
            copies = [Image.fromarray(im[:, :, 0:3], mode='RGB') for im in copies]
            self.log(images, resp, row['success'], metadata, copy_images=copies)
            
        if row['goal_reached']:
            return None
        self.vlm.get_spend()
        random.setstate(rng_state)

        if self.pivot is not None and row['actions'] != -1:
            return pivot_actions
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
            points=None, consistency=1, max_steps_per_goal=5, uniform=False, use_map=True, explore_factor=1.5, map_type='priv', **kwargs):
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
            'map_type': map_type
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
            extra = "\n" + es[len(choices)] + "." + " " + "I need the agent to move to a better location in order to answer the question. "
            pred = 'E'
            answer_prompt = (f"The agent has sent you an image, and is asking you the following question [QUESTION]: {vlm_question+extra}\n\n[TASK]: First, tell me where you are in the environmemt, and if there are any notable objects that are relevant to the question. Second, explain what the answer is and why. Lastly, return it as a JSON like {{'answer': <answer letter>}} Pay close attention to the specific details in the question, note you might not be able to answer the question from the image. For example if the question asks about a shower, but you see the kitchen, you should return E")
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
            return pred, answer_mdata
  

        def action_thread():
            row = {'actions': -10, 'tokens_generated':0, 'success': 1, 'metadata': self.run_metadata, 'cum_answer': None, 'shortest_path': int(self.run_metadata['max_steps']/10),
            'speed': 0, 'scene_id': self.annotatedSimulator.scene_id, 'question': question, 'choices': choices, 'answer': None, 'ground_truth': answer,
            'model': self.vlm.name, 'input_tokens': 0, 'agent_location': agent_state.position, 'actions': -10, 'prediction': None}
            # # 
    
            real_actions = self.annotate_image(agent_state, obs) 
            
            images = self.get_sensor_images(obs, convert=False)
            
            zoomed_map = [self.generate_unpriv(real_actions, zoom=9) if self.run_metadata['map_type'] == 'unpriv' else self.generate_topdown(real_actions, zoom=10)]
            prompt_question = (
                "Your task is to navigate through the environment and learn the answer to the following quesiton\n"
                f"[QUESTION]: {vlm_question}\n"
                f"There are {len(real_actions) - 1} red arrows superimposed onto your observation{'s' if multi else''}, which represent potential actions. " 
                f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or are at a dead end .' if self.step - self.turned >= 3 else ''}"
                "First, tell me what you see from your each of your current sensor observations, and if there are any notable objects that are relevant to the question. "
                "Second, tell me a room or location you should navigate to in order to answer the question, and which general direction you should go. "
                "Lastly, explain which is the best action and return it in the format {'action': <action_number>}. Don't answer the question, just return an action. "
                "Note you CANNOT GO THROUGH CLOSED DOORS."
            )

            if self.run_metadata['use_map']:
                prompt_question = (
                "Your task is to navigate throughout the environment and learn the answer to the following quesiton\n"
                f"[QUESTION]: {vlm_question}\n"
                f"There are {len(real_actions) - 1} red arrows superimposed onto your observation{'s' if multi else''}, which represent potential actions. " 
                f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or are at a dead end .' if self.step - self.turned >= 3 else ''}"
    
                "\nYou ALSO have a topdown map of the environment, with unexplored area shown in GREEN. This map shows you where you have been in the past, shown in GRAY. "
                "The same actions you see superimposed on the images are also shown on the topdown map. "
                
                "First, tell me what you see from your current sensor observations and if there are any notable objects that are relevant to the question. "
                "Second, tell me what room or location you should navigate to in order to answer the question, and which general direction you should go. "
                "If you are not sure where to go, use the map to plan an exploration strategy that gets you to the GREEN areas ."
                "Lastly, explain which is the best action and return it in the format {'action': <action_number>}. Don't answer the question, just return an action. "
                "Note you CANNOT GO THROUGH CLOSED DOORS."
                )

                images += zoomed_map
            logs = []
            if self.pivot is not None:
                instruction = ("Your task is to navigate throughout the environment and learn the answer to the following quesiton\n"
                f"[QUESTION]: {vlm_question}\n")
                pivot_actions, log_images = self.pivot.run(raw_images[0], obs['depth_sensor_0'], instruction, agent_state, agent_state.sensor_states['color_sensor_0'])
                metadata = {'pivot_actions': pivot_actions}
                logs += log_images
                resp = ""
                row['actions'] = -10
                row['pivot'] = pivot_actions
                return row, metadata, resp, zoomed_map+logs, real_actions
            return *self.agent_self_consitency(prompt_question, images, row ,1), zoomed_map, real_actions
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future2 = executor.submit(action_thread)
            future1 = executor.submit(answer_thread)
            
            row, metadata, resp, zoomed_map, real_actions = future2.result()
            pred, answer_mdata = future1.result()
        
        row['answer'] = pred    
        row['cum_answer'] = max(self.answer_counter, key=self.answer_counter.get)
        
        images = self.get_sensor_images(obs) + zoomed_map
        print(f'action {row["actions"]}, pred {row["answer"]}, ground {answer}')
        if row["actions"] <= len(list(real_actions.keys())) and row["actions"] > 0:
            mag, theta = list(real_actions.keys())[row["actions"]-1]
            self.update_unpriv(mag, theta, agent_state, mag, clip_frac=1)
            # self.update_topdown(mag, theta, agent_state, mag, clip_frac=1)
        metadata['PREDICTION'] = row['answer']
        metadata['GROUND TRUTH'] = answer
        metadata.update(answer_mdata)

        copies = []
        for i, sensor in enumerate(self.annotatedSimulator.sensors):
            copy = images[i].copy()

            self.draw_arrows(real_actions, copy, agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], chosen_action=row['actions'], real_actions=real_actions)
            put_text_on_image(images[i], f"QUESTION: {question}", background_color=(255, 255, 255), location='top_left', text_size=1.5, text_thickness=2)
            put_text_on_image(copy, f"QUESTION: {question}", background_color=(255, 255, 255), location='top_left', text_size=1.5, text_thickness=2)
            if row['answer'] and not row['answer'] == 'E' and row['answer'] in vlm_pred_candidates:
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
        if(sorted_counter[0][1] > 5 and sorted_counter[0][1]/(sorted_counter[1][1]+0.001) > 2 and self.step > 12) or (sorted_counter[0][1] > 7 and self.step > 12):
            print("STOPPING EARLY, DONE")
            return None

        self.answerVLM.get_spend()
        self.vlm.get_spend()
        if self.pivot is not None:
            return row['pivot']
        return self.annotatedSimulator.move_choices(row['actions'], points=list(real_actions.keys()))        

    def post_run(self):

        self.final_answer()
        return super().post_run()

    def final_answer(self):

        self.run_metadata['final_answer1'] = 'E' #random.choice(['A', 'B', 'C', 'D'])
        path = f'logs/{self.outer_run_name}/{self.curr_run_name}/step_FINAL'
        os.makedirs(path, exist_ok=True)
        self.finalAnswerVlm = self.answerVLM
        def final_answer_thread():
            images = []
            for k, v in self.interesting_images.items():
                if len(v) > 0:
                    images += random.choices(v, k=min(1, len(v)))
                if len(images) > 2:
                    break
            
            answer_prompt = (f"The agent has sent you {len(images)} images from the SAME environment, and is asking you the following question about the environment [QUESTION]: {self.vlm_question}\n\n [TASK]: First, tell me what you see in each image, and if there any notable objects that are relevant to the question.  Second, explain what the best answer choice is and why. Lastly, return it as a JSON like {{'answer': <answer letter>}} ")

            final_ans = None
            r = 'Non images'
            if len(images) > 0:
                r, p = self.finalAnswerVlm.call(images, answer_prompt)
                print('FINAL ANSWER MODEL:', r)
                dct = self.parse_response(r)
                if 'answer' in dct and dct['answer'] in ['A', 'B', 'C', 'D']:
                    final_ans = dct['answer']

            return images, final_ans, r
        
        final_counter = {'A': [], 'B': [], 'C': [], 'D': [], 'E': []}
        num_parrallel = 5
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(final_answer_thread) for i in range(num_parrallel)]

            for future in concurrent.futures.as_completed(futures):
                images, answer, r = future.result()
                if answer:
                    final_counter[answer].append([images, r])
        if sum([len(v) for k, v in final_counter.items()]) == 0:
            print('NO FINAL ANSWER')
            return
        v = max(final_counter.items(), key=lambda x: len(x[1]))
        images, r = v[1][0]
        self.run_metadata['final_answer1'] = v[0]
        self.finalAnswerVlm.get_spend()

        if len(images) > 0:
            for i, im in enumerate(images):
                im = Image.fromarray(im[:, :, 0:3], mode='RGB')
                im.save(f'{path}/final_image_{i}.png')
            
        with open(f'{path}/final_answer.txt', 'w') as f:
            f.write(r)
            f.write(f'\n Answer counter {self.answer_counter}')
            f.write(f'\n final answer counters, {[(k, len(v)) for k, v in final_counter.items()]}')
    