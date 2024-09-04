import csv
from email import message
from gettext import find
import gzip
from hmac import new
import io
import json
from math import e
import math
from operator import index
import os
from pydoc import resolve
import re
from sqlite3 import DatabaseError
import sys
import pdb
import pickle
import random
from tracemalloc import start
from turtle import distance
from typing import Counter
from arrow import get
import arrow
from matplotlib import pyplot as plt
import numpy as np
import datetime
import cv2
import ast
import pandas as pd
from PIL import Image
from regex import E
from src.utils import *
from src.vlm import VLM, GeminiModel
from src.annoatedSimulator import AnnotatedSimulator
import habitat_sim
import cv2
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis
from scipy.ndimage import map_coordinates
from src.dynamicBench import *
from habitat.utils.visualizations import maps




class MultiAgentEQA(EQABench):


    task = 'MULTI_AGENT_EQA'

    def setup_experiment(self, split, scene_ids):
        file1 = 'datasets/EQA/questions.csv'
        file2 = 'datasets/EQA/scene_init_poses.csv'
        # self.vlm1 = GeminiModel(sys_instruction="You are an assistant with visual perception capabilities. Your objective is to provide helpful descriptions of what you see in the environemnt around you. ")
        # self.llm = GeminiModel(sys_instruction="You are the brains behind an embodied agent. Your objective is to analyze information and form high level plans for the agent to execute. You can also communicate with your partner agent, and use information from your partner to plan. ")
        # self.vlm2 = GeminiModel(sys_instruction="You are an embodied agent that can observe the environment around you and take actions. Your objective is to look at your available actions, and chose one that adheres to the high-level plan given to you")
        self.vlm = GeminiModel(sys_instruction="You control two embodied agents, each of which report visual observations to you. Your objective is to analyze the observations and choose actions for each agent to take. Your objective is to distribute tasks between the two agents to maximize efficiency. ", model='gemini-1.5-flash')
        with open(file1) as f:
            self.questions_data = [
                {k: v for k, v in row.items()}
                for row in csv.DictReader(f, skipinitialspace=True)
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
        points=None, consistency=1, max_steps_per_goal=5, uniform=False):


        scene_dir = 'datasets/hm3d/val'
        scene = '00891-cvZr5TUy5C5'
        scene_mesh_dir = os.path.join(
                scene_dir, scene, scene[6:] + ".basis" + ".glb"
            )
        self.sim_kwargs['scene_path'] = scene_mesh_dir
        self.sim_kwargs['scene_config'] = f"{scene_dir}/hm3d_train_basis.scene_dataset_config.json"
        self.sim_kwargs['scene_id'] = scene[2:5]
        init_angle = np.random.uniform(0, 2 * np.pi)
        rotation = quat_to_coeffs(
            quat_from_angle_axis(init_angle, np.array([0, 1, 0]))
        ).tolist()
        self.answer_counter = Counter()
        self.annotatedSimulator = AnnotatedSimulator(**self.sim_kwargs)

        self.curr_run_steps = 50
        print(f'BUDGET STEPS FOR THIS RUN: {50}')

        self.run_metadata = {
        'max_steps': 50,
        'q_index': self.q_index,
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
        'uniform': uniform
        }  

        self.annotatedSimulator.priv_actions = False
        self.annotatedSimulator.do_draw_arrows = False #points if draw_arrows else None
        self.annotatedSimulator.do_annotate_image = False
        self.annotatedSimulator.objects_to_annotate = []
        init_pts = self.annotatedSimulator.sim.pathfinder.get_random_navigable_point() #np.array([-8.071849, 0.07216382, 8.127762])
        init_angle = np.random.uniform(0, 2 * np.pi)
        rotation = quat_to_coeffs(
            quat_from_angle_axis(init_angle, np.array([0, 1, 0]))
        ).tolist()
        self.set_state(init_pts, rotation)

        self.curr_run_name = f'{self.q_index}_{self.annotatedSimulator.scene_id}'
        obs0 = self.annotatedSimulator.step([('forward', 0)])
        state = self.get_agent_state(0)
        init_angle = np.random.uniform(0, 2 * np.pi)
        rotation = quat_to_coeffs(
            quat_from_angle_axis(init_angle, np.array([0, 1, 0]))
        ).tolist()            
        self.set_state(state.position, rotation, agent_id = 1)
        obs1 = self.annotatedSimulator.step([('forward', 0)], agent_id=1)             
        return obs0, obs1

    # def run_trajectory(self, inner_loop, log_freq, **run_kwargs):
    #     self.step = 0
    #     self.init_pos = None
    #     self.df = pd.DataFrame({})
    #     self.log_freq = log_freq
    #     obs0, obs1 = self.setup_run(**run_kwargs)
    #     print(f'\n===================STARTING RUN: {self.curr_run_name} ===================\n')
    #     self.chat_history = []
    #     self.plans = {0: None, 1: None}
    #     for _ in range(inner_loop):
    #         self.agent_select = 0
    #         actions0 = self.step_env(obs0)
    #         if actions0 is None:
    #             break
    #         obs0 = self.annotatedSimulator.step(actions0, agent_id=0)
    #         self.agent_select = 1
    #         actions1 = self.step_env(obs1)
    #         if actions1 is None:
    #             break
    #         obs1 = self.annotatedSimulator.step(actions1, agent_id=1)
    #         self.step += 1

    #     self.df.to_pickle(f'logs/{self.outer_run_name}/{self.curr_run_name}/df_results.pkl')
    #     self.vlm.reset()
    #     self.annotatedSimulator.sim.close()
    #     self.get_costs()
    #     print('\n===================RUN COMPLETE===================\n')
    #     multi_agent_gif(f'logs/{self.outer_run_name}/{self.curr_run_name}')
    #     print('saved gif')
    def run_trajectory(self, inner_loop, log_freq, **run_kwargs):
        self.step = 0
        self.init_pos = None
        self.df = pd.DataFrame({})
        self.log_freq = log_freq
        obs0, obs1 = self.setup_run(**run_kwargs)
        print(f'\n===================STARTING RUN: {self.curr_run_name} ===================\n')
        self.chat_history = []
        self.plans = {0: None, 1: None}
        for _ in range(inner_loop):
            out = self.step_env([obs0, obs1])
            if out is None:
                break
            obs0, obs1 = out

            self.step += 1

        self.df.to_pickle(f'logs/{self.outer_run_name}/{self.curr_run_name}/df_results.pkl')
        self.vlm.reset()
        self.annotatedSimulator.sim.close()
        self.get_costs()
        print('\n===================RUN COMPLETE===================\n')
        multi_agent_gif(f'logs/{self.outer_run_name}/{self.curr_run_name}')
        print('saved gif')



    
    
    def step_env(self, obss):

        if self.step >= self.curr_run_steps:
            print("MAX STEPS REACHED")
            return None

        rnge = 1.7 if len(self.annotatedSimulator.sensors) == 1 else 2.2
        spacing = 0.4 if len(self.annotatedSimulator.sensors) == 1 else 0.27
        question = "Find the staircase"
        all_images = []
        all_real_actions = []
        all_points = []
        agent_states = []

        for agent_id in range(2):
            obs = obss[agent_id]
            agent_state = self.get_agent_state(agent_id)
            agent_states.append(agent_state)
            points = []
            for sensor in self.annotatedSimulator.sensors:
                points += self.get_arrow_options(obs[f'depth_sensor_{sensor}'], agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], rnge)
            points = self.select_arrrows(points, spacing)
            all_points.append(points)
            real_actions = {}    
            arrowed_images = []
            for sensor in self.annotatedSimulator.sensors:
                im = obs[f'color_sensor_{sensor}']['image']
                real_actions = self.draw_arrows(points, im, agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], real_actions=real_actions)
                put_text_on_image(im, f"AGENT: {agent_id}", background_color=(255, 255, 255), location='top_center', text_size=1.5, text_thickness=2)              
                arrowed_images.append(im)

            all_images.append(arrowed_images)
            all_real_actions.append(real_actions)
        
        topdown_map0 = self.generate_topdown(all_real_actions, aid=0)
        topdown_map1 = self.generate_topdown(all_real_actions, aid=1)
        topdown_maps = [topdown_map0, topdown_map1]
        prompt = (
            "You are responsible for controlling two agents. Each one has reported a visual observation to you, which has potential actions labeled on it. Each agent can also choose action 0 which turns it completely around "
            f"TASK: {question}\n"
            "As the central controller, your job is to control the agents such that they complete the task as efficiently as possible. This means you should distribute exploration and minimize overlap between where the agents go. "
            "You also have access to a topdown map of the environment. The past trajectory of each agent is labeled with green line segments. There are currently several actions that the agent can take, These are labeled by red lines pointing to a circle that has the action number labeled. The location of the white circle shows exactly where the agent would land if it were to take the labeled action Your job is to select an action for the agents that achieves the goal of exploring the environment as efficiently as possiblel"
            "First, describe the environment around agent0, come up with a high level plan, and select an action for agent0. "
            "Second, describe the environment around agent1, come up with a high level plan, and select an action for agent1."
            "Return the acitons in the following JSON format: {'agent0': <action number for agent 0>, 'agent1': <action_>}. Reminder that action 0 turns the agent completely around. "    
        )
        
        row = {'agent0': -10, 'agent1': -10, 'tokens_generated':0, 'success': 1, 'metadata': self.run_metadata, 
        'speed': 0, 'scene_id': self.annotatedSimulator.scene_id, 'question': question, 
        'model': self.vlm.name, 'input_tokens': 0, 'agent0_location': agent_states[0].position, 'agent1_location': agent_states[1].position}
        try:
            resp, performance = self.vlm.call_chat(self.run_metadata['history'], all_images[0] + all_images[1] + topdown_maps, prompt, add_timesteps_prompt=self.run_metadata['add_timesteps_prompt'], step=self.step)
            self.total_input_tokens += performance['input_tokens']
            self.total_output_tokens += performance['tokens_generated']
        
            metadata = {}
            try:
                resp_dict = self.parse_response(resp)
            except (IndexError, KeyError, TypeError) as e:
                print(e)
                resp_dict = {}
            finally:
                row.update(resp_dict)
        except Exception as e:
            print('GEMINI ERROR', e)

        for agent_id in range(2):
            agent_state = self.get_agent_state(agent_id)

            obs = obss[agent_id]
            copies = []
            for sensor in self.annotatedSimulator.sensors:
                copy = obs[f'color_sensor_{sensor}']['image'].copy()

                self.draw_arrows(all_points[agent_id], copy, agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], chosen_action=row[f'agent{agent_id}'], real_actions=all_real_actions[agent_id])
                copies.append(copy)

            if self.step % self.log_freq == 0 or row['success'] == 0:
                images = [Image.fromarray(im[:, :, 0:3], mode='RGB') for im in all_images[agent_id]]
                copies = [Image.fromarray(im[:, :, 0:3], mode='RGB') for im in copies + [topdown_maps[agent_id]]]
                self.log(prompt, images, resp, row['success'], metadata, copy_images=copies, agent=agent_id)

        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        # if self.run_metadata['mask_thinking'] and row['success'] == 1 and self.run_metadata['history'] > 0:
        #     self.vlm.session.history[-1].parts[0].text = f'\n' + '{"action": ' + str(row["actions"]) + '}'
        actions0 = self.annotatedSimulator.move_choices(row['agent0'], points=list(all_real_actions[0].keys()))
        actions1 = self.annotatedSimulator.move_choices(row['agent1'], points=list(all_real_actions[1].keys()))
        obs0 = self.annotatedSimulator.step(actions0, agent_id=0)
        obs1 = self.annotatedSimulator.step(actions1, agent_id=1)
        return obs0, obs1


    def step_env1(self, obs):
        if self.step >= self.curr_run_steps:
            print("MAX STEPS REACHED")
            return None
        agent_state = self.get_agent_state(self.agent_select)
        question_data = self.questions_data[self.q_index]

        question = question_data["question"]
        choices = [c.split("'")[1] for c in question_data["choices"].split("',")]
        choices.append('I need to explore the environment further')
        answer = question_data["answer"]
        vlm_question = question
        vlm_pred_candidates = ["A", "B", "C", "D", 'E']
        for token, choice in zip(vlm_pred_candidates, choices):
            vlm_question += "\n" + token + "." + " " + choice
        multi = len(self.run_metadata['sensors']) > 1

        points = []
        rnge = 1.7 if len(self.annotatedSimulator.sensors) == 1 else 2.2
        spacing = 0.35 if len(self.annotatedSimulator.sensors) == 1 else 0.27
        
        for sensor in self.annotatedSimulator.sensors:
            points += self.get_arrow_options(obs[f'depth_sensor_{sensor}'], agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], rnge)
        points = self.select_arrrows(points, spacing)
        real_actions = {}    
        arrowed_images = []
        for sensor in self.annotatedSimulator.sensors:
            im = obs[f'color_sensor_{sensor}']['image'].copy()
            real_actions = self.draw_arrows(points, im, agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], real_actions=real_actions)
            arrowed_images.append(im)
        # vlm_question = "What room is the guitar located in?"
        # choices = "\n A. Living Room\n B. Bathroom\n C. Bedroom\n D. None of the above\n E. I need to explore the environment further"
        chat_str = ""
        for agent_id, msg in self.chat_history:
            if agent_id == self.agent_select:
                chat_str += f"YOU: {msg}\n"
            else:
                chat_str += f"OTHER AGENT: {msg}\n"
        images = self.get_sensor_images(obs, convert=False)
        description, perf = self.vlm1.call(images, f"Describe the spatial layout of the environment around you. Specifically note of things that relate to the following question {vlm_question}\n Do not answer the quesiton")
        prompt = f"The task is to answer the following question: {vlm_question}\n. Your visual sensors have reported the following description of the environment: {description}. \nBased on this description of the environment, think about how the agent can move to better answer this question. Return a 1 sentence plan."
        plan, perf = self.llm.call([], prompt)

        # if not self.plans[self.agent_select]:

        # if chat_str == "": 
        #     chat_str = "You are "
        # prompt_question = (
        #     "Your task is to work together with another agent to answer the following question\n"
        #     f"[QUESTION]: {vlm_question + choices}\n"
        #     f"There are {len(real_actions)} arrows superimposed onto your observation{'s' if multi else''}, which represent potential actions. "
        #     "In addition to those on the image, you have the following special actions.\n"
        #     "{\n"
        #     "0: turn completely around, use this when you DONT SEE ANY GOOD ACTIONS, and want fresh observations.\n"
        #     "-1: DONE, you know the answer to the question!\n"
        #     "}\n"
        #     "First, tell me what you see from your current sensor observations, and how they relate to the question. "
        #     f"Second, observe your conversation with the other agent, and think about how you should keep collaborating:\n{chat_str}\n"
        #     f"You can also add to this conversation and send a message to the other agent, which you should only do if you see something extremely relevant to the question. "
        #     f"Lastly, choose an action to take based on your observations and conversation. "
        #     "Return an answer, and action in the JSON: {'answer': <answer_letter>, 'action': <action_number>}. If you want to send a message, follow with the JSON with the text SEND: <your optional message>"
        # )
        # if len(real_actions) == 0:
        #     prompt_question = (
        #         "Your task is to answer the following question based on your observation of the environment\n"
        #         f"[QUESTION]: {vlm_question}\n"
        #         f"You have the following actions.\n"
        #         "{\n"
        #         "0: turn completely around to get fresh observations.\n"
        #         "-1: DONE, you know the answer to the question!\n"
        #         "}\n"
        #         "First, tell me what you see from your current sensor observations, and how they relate to the question. "
        #         "Lastly, return an answer and an action in the format {'answer': <answer_letter>, 'action': <action_number>}. "
        #         "Note you CANNOT GO THROUGH CLOSED DOORS."
        #     )
        
        row = {'actions': -10, 'tokens_generated':0, 'success': 1, 'metadata': self.run_metadata, 'message': None,
        'speed': 0, 'scene_id': self.annotatedSimulator.scene_id, 'question': vlm_question, 'choices': choices, 'answer': None, 'ground_truth': answer,
        'model': self.vlm.name, 'input_tokens': 0, 'agent_location': agent_state.position, 'actions': -10, 'prediction': None}
        vlm2_prompt = ("Observe the actions labeled on the image. "
        "you have the following special action.\n"
        "{\n"
        "0: turn completely around\n"
        # "-1: DONE, you know the answer to the question!\n"
        "}\n"
        f"You should adhere to this high-level plan: {plan} Tell me which actions will best achieve this. Then return it as {{'action': <action_number>}}. "
        )
        row, metadata, resp = self.agent_self_consitency(vlm2_prompt, arrowed_images, row, self.run_metadata['consistency'])    
        # if 'SEND' in resp:
        #     end_ndx = -1
        #     if resp.index('{') > resp.index('SEND'):
        #         end_ndx = resp.index('{')
        #     msg = resp[resp.index('SEND') + 5: end_ndx]
        #     row['message'] = msg
        #     self.chat_history.append((self.agent_select, row['message']))
        # print(f'agent {self.agent_select} action {row["actions"]}, pred {row["answer"]}, ground {answer}')
        metadata['DESC'] = description
        metadata['PLAN'] = plan
        metadata['PREDICTION'] = row['answer']
        metadata['GROUND TRUTH'] = answer
        self.answer_counter.update([row['answer']])
        copies = []
        for i, sensor in enumerate(self.annotatedSimulator.sensors):
            copy = obs[f'color_sensor_{sensor}']['image'].copy()

            self.draw_arrows(real_actions, copy, agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], chosen_action=row['actions'], real_actions=real_actions)
            put_text_on_image(images[i], f"QUESTION: {vlm_question}", background_color=(255, 255, 255), location='top_left', text_size=1.5, text_thickness=2)
            put_text_on_image(copy, f"QUESTION: {vlm_question}", background_color=(255, 255, 255), location='top_left', text_size=1.5, text_thickness=2)
            if row['answer'] and not row['answer'] == 'E':
                #ans = choices[vlm_pred_candidates.index(row['answer'])]
                color = (0, 255, 0) if row['answer'] == answer else (255, 0, 0)
                put_text_on_image(copy, f"{ans}", background_color=color, location='top_right', text_size=2, text_thickness=2)
                put_text_on_image(images[i], f"QUESTION: {question}", background_color=(255, 255, 255), location='top_left', text_size=1.5, text_thickness=2)

            copies.append(copy)


        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        if self.run_metadata['mask_thinking'] and row['success'] == 1 and self.run_metadata['history'] > 0:
            self.vlm.session.history[-1].parts[0].text = f'\n' + '{"action": ' + str(row["actions"]) + '}'

        if self.step % self.log_freq == 0 or row['success'] == 0:
            images = [Image.fromarray(im[:, :, 0:3], mode='RGB') for im in images]
            copies = [Image.fromarray(im[:, :, 0:3], mode='RGB') for im in copies]
            self.log(prompt, images, resp, row['success'], metadata, copy_images=copies)

        if self.answer_counter[row['answer']] == 2 and not row['answer'] in ['E', None]:
            print("STOPPING EARLY, DONE")
            return None

        return self.annotatedSimulator.move_choices(row['actions'], points=list(real_actions.keys()))        
        # return actionsbnn
    def log(self, prompt, images, response, success, metadata, copy_images=[], agent=None):
        if agent is None:
            agent = self.agent_select
        
        path = f'logs/{self.outer_run_name}/{self.curr_run_name}/step{self.step}/agent{agent}'
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

    def plot_trajectory(self):
        plt.figure(figsize=(10, 8))
        for agent_id in range(2):
            agent_location = self.df[f'agent{agent_id}_location']
            coordinates = np.array(agent_location.tolist())
            x = coordinates[:, 2]
            y = coordinates[:, 1]
            z = coordinates[:, 0]

            # Create a scatter plot
            #color = blue for agent_id 0, and yellow for agent-id 1
            color = 'blue' if agent_id == 0 else 'yellow'

            scatter = plt.scatter(x, z, c=color, s=50, label=f'Agent{agent_id} Location')

            # Plot arrows to show the direction of movement
            plt.quiver(x[:-1], z[:-1], x[1:] - x[:-1], z[1:] - z[:-1], angles='xy', scale_units='xy', scale=2, color='gray', alpha=0.5)

            # Label start and end points
            plt.text(x[0], z[0], 'Start', fontsize=12, color='red', ha='right')
            plt.text(x[-1], z[-1], f'Agent{agent_id}', fontsize=12, color='green', ha='left')

            # Add labels and title
        plt.xlabel('X Coordinate')
        plt.ylabel('Z Coordinate')

        plt.title(f'Agents Trajectory Over Time')
        plt.legend()  
        plt.grid(True)
    # plt.show()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        return img
        #return an Image object