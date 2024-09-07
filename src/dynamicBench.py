import os
from sqlite3 import DatabaseError
import pdb
from habitat.utils.visualizations import maps
import numpy as np
import cv2
import ast
import pandas as pd
from PIL import Image
from src.utils import *
from src.vlm import VLM, GPTModel, GeminiModel
import habitat_sim
import cv2


class DynamicBench: 

    task = 'Not defined'

    def __init__(self, sim_kwargs, vlm_agent: VLM, exp_kwargs, outer_run_name):

        self.sim_kwargs = sim_kwargs
        self.vlm = vlm_agent
        # self.map_vlm = GeminiModel('gemini-1.5-pro', 'You are an assistant that specializes in maps. You analyze the map and provide action for the agent to take')
        self.map_vlm = GPTModel('gpt-4o', sys_instruction='You are a 5 time world champion cartographer. You analyze the map you are given and provide action for the agent to take')
        self.answerVLM = GeminiModel(sys_instruction='You are a world champion question answerer. An agent sends you images and queries, and you intelligently respond with the correct answer ', 
                                  model='gemini-1.5-flash')
        # self.answerVLM = GPTModel(sys_instruction='You are a helpful assistant that answers questions about the observations you see', 
        #                           model='gpt-4o')
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
        self.explored_color = (0, 255, 0)
        self.unexplored_color = (200, 200, 200)
        recolor_map = np.array(
        [[40, 40, 40], self.unexplored_color, [0, 0, 0]], dtype=np.uint8)
        topdown_map = recolor_map[topdown_map]

        is_grey = topdown_map == self.unexplored_color
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

        self.unpriv_map = np.zeros((3000, 3000, 3), dtype=np.uint8)
        self.explored_map = np.zeros((3000, 3000, 3), dtype=np.uint8)
        self.scale = 100

        print(f'\n===================STARTING RUN: {self.curr_run_name} ===================\n')
        for _ in range(inner_loop):
            try:
                print('STEP ', self.step)
                actions = self.step_env(obs)
                if actions is None:
                    break
                obs = self.annotatedSimulator.step(actions)
                self.step += 1
            except DatabaseError as e:
                print(e)
                print('ERROR OCCURRED')
                
        self.post_run()
    
    def setup_run(self, **run_kwargs):
        raise NotImplementedError

    def step_env(self, obs):
        raise NotImplementedError

    def post_run_log(self, items=None):
        s = self.df['agent_location']
        pairs = list(zip(s[:-1], s[1:]))

        unpriv_map = self.unpriv_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        unpriv_map[mask] = self.explored_color

        for loc1, loc2 in pairs:
            c1 = self.toGrid(loc1)
            c2 = self.toGrid(loc2)
            cv2.arrowedLine(self.topdown_map, c1, c2, (0, 150, 0), 10)

            c1 = self.toGrid2(loc1)
            c2 = self.toGrid2(loc2)
            cv2.arrowedLine(unpriv_map, c1, c2, (0, 150, 0), 10)

        path = f'logs/{self.outer_run_name}/{self.curr_run_name}/step_FINAL'
        os.makedirs(path, exist_ok=True)
        im = Image.fromarray(self.topdown_map, mode='RGB')
        im.save(f'{path}/final_map.png')
        
        im = Image.fromarray(unpriv_map, mode='RGB')
        im.save(f'{path}/final_map_unpriv.png')

        topdown_map = maps.get_topdown_map_from_sim(self.annotatedSimulator.sim, map_resolution=2048)
        recolor_map = np.array(
        [[40, 40, 40], self.unexplored_color, [0, 0, 0]], dtype=np.uint8)
        topdown_map = recolor_map[topdown_map]
        im = Image.fromarray(topdown_map, mode='RGB')
        im.save(f'{path}/true_topdown.png')

        im = Image.fromarray(self.unpriv_map, mode='RGB')
        im.save(f'{path}/SLAM_topdown.png')

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
                row['actions'] = int(resp_dict['action'])
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
        
    def agent_frame_to_image_coords(self, point, agent_state, sensor_state, resolution=None):
        global_p = local_to_global(agent_state.position, agent_state.rotation, point)
        camera_pt = global_to_local(sensor_state.position, sensor_state.rotation, global_p)
        if camera_pt[2] > 0:
            return None
        return self.annotatedSimulator.project_2d(camera_pt, resolution)

    def get_arrow_options(self, depth_image, agent_state, sensor_state, rnge=1.5, im=None):
        
        if self.run_metadata['uniform']:
            return self.run_metadata['points']
        height_map = depth_to_height1(depth_image, self.annotatedSimulator.fov, sensor_state.position, 
                                      sensor_state.rotation, )
        height_map = abs(height_map - (agent_state.position[1]-0.1)) < 0.15
#         depth_image_normalized = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image)) * 255

# # Convert the normalized depth image to 8-bit unsigned integer
#         depth_image_8bit = depth_image_normalized.astype(np.uint8)

#         mask = im.copy()[:, :, 0:3]
#         mask[height_map] = [0, 0, 255]
#         Image.fromarray(mask, mode='RGB').save('logs/height_mask.png')
#         Image.fromarray(depth_image_8bit).save('logs/depth_image.png')
#         pdb.set_trace()
        arrowData = []
        num_pts = 30 if self.task == 'obj_nav' else 50
        points = [(1, val) for val in np.linspace(-rnge, rnge, num_pts)]
        start = self.agent_frame_to_image_coords([0, 0, 0], agent_state, sensor_state, resolution = depth_image.shape)
        arrowData = []
        for _, theta in points:
            
            arrow = self.get_end_pxl(start, theta, height_map, agent_state, sensor_state, depth_image)
            if arrow is not None:
                arrowData.append(arrow) 
        return arrowData
    

    def get_end_pxl(self, start, theta, height_map, agent_state, sensor_state, depth_image):
        cart = [2*np.sin(theta), 0, -2*np.cos(theta)]
        end = self.agent_frame_to_image_coords(cart, agent_state, sensor_state)
        if end is None or end[1] >= depth_image.shape[0]:
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

        out = (int(x_coords[-1]), int(y_coords[-1]))
        if height_map[int(y_coords[0]), int(x_coords[0])] == False:
            # print('First point is not valid')
            return 0, theta
        
        i = 0
        for i in range(num_points-4):
            y = int(y_coords[i])
            x = int(x_coords[i])
            if sum([height_map[int(y_coords[j]), int(x_coords[j])] for j in range(i, i+4)]) <= 2:
                out = (x, y)
                break
        if i < 5:
            # print('less than 5 pixels')
            return 0, theta
        

        out = (np.clip(out[0], 0, W-1), np.clip(out[1], 0, H-1))
               
        camera_coords = self.annotatedSimulator.unproject_2d(*out, depth_image[out[1], out[0]]) 
        local_coords = global_to_local(agent_state.position, agent_state.rotation, 
                                       local_to_global(sensor_state.position, sensor_state.rotation, camera_coords))   
        mag = np.linalg.norm([local_coords[0], local_coords[2]])
        self.update_topdown(mag, theta, agent_state)
        self.update_unpriv(mag, theta, agent_state)
        # print('found a point:', mag, theta, i)
        return (mag, theta)
        
    def annotate_image(self, agent_state, obs):
        points = []
        rnge = 1.5 if len(self.annotatedSimulator.sensors) == 1 else 2.2
        spacing = 0.4

        for sensor in self.annotatedSimulator.sensors:
            if sensor > 0:
                name = 'left'
            if sensor == 0:
                name = 'center'
            if sensor < 0:
                name = 'right'
            if len(self.annotatedSimulator.sensors) > 1:
                put_text_on_image(obs[f'color_sensor_{sensor}']['image'], f'{name} sensor', text_size=2, location='top_center', text_thickness=2)
            points += self.get_arrow_options(obs[f'depth_sensor_{sensor}'], agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], rnge, im=obs[f'color_sensor_{sensor}']['image'])
        
        points = self.select_arrrows(points, spacing, agent_state)
        real_actions = {}
        for sensor in self.annotatedSimulator.sensors:
            real_actions = self.draw_arrows(points, obs[f'color_sensor_{sensor}']['image'], agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], real_actions=real_actions)
        return real_actions

    def select_arrrows(self, aD, min_angle=0.3, agent_state=None):
        # print(f'There are {len(aD)} actions to choose from')

        explore_factor = self.run_metadata['explore_factor']
        explore = explore_factor > 0
        unique = {}
        for mag, theta in aD:
            if theta in unique:
                unique[theta].append(mag)
            else:
                unique[theta] = [mag]
        arrowData = []
        for theta, mags in unique.items():
            mag = 0.66*min(mags)

            cart = [mag*np.sin(theta), 0, -mag*np.cos(theta)]
            global_coords = local_to_global(agent_state.position, agent_state.rotation, cart)
            grid_coords = self.toGrid(global_coords)
            score = (sum(np.all(self.topdown_map[grid_coords[1]-2:grid_coords[1]+2, grid_coords[0]] == self.explored_color, axis=-1)) + 
                    sum(np.all(self.topdown_map[grid_coords[1], grid_coords[0]-2:grid_coords[0]+2] == self.explored_color, axis=-1)))
            arrowData.append([mag, theta, score<5])

        # print(f'There are now {len(arrowData)} actions to choose from')

        arrowData.sort(key=lambda x: x[1])
        thetas = set()
        out = []
        # print([a[0] for a in arrowData])
        filtered = list(filter(lambda x: x[0] > 0.65, arrowData))

        # print(f'There are now {len(filtered)} actions to choose from')

        filtered.sort(key=lambda x: x[1])
        if filtered == []:
            return []
        if explore:
            f = list(filter(lambda x: x[2], filtered))
            if len(f) > 0:
                longest = max(f, key=lambda x: x[0])
                longest_theta = longest[1]
                smallest_theta = longest[1]
                longest_ndx = f.index(longest)
            
                out.append([min(longest[0], 3), longest[1], longest[2]])
                thetas.add(longest[1])
                
                for i in range(longest_ndx+1, len(f)):
                    if f[i][1] - longest_theta > min_angle:
                        # out.append(f[i])
                        out.append([min(f[i][0], 3), f[i][1], f[i][2]])
                        thetas.add(f[i][1])
                        longest_theta = f[i][1]
                for i in range(longest_ndx-1, -1, -1):
                    if smallest_theta - f[i][1] > min_angle:
                        
                        out.append([min(f[i][0], 3), f[i][1], f[i][2]])
                        thetas.add(f[i][1])
                        smallest_theta = f[i][1]

                # print(f'Found {len(out)} actions that explore')
                for mag, theta, isGood in filtered:
                    if theta not in thetas and min([abs(theta - t) for t in thetas]) > min_angle*explore_factor:
                        out.append((min(mag, 3), theta, isGood))
                        thetas.add(theta)
                # print(f'Ended up with {len(out)} actions in total')

        if len(out) == 0:
            longest = max(filtered, key=lambda x: x[0])
            longest_theta = longest[1]
            smallest_theta = longest[1]
            longest_ndx = filtered.index(longest)
            out.append([min(longest[0], 3), longest[1], longest[2]])
            
            for i in range(longest_ndx+1, len(filtered)):
                if filtered[i][1] - longest_theta > min_angle:
                    out.append([min(filtered[i][0], 3), filtered[i][1], filtered[i][2]])
                    # out.append(filtered[i])
                    longest_theta = filtered[i][1]
            for i in range(longest_ndx-1, -1, -1):
                if smallest_theta - filtered[i][1] > min_angle:
                    out.append([min(filtered[i][0], 3), filtered[i][1], filtered[i][2]])
                    # out.append(filtered[i])
                    smallest_theta = filtered[i][1]

        out.sort(key=lambda x: x[1])
        return [(mag, theta) for mag, theta, _ in out]
    
    def draw_arrows(self, points, rgb_image, agent_state, sensor_state, chosen_action=None, real_actions=None):
        if real_actions is None:
            real_actions = {}
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
                intersections.append((x_at_yh, img_height-1))

        if m is not None:
            y_at_x0 = int(b)  # When x = 0, y = b
            if 0 <= y_at_x0 <= img_height:
                intersections.append((0, y_at_x0))
        
        if m is not None:
            y_at_xw = int(m * img_width + b)  # When x = img_width, y = m * img_width + b
            if 0 <= y_at_xw <= img_height:
                intersections.append((img_width-1, y_at_xw))
        
        if m is not None and m != 0:  # Avoid division by zero for horizontal lines
            x_at_y0 = int(-b / m)  # When y = 0, x = -b / m
            if 0 <= x_at_y0 <= img_width:
                intersections.append((x_at_y0, 0))
        
        if m is None:
            intersections.append((x1, img_height-1))  # Bottom edge
            intersections.append((x1, 0))  # Top edge
        
        if len(intersections) == 2:
            return intersections
        return None

    def update_topdown(self, mag, theta, agent_state, clip=1.75, clip_frac=0.8, goal=None, goal_name=None):
        if goal is not None:
            goal_coords = self.toGrid(goal)
            cv2.circle(self.topdown_map, goal_coords, radius=20, color=(255, 155, 0), thickness=-1)
            cv2.putText(self.topdown_map, goal_name, (goal_coords[0] + 10, goal_coords[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 155, 0), 2)
            return 
        mag = min(clip_frac*mag, clip)
        local_coords = np.array([mag*np.sin(theta), 0, -mag*np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        grid_coords = self.toGrid(global_coords)
        agent_coords = self.toGrid(agent_state.position)

        cv2.line(self.topdown_map, agent_coords, grid_coords, self.explored_color, 50)
        # pass
   
    def update_unpriv(self, mag, theta, agent_state, clip=1.75, clip_frac=0.8):
        agent_coords = self.toGrid2(agent_state.position)
        unclipped = max(mag - 1, 0)
        local_coords = np.array([unclipped*np.sin(theta), 0, -unclipped*np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self.toGrid2(global_coords)
        cv2.line(self.unpriv_map, agent_coords, point, self.unexplored_color, 60)

        clipped = min(clip_frac*mag, clip)
        local_coords = np.array([clipped*np.sin(theta), 0, -clipped*np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self.toGrid2(global_coords)
        cv2.line(self.explored_map, agent_coords, point, self.explored_color, 60)
        

    def toGrid(self, position):
        c = maps.to_grid(position[2], position[0], self.topdown_map.shape[0:2] , self.annotatedSimulator.sim)
        return (c[1], c[0])

    def toGrid2(self, position):
        dx = position[0] - self.init_pos[0]
        dz = position[2] - self.init_pos[2]
        resolution = self.unpriv_map.shape
        x = int(resolution[1]//2 + dx*self.scale)
        y = int(resolution[0]//2 + dz*self.scale)
        return (x, y)

    def generate_unpriv(self, real_actions=None, goal=None, zoom=12):
        agent_state = self.get_agent_state(0)
        agent_coords = self.toGrid2(agent_state.position)

        # if goal is not None:
        #     goal_coords = self.toGrid(goal)
        #     cv2.circle(self.topdown_map, goal_coords, radius=25, color=(255, 255, 0), thickness=-1)
            # cv2.putText(self.topdown_map, 'GOAL', (goal_coords[0] + 10, goal_coords[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  
        
        topdown_map = self.unpriv_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color
        text_size = 1.25
        text_thickness = 1
                
        font = cv2.FONT_HERSHEY_SIMPLEX
        if self.step - self.turned >= 3:
            real_actions[(0.75, np.pi)] = 0
        for (mag, theta), action in real_actions.items():
            local_pt = np.array([mag * np.sin(theta), 0, -mag * np.cos(theta)])
            global_pt = local_to_global(agent_state.position, agent_state.rotation, local_pt)
            act_coords = self.toGrid2(global_pt)

            cv2.arrowedLine(topdown_map, tuple(agent_coords), tuple(act_coords), (255, 0, 0), 5, tipLength=0.05)
            text = str(action) 
            (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
            circle_center = (act_coords[0], act_coords[1])
            circle_radius = max(text_width, text_height) // 2 + 15
            cv2.circle(topdown_map, circle_center, circle_radius, (255, 255, 255), -1)
            text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
            cv2.putText(topdown_map, text, text_position, font, text_size, (0, 0, 0), text_thickness+1)

        cv2.circle(topdown_map, agent_coords, radius=15, color=(255, 0, 0), thickness=-1)
        right = (agent_state.position[0] + zoom, 0, agent_state.position[2])
        right_coords = self.toGrid2(right)
        delta = abs(agent_coords[0] - right_coords[0])
        x, y = agent_coords
        # Calculate crop boundaries
        max_x, max_y = topdown_map.shape[1], topdown_map.shape[0]
        x1 = max(0, x - delta)
        x2 = min(max_x, x + delta)
        y1 = max(0, y - delta)
        y2 = min(max_y, y + delta)
        
        zoomed_map = topdown_map[y1:y2, x1:x2]
        
        return zoomed_map

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


