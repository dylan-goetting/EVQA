import logging
import math
from habitat.utils.visualizations import maps
import os
import random
from sqlite3 import DatabaseError
import pdb
from habitat.utils.visualizations import maps
import numpy as np
import cv2
import ast
from PIL import Image
from src.annoatedSimulator import AnnotatedSimulator
from src.utils import *
from src.vlm import *
import habitat_sim
import cv2
from src.pivot import PIVOT
import traceback
import wandb
import concurrent.futures


class VLMNav:

    def __init__(self, fov, resolution, sensors, vlm, task, **kwargs):

        self.image_width = resolution[1]
        self.image_height = resolution[0]
        self.fov = fov
        self.resolution = resolution
        self.sensors = sensors
        self.vlm: VLM = vlm
        self.answerVLM = GeminiModel(model=self.vlm.name)
        
        fov_radians = np.deg2rad(fov)
        self.focal_length = (self.image_width / 2) / np.tan(fov_radians / 2)
        assert(task in ['hmon', 'eqa','goat', 'vlnce', 'objnav'])
        self.task = task
        self.config = kwargs
        self.config.update({'fov': fov, 'resolution': resolution, 'sensors': sensors, 'task': task, 'model': self.vlm.name})
        self.pivot = None
        self.depth_est = None
        self.explored_color = (200, 200, 200)
        self.unexplored_color = (0, 255, 0)
        self.scale = 100

        if kwargs['pivot']:
            self.pivot = PIVOT(self.vlm, fov, resolution)
        if kwargs['depth_est']:
            self.floor_mask = FloorMask()

        self.reset()

    
    def reset(self):
        self.unpriv_map = np.zeros((5000, 5000, 3), dtype=np.uint8)
        self.explored_map = np.zeros((5000, 5000, 3), dtype=np.uint8)
        self.interesting_images = {'A': [], 'B': [], 'C': [], 'D': []}
        self.answer_counter = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
        self.topdown_map = None
        self.annotatedSimulator = None
        self.done_calls = [-5]
        self.distances = [0]
        self.errors = [0]
        self.distance_traveled = 0
        self.step = 0
        self.init_pos = None
        self.turned = -3
        self.vlm.reset()
        self.expected_move = 0


    def choose_action(self, obs):
        raise NotImplementedError

    def step_env(self, obs):
        print('Step:', self.step)
        agent_state = obs['agent_state']
        if self.step == 0:
            self.init_pos = agent_state.position 
            self.last_position = agent_state.position
        else:
            dist_traveled = np.linalg.norm(agent_state.position - self.last_position)
            if not self.annotatedSimulator.slide:
                print('Distance traveled:', dist_traveled)
                print('Expected to move:', self.expected_move)
                if self.expected_move == 0:
                    error = 0
                else:
                    error = abs(dist_traveled - self.expected_move)/self.expected_move
                    print('Error %:', error)
                self.errors.append(error)
            self.distances.append(dist_traveled)
            self.distance_traveled += dist_traveled

        row, metadata, real_actions, zoomed_map = self.choose_action(obs)

        row.update(self.config)
        rgb = self.topdown_map
        green = np.sum(rgb == self.explored_color)
        light_grey = np.sum(rgb == self.unexplored_color)
        row['explored'] = green / (green + light_grey)
        row['distance_traveled'] = self.distance_traveled

        if row['success'] == 1 and self.config['history'] > 0:
            self.vlm.session.history[-1].parts[0].text = f'\n' + '{"action": ' + str(row["actions"]) + '}'

        if len(self.errors) >= 3 and sum(self.errors[-3:]) >  2.8 and self.step - self.done_calls[-1] >= 3:
            row['actions'] = 0
            print('forcibly turning around')

        if row['actions'] == 0:
            self.turned = self.step

        if self.config['pivot'] and row['actions'] != -1:
            actions = metadata['pivot_actions']
        else:
            actions = self.move_choices(row['actions'], list(real_actions))
        self.expected_move = 0
        for action in actions:
            if action[0] == 'forward' and action[1] > 0:
                self.expected_move = action[1]

        if self.errors[-1] > 0.5 and not self.annotatedSimulator.slide:
            actions = self.actions_control(actions)

        self.last_position = agent_state.position
        self.step += 1
        return actions, row, metadata, real_actions, zoomed_map

    def actions_control(self, actions):
        if len(actions) != 2:
            return actions
        backtrack = 0.3
        (_, theta_rad), (_, M) = actions
        # Step 1: Calculate the original coordinates

        x_prime = backtrack + M * math.cos(theta_rad)
        y_prime = M * math.sin(theta_rad)
        
        # Step 3: Calculate new magnitude m2 and angle theta2
        m2 = math.sqrt(x_prime**2 + y_prime**2)
        theta2 = math.atan2(y_prime, x_prime)  # Convert back to degrees
        theta2 += np.random.uniform(-0.15, 0.15)

        return [('forward', -backtrack), ('rotate', theta2), ('forward', m2)]

    def get_spend(self):
        return self.vlm.spend + self.answerVLM.spend

    def parse_response(self, response):
        try:
            eval_resp = ast.literal_eval(response[response.rindex('{'):response.rindex('}')+1])
            if isinstance(eval_resp, dict):
                return eval_resp
            else:
                return {'action': list(eval_resp)[0]}
        except (ValueError, SyntaxError):
            logging.error(f'Error parsing response {response}')
            return {'action': -10}

    def agent_self_consitency(self, prompt, images, original_row, consistency):
        action_counter = {}
        num_calls = 0
        while True:
            num_calls += 1
            resp, _ = self.vlm.call_chat(self.config['history'], images, prompt, add_timesteps_prompt=True, step=self.step, ex_type = Exception)

            metadata = {}
            row = original_row.copy()
            try:
                resp_dict = self.parse_response(resp)
                row['actions'] = int(resp_dict['action'])
                if row['actions'] == 0:
                    if self.step - self.turned < 3:
                        row['actions'] = -10

            except (IndexError, KeyError, TypeError, ValueError) as e:
                logging.error(f'Error parsing response {e}')
                print(e)
                row['success'] = 0
            finally:
                row.update(resp_dict)
                metadata['ACTIONS'] = row['actions']
                metadata['PROMPT'] = prompt
                metadata['RESPONSE'] = resp

            if row['actions'] in action_counter:
                action_counter[row['actions']]+= 1
            else:
                action_counter[row['actions']] = 1
            
            if action_counter[row['actions']] == consistency:
                break
            else:
                if row['success']==1:
                    self.vlm.rewind()

        row['num_calls'] = num_calls

        return row, metadata, resp

    def get_sensor_images(self, obs, convert=False):
        ims = [obs[f'color_sensor_{sensor}']['image'] for sensor in self.sensors]
        if convert:
            images = []
            for im in ims:
                if im.shape[-1] == 4:
                    im = im[:, :, 0:3]
                images.append(Image.fromarray(im, mode='RGB'))
            return images
        return ims
    
    def project_2d(self, local_point):

        resolution = self.resolution
        point_3d = [local_point[0], -local_point[1], -local_point[2]] #inconsistency between habitat camera frame and classical convention
        if point_3d[2] == 0:
            point_3d[2] = 0.0001
        x = self.focal_length * point_3d[0] / point_3d[2]
        x_pixel = int(resolution[1] / 2 + x)

        y = self.focal_length * point_3d[1] / point_3d[2]
        y_pixel = int(resolution[0] / 2 + y)
        return x_pixel, y_pixel
    
    def unproject_2d(self, x_pixel, y_pixel, depth):
        resolution = self.resolution
        x = (x_pixel - resolution[1] / 2) * depth / self.focal_length
        y = (y_pixel - resolution[0] / 2) * depth / self.focal_length
        return x, -y, -depth

    def agent_frame_to_image_coords(self, point, agent_state, sensor_state):
        global_p = local_to_global(agent_state.position, agent_state.rotation, point)
        camera_pt = global_to_local(sensor_state.position, sensor_state.rotation, global_p)
        if camera_pt[2] > 0:
            return None
        return self.project_2d(camera_pt)
    
    def get_arrow_options(self, depth_image, agent_state, sensor_state, rnge=1.5, im=None):
        

        # estimated_depth = self.depth_est.call([im])[0][0].numpy()
        # depth_image = cv2.resize(depth_image, (estimated_depth.shape[1], estimated_depth.shape[0]))
        if self.config['depth_est']:
            t =  time.time()
            height_map = self.floor_mask.call(im)
            logging.log(f'Floor mask took {time.time() - t}')
        else:
            height_map = depth_to_height1(depth_image, self.fov, sensor_state.position, 
                                        sensor_state.rotation)
            # pdb.set_trace()
            # height_map2 = depth_to_height1(estimated_depth, self.fov, sensor_state.position, 
            #                               sensor_state.rotation)
            thresh = 0.2
            if self.task == 'vlnce':
                thresh = 0.35
            height_map = abs(height_map - (agent_state.position[1]-0.05)) < thresh
        # height_map2 = abs(height_map2 - (agent_state.position[1]-0.05)) < thresh
#         depth_image_normalized = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image)) * 255

# # Convert the normalized depth image to 8n-bit unsigned integer
#         depth_image_8bit = depth_image_normalized.astype(np.uint8)

        # mask = im.copy()[:, :, 0:3]
        # mask = cv2.resize(mask, (estimated_depth.shape[1], estimated_depth.shape[0]))
        # mask[height_map] = [0, 0, 255]
        # mask2 = im.copy()[:, :, 0:3]
        # mask2 = cv2.resize(mask2, (estimated_depth.shape[1], estimated_depth.shape[0]))
        # mask2[height_map2] = [0, 0, 255]
        # Image.fromarray(mask, mode='RGB').save(f'logs/floor_mask_TRUE{random.random()}.png')
        # Image.fromarray(mask2, mode='RGB').save(f'logs/floor_mask_EST{random.random()}.png')

        # Image.fromarray(depth_image_8bit).save('logs/depth_image.png')
#         pdb.set_trace()

        arrowData = []
        num_pts = 60
        points = [(1, val) for val in np.linspace(-rnge, rnge, num_pts)]
        start = self.agent_frame_to_image_coords([0, 0, 0], agent_state, sensor_state)
        arrowData = []
        for _, theta in points:
            
            arrow = self.get_end_pxl(start, theta, height_map, agent_state, sensor_state, depth_image)
            if arrow is not None:
                arrowData.append(arrow)
        return arrowData
    
    def get_default_arrows(self):

        angle = np.deg2rad(self.fov/2) * 0.7
        if len(self.sensors) > 1:
            angle = abs(self.sensors[0])
        if self.step % 2 == 0:
            default_actions = [(1.7, -angle), (1.7, -angle/4), (1.7, angle/4),  (1.7, angle)]
        else:
            if len(self.sensors) > 1:
                default_actions = [(1.7, -angle), (1.7, -angle/3), (1.7, 0), (1.7, angle/3),  (1.7, angle)]
            else:
                default_actions = [(1.7, -angle), (1.7, 0),  (1.7, angle)]
        default_actions.sort(key=lambda x: x[1])
        return default_actions

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
               
        camera_coords = self.unproject_2d(*out, depth_image[out[1], out[0]]) 
        local_coords = global_to_local(agent_state.position, agent_state.rotation, 
                                       local_to_global(sensor_state.position, sensor_state.rotation, camera_coords))   
        mag = np.linalg.norm([local_coords[0], local_coords[2]])
        # print('found a point:', mag, theta, i)
        return (mag, theta)
        
    def annotate_image(self, obs):
        self.add_noise = False
        agent_state = obs['agent_state']

        points = []
        rnge = np.deg2rad(self.fov/2)
        if len(self.sensors) > 1:
            rnge = np.deg2rad(self.fov/2) + abs(self.sensors[0])

        spacing = self.fov/260

        for sensor in self.sensors:
            if sensor > 0:
                name = 'left'
            if sensor == 0:
                name = 'center'
            if sensor < 0:
                name = 'right'
            if len(self.sensors) > 1:
                put_text_on_image(obs[f'color_sensor_{sensor}']['image'], f'{name} sensor', text_size=2, location='top_center', text_thickness=2)
            points += self.get_arrow_options(obs[f'depth_sensor_{sensor}'], agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], rnge, im=obs[f'color_sensor_{sensor}']['image'])
        
        points = self.select_arrrows(points, spacing, agent_state)
        real_actions = {}
        for sensor in self.sensors:
            real_actions = self.draw_arrows(points, obs[f'color_sensor_{sensor}']['image'], agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], real_actions=real_actions)
        
        if real_actions == {} and (self.step - self.turned < 3):
            logging.info('No actions found and cant turn around')
            points = [(1.4, s) for s in self.sensors] if len(self.sensors) > 1 else [(1.4, -0.8),  (1.4, 0), (1.4, 0.8)]
            points.sort(key=lambda x: x[1])
            for sensor in self.sensors:
                real_actions = self.draw_arrows(points, obs[f'color_sensor_{sensor}']['image'], agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], real_actions=real_actions)
        
        return real_actions

    def select_arrrows(self, aD, min_angle=0.3, agent_state=None):
        

        explore_factor = self.config['explore_factor']

        clip_frac = 0.66
        clip_mag = self.config['clip_mag']

        explore = explore_factor > 0
        unique = {}
        for mag, theta in aD:
            if theta in unique:
                unique[theta].append(mag)
            else:
                unique[theta] = [mag]
        arrowData = []
        for theta, mags in unique.items():
            mag = min(mags)
            self.update_unpriv(mag, theta, agent_state, clip=clip_mag, clip_frac=0.8)
            self.update_topdown(mag, theta, agent_state, clip=clip_mag, clip_frac=0.8)

        if self.config['uniform']:
            return self.config['points']

        topdown_map = self.unpriv_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color
        for theta, mags in unique.items():
            mag = min(mags)
            cart = [0.8*mag*np.sin(theta), 0, -0.8*mag*np.cos(theta)]
            global_coords = local_to_global(agent_state.position, agent_state.rotation, cart)
            grid_coords = self.toGrid2(global_coords)
            score = (sum(np.all((topdown_map[grid_coords[1]-2:grid_coords[1]+2, grid_coords[0]] == self.explored_color), axis=-1)) + 
                    sum(np.all(topdown_map[grid_coords[1], grid_coords[0]-2:grid_coords[0]+2] == self.explored_color, axis=-1)))
            arrowData.append([clip_frac*mag, theta, score<3])

        arrowData.sort(key=lambda x: x[1])
        thetas = set()
        out = []
        filter_thresh = 0.8
        if self.task == 'vlnce':
            filter_thresh = 0.65
        filtered = list(filter(lambda x: x[0] > filter_thresh, arrowData))

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
            
                out.append([min(longest[0], clip_mag), longest[1], longest[2]])
                thetas.add(longest[1])
                for i in range(longest_ndx+1, len(f)):
                    if f[i][1] - longest_theta > (min_angle*0.9):
                        out.append([min(f[i][0], clip_mag), f[i][1], f[i][2]])
                        thetas.add(f[i][1])
                        longest_theta = f[i][1]
                for i in range(longest_ndx-1, -1, -1):
                    if smallest_theta - f[i][1] > (min_angle*0.9):
                        
                        out.append([min(f[i][0], clip_mag), f[i][1], f[i][2]])
                        thetas.add(f[i][1])
                        smallest_theta = f[i][1]

                print(f'Found {len(out)} actions that explore')
                for mag, theta, isGood in filtered:
                    if theta not in thetas and min([abs(theta - t) for t in thetas]) > min_angle*explore_factor:
                        out.append((min(mag, clip_mag), theta, isGood))
                        thetas.add(theta)

        if len(out) == 0:
            longest = max(filtered, key=lambda x: x[0])
            longest_theta = longest[1]
            smallest_theta = longest[1]
            longest_ndx = filtered.index(longest)
            out.append([min(longest[0], clip_mag), longest[1], longest[2]])
            
            for i in range(longest_ndx+1, len(filtered)):
                if filtered[i][1] - longest_theta > min_angle:
                    out.append([min(filtered[i][0], clip_mag), filtered[i][1], filtered[i][2]])
                    longest_theta = filtered[i][1]
            for i in range(longest_ndx-1, -1, -1):
                if smallest_theta - filtered[i][1] > min_angle:
                    out.append([min(filtered[i][0], clip_mag), filtered[i][1], filtered[i][2]])
                    smallest_theta = filtered[i][1]


        if (out == [] or max(out, key=lambda x: x[0])[0] < 0.5) and (self.step - self.turned) < 3:
            print('DEFAULTING ARROW ACTIONS')

            return self.get_default_arrows()
        
        out.sort(key=lambda x: x[1])
        return [(mag, theta) for mag, theta, _ in out]
    
    def draw_arrows(self, points, rgb_image, agent_state, sensor_state, chosen_action=None, real_actions=None):
        scale_factor = rgb_image.shape[0] / 1080

        if real_actions is None:
            real_actions = {}
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (0, 0, 0) 
        circle_color = (255, 255, 255) 
        if chosen_action == -1:
            put_text_on_image(rgb_image, 'MODEL THINKS DONE', text_color=(0, 255, 0), text_size=4*scale_factor, location='center', text_thickness=math.ceil(3*scale_factor), highlight=False)
        start_px = self.agent_frame_to_image_coords([0, 0, 0], agent_state, sensor_state)
        for _, (mag, theta) in enumerate(points):
            text_size = 2.4*scale_factor
            text_thickness = math.ceil(3*scale_factor)
            
            cart = [mag*np.sin(theta), 0, -mag*np.cos(theta)]
            end_px = self.agent_frame_to_image_coords(cart, agent_state, sensor_state)
            if end_px is None:
                continue
            bottom_thresh = 0.1

            if bottom_thresh * rgb_image.shape[1] <= end_px[0] <= 0.95 * rgb_image.shape[1] and 0.05 * rgb_image.shape[0] <= end_px[1] <= 0.95 * rgb_image.shape[0]:
                if (mag, theta) in real_actions:
                    action_name = real_actions[(mag, theta)]
                else:
                    action_name = len(real_actions) + 1
                    real_actions[(mag, theta)] = action_name

                cv2.arrowedLine(rgb_image, tuple(start_px), tuple(end_px), (255, 0, 0), math.ceil(5*scale_factor), tipLength=0.)
                text = str(action_name) 
                (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
                circle_center = (end_px[0], end_px[1])
                circle_radius = max(text_width, text_height) // 2 + math.ceil(15*scale_factor)

                if chosen_action is not None and action_name == chosen_action:
                    cv2.circle(rgb_image, circle_center, circle_radius, (0, 255, 0), -1)
                else:
                    cv2.circle(rgb_image, circle_center, circle_radius, circle_color, -1)
                cv2.circle(rgb_image, circle_center, circle_radius, (255, 0, 0), math.ceil(2*scale_factor))
                text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
                cv2.putText(rgb_image, text, text_position, font, text_size, text_color, text_thickness)
            else:
                pass

        if (self.step - self.turned) >= 3 or self.step == self.turned or chosen_action is not None and chosen_action==0:
            text = '0'
            text_size = 3.1*scale_factor
            text_thickness = math.ceil(3*scale_factor)
            (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
            
            circle_center = (math.ceil(0.05 * rgb_image.shape[1]), math.ceil(rgb_image.shape[0] / 2))
            circle_radius = max(text_width, text_height) // 2 + math.ceil(15*scale_factor)
            if chosen_action is not None and chosen_action==0:
                cv2.circle(rgb_image, circle_center, circle_radius, (0, 255, 0), -1)
            else:
                cv2.circle(rgb_image, circle_center, circle_radius, circle_color, -1)
            cv2.circle(rgb_image, circle_center, circle_radius, (255, 0, 0), math.ceil(2*scale_factor))
            text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
            cv2.putText(rgb_image, text, text_position, font, text_size, text_color, text_thickness)

            cv2.putText(rgb_image, 'TURN AROUND', (text_position[0]//2, text_position[1] + math.ceil(80*scale_factor)), font, text_size*0.75, (255, 0, 0), text_thickness)

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

    def dist2d(self, p1, p2):
        return np.linalg.norm(np.array([p1[0], p1[2]]) - np.array([p2[0], p2[2]]))

    def update_topdown(self, mag, theta, agent_state, clip=1.75, clip_frac=0.8, goal=None, goal_name=None):
        if goal is not None and goal[1] > agent_state.position[1] and goal[1] - agent_state.position[1] < 4:
            goal_coords = self.toGrid(goal)
            cv2.circle(self.topdown_map, goal_coords, radius=20, color=(255, 255, 255), thickness=-1)
            cv2.putText(self.topdown_map, goal_name, (goal_coords[0] + 20, goal_coords[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)
            return 
        mag = min(clip_frac*mag, clip)
        local_coords = np.array([mag*np.sin(theta), 0, -mag*np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        grid_coords = self.toGrid(global_coords)
        agent_coords = self.toGrid(agent_state.position)

        cv2.line(self.topdown_map, agent_coords, grid_coords, self.explored_color, 60)
        # pass
   
    def update_unpriv(self, mag, theta, agent_state, clip=1.75, clip_frac=0.8):
        agent_coords = self.toGrid2(agent_state.position)
        unclipped = max(mag - 0.5, 0)
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

    def toGrid2(self, position, rotation=None):
        dx = position[0] - self.init_pos[0]
        dz = position[2] - self.init_pos[2]
        resolution = self.unpriv_map.shape
        x = int(resolution[1]//2 + dx*self.scale)
        y = int(resolution[0]//2 + dz*self.scale)

        if rotation is not None:
            original_coords = np.array([x, y, 1])
            new_coords = np.dot(rotation, original_coords)
    
            new_x, new_y = new_coords[0], new_coords[1]
            return (int(new_x), int(new_y))

        return (x, y)

    def generate_unpriv(self, real_actions=None, goal=None, zoom=12, agent_state=None, chosen_action=None,):
        agent_coords = self.toGrid2(agent_state.position)
        right = (agent_state.position[0] + zoom, 0, agent_state.position[2])
        right_coords = self.toGrid2(right)
        delta = abs(agent_coords[0] - right_coords[0])

        topdown_map = self.unpriv_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color
        text_size = 1.25
        text_thickness = 1
        x, y = agent_coords
        x1, y1 = self.toGrid2(local_to_global(agent_state.position, agent_state.rotation, [0, 0, -2]))
        angle = np.arctan2(y1 - y, x1 - x) * 180.0 / np.pi  # Convert from radians to degrees
        rotation_angle = 90 + angle
        (h, w) = topdown_map.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

        topdown_map = cv2.warpAffine(topdown_map, rotation_matrix, (w, h))
        agent_coords = self.toGrid2(agent_state.position, rotation = rotation_matrix)
        x, y = agent_coords
        font = cv2.FONT_HERSHEY_SIMPLEX
        if self.step - self.turned >= 3:
            real_actions[(0.75, np.pi)] = 0
        for (mag, theta), action in real_actions.items():
            local_pt = np.array([mag * np.sin(theta), 0, -mag * np.cos(theta)])
            global_pt = local_to_global(agent_state.position, agent_state.rotation, local_pt)
            act_coords = self.toGrid2(global_pt, rotation = rotation_matrix)

            cv2.arrowedLine(topdown_map, tuple(agent_coords), tuple(act_coords), (255, 0, 0), 5, tipLength=0.05)
            text = str(action) 
            (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
            circle_center = (act_coords[0], act_coords[1])
            circle_radius = max(text_width, text_height) // 2 + 15
            if chosen_action is not None and action == chosen_action:
                cv2.circle(topdown_map, circle_center, circle_radius, (0, 255, 0), -1)
            else:
                cv2.circle(topdown_map, circle_center, circle_radius, (255, 255, 255), -1)
            text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
            cv2.circle(topdown_map, circle_center, circle_radius, (255, 0, 0), 1)
            cv2.putText(topdown_map, text, text_position, font, text_size, (0, 0, 0), text_thickness+1)

        cv2.circle(topdown_map, agent_coords, radius=15, color=(255, 0, 0), thickness=-1)

        max_x, max_y = topdown_map.shape[1], topdown_map.shape[0]
        x1 = max(0, x - delta)
        x2 = min(max_x, x + delta)
        y1 = max(0, y - delta)
        y2 = min(max_y, y + delta)
        
        zoomed_map = topdown_map[y1:y2, x1:x2]
        return zoomed_map

    def move_choices(self, action, points=None):
        if points is not None:
            try:
                action = int(action)
            except:
                action = -10
            if action == -10:
                print("DEFAULTING ACTION")
                return (['forward' , 0.2],)
            
            if action == -1:
                return (['forward', 0],)
            if action <= len(points) and action > 0:
                mag, theta = points[action-1]
                return (['rotate', -theta], ['forward', mag],)
            if action == 0:
                return (['rotate', np.pi],)

        if action == 'w':
            return [('forward', 0.2)]
        elif action == 'a':
            return [('rotate', np.pi/16)]
        elif action == 's':
            return [('forward', -0.2)]
        elif action == 'd':
            return [('rotate', -np.pi/16)]
        elif action == 'r':
            return 'r'
        elif action == 'l':
            return 'l'

        print("DEFAULTING ACTION")
        return (['forward' , 0.2],)
    
class NavAgent(VLMNav):

    def choose_action(self, obs):
        agent_state = obs['agent_state']
        # return super().choose_action(obs)
        raw_images = [obs[f'color_sensor_{i}']['image'][:, :, 0:3].copy() for i in self.sensors]
        real_actions = self.annotate_image(obs)

        zoomed_map = self.generate_unpriv(real_actions, zoom=9, agent_state=agent_state)


        multi = len(self.annotatedSimulator.sensors) > 1
        prompt = (
        # f"First, analyze your updated camera observation and tell me the spatial layout of what you see. "
        f"I have lost my diamond ring! Your task is to search every inch of this floor to help me find it. It is a huge, bright diamond that is unmistakable. "
        f"There are {len(real_actions) - 1} red arrows superimposed onto your observation{'s' if multi else''}, which represent potential actions. " 
        f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or at a dead end .' if self.step - self.turned >= 3 else ''}"
        #f"Your task is to navigate to the {self.curr_target.upper()}. Think of a high level plan on how you can reach the {self.curr_target.upper()} from where you are now. If you have already reached the {self.curr_target.upper()} choose special action -1 (done). "
        f"First, tell me what you see in your sensor observations. Then, tell me a high level plan on how you will find the ring and where you will go next. {'Recall your past observations so that you dont waste time exploring the same locations. ' if self.config['history'] > 0 else ''}"
        f"Think about how each action will move you. Lastly, select one action from the image and explain how it helps you reach your goal. Return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS"
        )

        images = self.get_sensor_images(obs, convert=False)
        if self.config['use_map'] > 0:
            prompt = (

            f"I have lost my diamond ring! Your task is to search every inch of this floor to help me find it. It is a huge, bright diamond that is unmistakable. "
        
           "\nTo help you with exploration, you have a topdown map of the environment, with unexplored area shown in GREEN, and it shows where you have been in the past, in GRAY. Your current location is shown by a RED dot. "
            f"You have {len(real_actions) - 1} actions, which are red arrows and white circles, and show the location you would move to if you took that action number. Use the map to plan actions that will help you reach these unexplored areas (GREEN). Note you will sometimes need to backtrack through gray areas to reach new rooms. "
            f"{'NOTE: choose action 0 if you want to TURN AROUND or at a dead end .' if (self.step - self.turned >= 3) and (self.config['use_map'] != 2) else ''}"
            f'{"First, describe the map you see, as well the sensor image. Tell me a brief high level plan on how you will find the ring and where you will reach unexplored areas. " if self.config["use_map"] != 2 else "First, tell me the map you see, and your plan for how you will explore more areas"}'
            f'Lastly, use {"both the map and your sensor observation" if self.config["use_map"] != 2 else "the map"} to select the best action and explain how it helps you reach your goal. '
            f"Return it as {{'action': <action_number>}}"
            )
            if self.config['use_map'] == 2:
                images = [zoomed_map]
            else:
                images.append(zoomed_map)

        
        row = {'actions': -10, 'success': 1, 'config': self.config,
        'speed': 0,  'pivot': 1 if self.pivot is not None else 0,
        'model': self.vlm.name, 'input_tokens': 0, 'agent_location': agent_state.position}
        logs = []
        if self.pivot is not None:
            instruction = f'I have lost my diamond ring! Your task is to search every inch of this floor to help me find it. It is a huge, bright diamond that is unmistakable.'
            pivot_actions, log_images = self.pivot.run(raw_images[0], obs['depth_sensor_0'], instruction, agent_state, agent_state.sensor_states['color_sensor_0'])
            metadata = {'pivot_actions': pivot_actions}
            logs += log_images

        else:
            row, metadata, _ = self.agent_self_consitency(prompt, images, row, self.config['consistency'])

        return row, metadata, real_actions, zoomed_map
    

class GOATAgent(VLMNav):

    def choose_action(self, obs):
        agent_state = obs['agent_state']
        goal = obs['goal']
        multi = len(self.sensors) > 1
        goal_ims = []
        raw_images = [a.copy() for a in self.get_sensor_images(obs, convert=False)]

        if goal['mode'] == 'object':
            t = f'Navigate to the nearest {goal["name"]}'
            inst = f'Find the nearest {goal["name"]} and navigate as close as you can to it. '
            inst2 =  f'Tell me which room you would find this {goal["name"]} in? Do you see any {goal["name"]} in your current observations?' 
        if goal['mode'] == 'description':
            inst = f"Find and navigate to the {goal['lang_desc']}. Navigate as close as you can to it "
            t = inst
            inst2 =  f"Tell me which room you would find this specifc {goal['name']} in, {'and which sensor looks the most promising. ' if multi else 'and which general direction you should go in. '}  {' Remember you can always turn around to search in a different area' if self.step - self.turned >= 3 else ''} "
        if goal['mode'] == 'image':
            t = f'Navigate to the specific {goal["name"]} shown in the image labeled GOAL IMAGE. Pay close attention to the details, and note you may see the object from a different angle than in the goal image. Navigate as close as you can to it '
            inst = f"Observe the image labeled GOAL IMAGE. Find this specific {goal['name']} shown in the image and navigate to it. "
            inst2 =  f"Tell me which room you would find this {goal['name']} in, {'and which sensor looks the most promising. ' if multi else 'and which general direction you should go in. '}  {' Remember you can always turn around to search in a different area' if self.step - self.turned >= 3 else ''}"
            goal_im = obs['goal_image']
            put_text_on_image(goal_im, f"GOAL IMAGE: {goal['name']}", background_color=(255, 255, 255), location='top_center')
            goal_ims.append(goal_im)

        def done_thread():
            answer_prompt = (f"The agent has the following navigation task: {t}. The agent has sent you an image taken from its current location{' as well as the goal image. ' if goal['mode'] == 'image' else '. '} "
                                f'Your job is to determine whether the agent is VERY CLOSE to the specified {goal["name"]}'
                                f"First, tell me what you see in the image, and tell me if there is a {goal['name']} that matches the description. Then, return 1 if the agent is very close to the {goal['name']}, and 0 if it isn't. Format your answer in the json {{'done': <1 or 0>}}")

            def process_image(image):
                r, p = self.answerVLM.call([image] + goal_ims, answer_prompt)
                dct = self.parse_response(r)
                if 'done' in dct and int(dct['done']) == 1:
                    return 1, r
                return 0, r

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(process_image, image): image for image in raw_images}

                for future in concurrent.futures.as_completed(futures):
                    isDone, r = future.result()
                    if isDone:
                        return True, r
            return False, r


        def preprocessing_thread():
            images = []
            real_actions = self.annotate_image(obs)
            images += self.get_sensor_images(obs, convert=False)
            zoomed_map = self.generate_unpriv(real_actions, zoom=9, agent_state=agent_state)
            if self.config['use_map']:
                images.append(zoomed_map)       

            return real_actions, images, zoomed_map
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future1 = executor.submit(preprocessing_thread)
            future2 = executor.submit(done_thread)

            real_actions, images, zoomed_map = future1.result() 
            done, r = future2.result()   

        if done:
            print('Model called done')
            self.done_calls.append(self.step)

            i = 0
            real_actions = {}
            for sensor in self.sensors:
                real_actions = self.draw_arrows(self.get_default_arrows(), raw_images[i],  agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], real_actions=real_actions)
                obs[f'color_sensor_{sensor}']['image'] = raw_images[i]
                i += 1

            images = raw_images
        
        
        prompt = (

        f"TASK: {inst} use your prior knowledge about where items are typically located within a home. "
        f"There are {len(real_actions) - 1} red arrows superimposed onto your observation{'s' if multi else''}, which represent potential actions. " 
        f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS.' if self.step - self.turned >= 3 else ''}"
        f"First, tell me what you see in each of your sensor observations, and if you have any leads on finding the {goal['name']}. Second, {inst2}. "
        f"Lastly, explain which action is the best and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
        )

        if self.config['use_map'] and self.step - self.done_calls[-1] >= 1:
            prompt = (
        f"TASK: {inst} use your prior knowledge about where items are typically located within a home. "
        f"There are {len(real_actions) - 1} red arrows superimposed onto your observation{'s' if multi else''}, which represent potential actions. " 
        f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS .' if self.step - self.turned >= 3 else ''}"
        "\nYou also have a topdown map of the environment, with unexplored area shown in GREEN. This map shows you where you have been in the past, shown in GRAY "
        "The same actions you see superimposed on the RGB image are also shown on the top-down map. "
        f"First, tell me what you see in each of your sensor observations, and if you have any leads on finding the {goal['name']}. Second, {inst2}. "
        "If you are not sure where to go, analyze the map to help find actions that lead to new rooms (GREEN AREAS). "
        f"Lastly, use both sources of information and explain which action is the best and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
            )
        
        row = {'actions': -10, 'success': 1, 'metadata': self.config, 'goal': goal['name'], 'goal_mode': goal['mode'], 'model': self.vlm.name, 'agent_location': agent_state.position, 'called_done': done}
        if len(self.done_calls) >= 2 and self.done_calls[-2] == self.step-1:
            row['actions'] = -1
            metadata = {}

        else:
            if self.pivot is not None:
                instruction = inst
                pivot_actions, _ = self.pivot.run(raw_images[0], obs['depth_sensor_0'], instruction, agent_state, agent_state.sensor_states['color_sensor_0'], goal_image = goal_ims[0] if len(goal_ims) > 0 else None)
                metadata = {'pivot_actions': pivot_actions}
                # logs += log_images
                row['actions'] = -10
            else:
                row, metadata, _ = self.agent_self_consitency(prompt, images + goal_ims, row, self.config['consistency'])
        metadata['DONE RESP'] = r
        return row, metadata, real_actions, zoomed_map
    
    def reset_goal(self):
        self.done_calls = [self.step-2]
        self.explored_map = np.zeros_like(self.explored_map)
        self.distance_traveled = 0
        self.distances = [0]
        self.errors = [0]
        self.turned = self.step - 3
        self.expected_move = 0


class HMONAgent(VLMNav):

    def choose_action(self, obs):
        agent_state = obs['agent_state']
        raw_images = [a.copy() for a in self.get_sensor_images(obs, convert=False)]
        goal = obs['goal']
        multi = len(self.annotatedSimulator.sensors) > 1

        def done_thread():
            answer_prompt = (f"The agent has has been tasked with navigating to a {goal.upper()}. The agent has sent you an image taken from its current location. "
                             f'Your job is to determine whether the agent is VERY CLOSE to a {goal}. Note a chair is distinct from a sofa which is distinct from a bed. '
                             f"First, tell me what you see in the image, and tell me if there is a {goal}. Second, return 1 if the agent is very close to the {goal} - make sure the object you see is ACTUALLY a {goal}, Return 0 if if there is no {goal}, or if it is very far away, or if you are not sure. Format your answer in the json {{'done': <1 or 0>}}")

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
                        return True, r

            return False, r
        
        def preprocessing_thread():
            images = []
            real_actions = self.annotate_image(obs)
            images += self.get_sensor_images(obs, convert=False)

            zoomed_map = self.generate_unpriv(real_actions, zoom=9, agent_state=agent_state)
            if self.config['use_map']:
                images.append(zoomed_map)

            return real_actions, images, zoomed_map,
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future1 = executor.submit(done_thread)
            future2 = executor.submit(preprocessing_thread)
            
            real_actions, images, zoomed_map  = future2.result()
            done, r = future1.result()

        if done:
            print('Model called done')
            self.done_calls.append(self.step)
            i = 0
            real_actions = {}
            for sensor in self.sensors:
                real_actions = self.draw_arrows(self.get_default_arrows(), raw_images[i],  agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], real_actions=real_actions)
                obs[f'color_sensor_{sensor}']['image'] = raw_images[i]
                i += 1

            images = raw_images
            
        prompt = (

        f"TASK: NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
        f"There are {len(real_actions) - 1} red arrows superimposed onto your observation{'s' if multi else''}, which represent potential actions. " 
        f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step - self.turned >= 3 else ''}"
        f"First, tell me what you see in your sensor observations, and if you have any leads on finding the {goal.upper()}. {'Second, tell me which sensor looks the most promising. ' if multi else 'Second, tell me which general direction you should go in. '}"
        f"Lastly, explain which action acheives that best, and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
        )

        if self.config['use_map'] and self.step - self.done_calls[-1] >= 1:
            prompt = (
        f"TASK: NAVIGATE TO THE NEAREST {goal.upper()} and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
        f"There are {len(real_actions) - 1} red arrows superimposed onto your observation{'s' if multi else''}, which represent potential actions. " 
        f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step - self.turned >= 3 else ''}"
        "\nYou also have a topdown map of the environment, with unexplored area shown in GREEN. This map shows you where you have been in the past, shown in GRAY. "
        "The same actions you see superimposed on the RGB image are also shown on the top-down map. "
        f"First, tell me what you see in your sensor observations, and if you have any leads on finding the {goal.upper()}. {'Second, tell me which sensor looks the most promising. ' if multi else 'Second, tell me which general direction you should go in. '} {' Remember you can always turn around to search in a different area' if self.step - self.turned >= 3 else ''}"
        "If you are not sure where to go, analyze map to help you plan actions that lead towards the GREEN AREAS. "
        f"Lastly, combine both sources of informaiton and explain which action is the best and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
            )

        row = {'actions': -10, 'success': 1, 'metadata': self.config, 'object': goal, 'distance_traveled': self.distance_traveled, 'pivot': 1 if self.pivot is not None else 0,
        'model': self.vlm.name, 'agent_location': agent_state.position, 'called_done': done}

        if len(self.done_calls) >= 2 and self.done_calls[-2] == self.step-1:
            row['actions'] = -1
            metadata = {}
        else:
            if self.pivot is not None:
                instruction = f"TASK: NAVIGATE TO THE NEAREST {goal.upper()} and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
                pivot_actions, _ = self.pivot.run(raw_images[0], obs['depth_sensor_0'], instruction, agent_state, agent_state.sensor_states['color_sensor_0'])
                print(pivot_actions)
                metadata = {'pivot_actions': pivot_actions}
                row['actions'] = -10
            else:
                row, metadata, _ = self.agent_self_consitency(prompt, images, row, self.config['consistency'])

        metadata['DONE RESP'] = r
        # if done:
        #     real_actions = {(min(1, key[0]), key[1]): value for key, value in real_actions.items()}
        return row, metadata, real_actions, zoomed_map

class EQAAgent(VLMNav):

    def choose_action(self, obs):
        agent_state = obs['agent_state']
        self.question_data = obs['question_data']
        question = self.question_data["question"]
        choices = [c.split("'")[1] for c in self.question_data["choices"].split("',")]

        vlm_question = question
        vlm_pred_candidates = ["A", "B", "C", "D"]
        for token, choice in zip(vlm_pred_candidates, choices):
            vlm_question += "\n" + token + "." + " " + choice
        multi = len(self.config['sensors']) > 1
 
        self.vlm_question = vlm_question
        raw_images = [obs[f'color_sensor_{i}']['image'].copy() for i in self.sensors]
        
        def answer_thread():
            es = ["A", "B", "C", "D", "E"]
            extra = "\n" + es[len(choices)] + "." + " " + "I need the agent to move to a better location in order to answer the question. "
            pred = 'E'
            answer_prompt = (f"The agent has sent you an image, and is asking you the following question [QUESTION]: {vlm_question+extra}\n\n[TASK]: First, tell me where you are in the environmemt, and if there are any notable objects that are relevant to the question. Second, explain what the answer is and why. Lastly, tell it to me as a JSON like {{'answer': <answer letter>}} Pay close attention to the specific details in the question, note you might not be able to answer the question from the image. For example if the quesiton asks about where something is, and you dont see it in the current image, choose answer E because the agent still needs to check the other places. ")
            res = None 

            def process_image(image):
                r, p = self.answerVLM.call([image], answer_prompt)
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
            row = {'actions': -10, 'success': 1, 'metadata': self.config, 'cum_answer': None, 'question': question, 'choices': choices, 'answer': None, 'pivot': 1 if self.pivot is not None else 0,
            'model': self.vlm.name, 'agent_location': agent_state.position, 'actions': -10, 'prediction': None}
    
            real_actions = self.annotate_image(obs) 

            images = self.get_sensor_images(obs, convert=False)
            
            zoomed_map = [self.generate_unpriv(real_actions, zoom=9, agent_state=agent_state)]
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

            if self.config['use_map']:
                prompt_question = (
                "Your task is to navigate throughout the environment and learn the answer to the following quesiton\n"
                f"[QUESTION]: {vlm_question}\n"
                f"There are {len(real_actions) - 1} red arrows superimposed onto your observation{'s' if multi else''}, which represent potential actions. " 
                f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or are at a dead end .' if self.step - self.turned >= 3 else ''}"
    
                "\nYou also have a topdown map of the environment, with unexplored area shown in GREEN. This map shows you where you have been in the past, shown in GRAY. "
                "The same actions you see superimposed on the images are also shown on the topdown map. "
                
                "First, tell me what you see from your current sensor observations and if there are any notable objects that are relevant to the question. "
                "Second, tell me what room or location you should navigate to in order to answer the question, and which general direction you should go. "
                "Lastly, explain which is the best action and return it in the format {'action': <action_number>}. Don't answer the question, just return an action. "
                "Note you CANNOT GO THROUGH CLOSED DOORS, and you do not need to go UP OR DOWN STAIRS."
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
                row['pivot_actions'] = pivot_actions
                return row, metadata, resp, zoomed_map+logs, real_actions
            
            return *self.agent_self_consitency(prompt_question, images, row ,1), zoomed_map, real_actions
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future2 = executor.submit(action_thread)
            future1 = executor.submit(answer_thread)
            
            row, metadata, resp, zoomed_map, real_actions = future2.result()
            pred, answer_mdata = future1.result()
            row['answer'] = pred    
            row['cum_answer'] = max(self.answer_counter, key=self.answer_counter.get)

        metadata.update(answer_mdata)
        return row, metadata, real_actions, zoomed_map
    
    def final_answer(self):
        self.finalAnswerVlm = GeminiModel(model='gemini-1.5-pro-002')
        def final_answer_thread():
            images = []
            for k, v in self.interesting_images.items():
                if len(v) > 0:
                    images += random.choices(v, k=min(1, len(v)))
                if len(images) > 2:
                    break
            
            answer_prompt = (f"The agent has sent you {len(images)} images from the SAME environment, and is asking you the following question about the environment [QUESTION]: {self.vlm_question}\n\n [TASK]: First, tell me what you see in each image, and if there any notable objects that are relevant to the question.  Second, tell me what the best answer choice is and why. Lastly, tell it to me as JSON like {{'answer': <answer letter>}} ")

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
        num_parrallel = 7
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(final_answer_thread) for _ in range(num_parrallel)]

            for future in concurrent.futures.as_completed(futures):
                images, answer, r = future.result()
                if answer:
                    final_counter[answer].append([images, r])
        
        return final_counter
