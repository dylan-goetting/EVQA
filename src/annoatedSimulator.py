import pdb
from collections import Counter
import pickle
from habitat_sim.utils.common import quat_from_angle_axis, quat_to_angle_axis
import habitat_sim
import cv2
import numpy as np
import magnum as mn
import pandas as pd
from src.utils import *

class AnnotatedSimulator:

    def __init__(self, scene_path, scene_config, resolution=(720, 1280), fov=90, headless=False, show_semantic=False, 
                 verbose=False, scene_id=None, sensors=['center'], random_seed=100, goal_image_agent=False, 
                 num_agents=1, height=1.5, pitch=-0.5, slide=False):

        self.scene_id = scene_id
        self.verbose = verbose
        self.steps = 0 
        self.action_mapping = {
            ord('w'): "move_forward",
            ord('a'): "turn_left",
            ord('d'): "turn_right",
            ord('q'): "stop",
            ord('r'): "r"
        }
        self.RESOLUTION = resolution
        self.show_semantic = show_semantic
        self.headless = headless
        self.sensors = sensors
        self.priv_actions = True

        self._objects_to_annotate = None
        self._draw_arrows = False
        self._draw_image_annotations = False
        self._num_objects = 2
        self.goal_image_agent = goal_image_agent
        self.agents = list(range(num_agents))
        self.bad_categories = ['floor', 'wall', 'ceiling', 'Unknown', 'unknown', 'surface', 'beam', 'board', 'door', 'door frame', 'door window']
        self.slide = slide
        if not self.headless:
            for sensor in sensors:
                cv2.namedWindow(f"RGB View {sensor}", cv2.WINDOW_NORMAL)
                cv2.resizeWindow(f"RGB View {sensor}", self.RESOLUTION[1]//3, self.RESOLUTION[0]//3)


        if show_semantic:
            for sensor in sensors:
                cv2.namedWindow(f"Semantic View {sensor}", cv2.WINDOW_NORMAL)
                cv2.resizeWindow(f"Semantic View {sensor}", self.RESOLUTION[1], self.RESOLUTION[0])

        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = scene_path
        backend_cfg.scene_dataset_config_file = scene_config
        backend_cfg.enable_physics = True
        # backend_cfg.random_seed = random_seed

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.radius = 0.17
        agent_cfg.height = 1.41
        agent_cfg.sensor_specifications = []
        self.fov = fov
        self.sem_res = [240, 320]
        self.height = height
        pitch = pitch
        for sensor in sensors:
            sem_cfg = habitat_sim.CameraSensorSpec()
            sem_cfg.uuid = f"semantic_sensor_{sensor}"
            sem_cfg.sensor_type = habitat_sim.SensorType.SEMANTIC
            sem_cfg.resolution = self.sem_res
            sem_cfg.hfov = fov
            sem_cfg.position = mn.Vector3([0, self.height, 0])
            sem_cfg.orientation = mn.Vector3([pitch, sensor, 0])
            agent_cfg.sensor_specifications.append(sem_cfg)

            rgb_sensor_spec = habitat_sim.CameraSensorSpec()
            rgb_sensor_spec.uuid = f"color_sensor_{sensor}"
            rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            rgb_sensor_spec.resolution = self.RESOLUTION
            rgb_sensor_spec.hfov = fov            
            rgb_sensor_spec.position = mn.Vector3([0, self.height, 0])
            rgb_sensor_spec.orientation = mn.Vector3([pitch, sensor, 0])
            agent_cfg.sensor_specifications.append(rgb_sensor_spec)

            depth_sensor_spec = habitat_sim.CameraSensorSpec()
            depth_sensor_spec.uuid = f"depth_sensor_{sensor}"
            depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
            depth_sensor_spec.resolution = self.RESOLUTION 
            depth_sensor_spec.hfov = fov            
            depth_sensor_spec.position = mn.Vector3([0, self.height, 0])
            depth_sensor_spec.orientation = mn.Vector3([pitch, sensor, 0])
            agent_cfg.sensor_specifications.append(depth_sensor_spec)

        self.focal_length = calculate_focal_length(fov, self.RESOLUTION[1])
        agents = [agent_cfg]
        if goal_image_agent:
            goal_cfg = habitat_sim.agent.AgentConfiguration()
            goal_sensor_spec = habitat_sim.CameraSensorSpec()
            goal_sensor_spec.uuid = "goal_sensor"
            goal_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            goal_sensor_spec.resolution = self.RESOLUTION
            goal_sensor_spec.hfov = 100     
            goal_sensor_spec.orientation = mn.Vector3([0, 0, 0])
            goal_sensor_spec.position = mn.Vector3([0, 0, 0])       
            goal_cfg.sensor_specifications = [goal_sensor_spec]
            agents.append(goal_cfg)

        self.sim_cfg = habitat_sim.Configuration(backend_cfg, agents)

        print('right before sim')

        try:
            self.sim = habitat_sim.Simulator(self.sim_cfg)
            print('SIM Initialized!')
        except Exception as e:
            print(e)
            self.sim.close()
            raise SystemError("Could not initialize simulator")
        self.sim.seed(random_seed)

        self.floors = [0]
    
        self.floors.append(self.floors[-1] + 100)

    @property
    def objects_to_annotate(self):
        return self._objects_to_annotate

    @objects_to_annotate.setter
    def objects_to_annotate(self, value):
        self._objects_to_annotate = value

    @property
    def do_draw_arrows(self):
        return self._draw_arrows

    @do_draw_arrows.setter
    def do_draw_arrows(self, value):
        self._draw_arrows = value

    @property
    def do_annotate_image(self):
        return self._draw_image_annotations

    @do_annotate_image.setter
    def do_annotate_image(self, value):
        self._draw_image_annotations = value

    @property
    def num_objects(self):
        return self._num_objects

    @num_objects.setter
    def num_objects(self, value):
        self._num_objects = value

    def filter_objects(self, sem_image, sensor_state, max_objects=5):
        obj_ids = Counter(sem_image.flatten())
        objects = [self.sim.semantic_scene.objects[i] for i in obj_ids.keys()]
        filtered = []
        counted_categories = Counter([a.category for a in objects])

        for obj in objects:

            if len(filtered) == max_objects:
                break
            if obj.category.name() in self.bad_categories:
                continue
            if counted_categories[obj.category] > 1:
                continue
            if obj_ids[obj.semantic_id] < 40:
                continue
            local_pt = global_to_local(sensor_state.position, sensor_state.rotation, obj.aabb.center)
            x_p, y_p = self.project_2d(local_pt)

            if x_p < 0.15 * self.RESOLUTION[1] or x_p > 0.85 * self.RESOLUTION[1]:
                continue
            if y_p < 0.05 * self.RESOLUTION[0] or y_p > 0.95 * self.RESOLUTION[0]:
                continue
            valid = True
            if len(filtered) > 0:
                for _, (xp, yp), _ in filtered:
                    if abs(xp - x_p) < 300 and abs(yp - y_p) < 100:
                        valid = False
                        break
            if not valid:
                continue
            
            if not self.can_see(obj, sensor_state):
                continue

            filtered.append([obj, (x_p, y_p), local_pt])

        return filtered
    
    def can_see(self, obj, sensor_state):
        local_pt = global_to_local(sensor_state.position, sensor_state.rotation, obj.aabb.center)
        ray = habitat_sim.geo.Ray(sensor_state.position, obj.aabb.center - sensor_state.position)
        max_distance = 100.0  # Set a max distance for the ray
        raycast_results = self.sim.cast_ray(ray, max_distance)
        if raycast_results.has_hits():
            distance = np.linalg.norm(global_to_local(sensor_state.position, sensor_state.rotation, raycast_results.hits[0].point))
            com_distance = np.linalg.norm(local_pt)
            error = abs(distance-com_distance)/distance
            if error < 0.1:
                return True
        return False

    def project_2d(self, local_point, resolution=None):
        if resolution is None:
            resolution = self.RESOLUTION

        point_3d = [local_point[0], -local_point[1], -local_point[2]] #inconsistency between habitat camera frame and classical convention
        if point_3d[2] == 0:
            point_3d[2] = 0.0001
        x = self.focal_length * point_3d[0] / point_3d[2]
        x_pixel = int(resolution[1] / 2 + x)

        y = self.focal_length * point_3d[1] / point_3d[2]
        y_pixel = int(resolution[0] / 2 + y)
        return x_pixel, y_pixel

    def unproject_2d(self, x_pixel, y_pixel, depth, resolution=None):
        if resolution is None:
            resolution = self.RESOLUTION
        x = (x_pixel - resolution[1] / 2) * depth / self.focal_length
        y = (y_pixel - resolution[0] / 2) * depth / self.focal_length
        return x, -y, -depth

    def annotate_image(self, img, obj_wrapped):
        x_pixel, y_pixel = self.project_2d(obj_wrapped['curr_local_coords'])
        label = f'{obj_wrapped["obj"]}'
        # Assuming you have an image captured from the sensor
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.85     
        font_color = (0, 0, 0)
        font_thickness = 1
        text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_x = int(x_pixel - text_size[0] // 2)
        text_y = int(y_pixel + text_size[1] + 10)
        rect_top_left = (text_x-3, text_y - text_size[1])  # Top-left corner
        rect_bottom_right = (text_x + text_size[0], text_y + 3)  # Bottom-right corner

        # Check if the annotation is within the image bounds
        if 0 <= x_pixel < img.shape[1] and 0 <= y_pixel < img.shape[0] and 0 <= rect_top_left[0] < img.shape[1] and 0 <= rect_top_left[1] < img.shape[0] and 0 <= rect_bottom_right[0] < img.shape[1] and 0 <= rect_bottom_right[1] < img.shape[0]:
            # Draw the rectangle to highlight the background
            cv2.circle(img, (x_pixel, y_pixel), 4, (255, 0, 0), -1)
            cv2.rectangle(img, rect_top_left, rect_bottom_right, (255, 255, 255), -1)
            cv2.putText(img, label, (text_x, text_y), font, font_scale, font_color, font_thickness)
            return True
        
        return False
    
    
    def run_user_input(self, arrows=False, points=None, annotate_image=False, num_objects=2, objects_to_annotate=[]):
        assert not self.headless
        while True:
            key = cv2.waitKey(0)
            key = chr(key)
            if key == "p":
                pdb.set_trace()
            elif key == 'q':
                break
            actions = self.move_choices(key, points)
            # actual_key = self.action_mapping.get(key, 'move_forward')
            _ = self.step(actions, num_objects=num_objects, annotate_image=annotate_image, draw_arrows=points, objects_to_annotate=objects_to_annotate)

        self.sim.close()
        cv2.destroyAllWindows()

    def search_objects(self, name="", exact=True, mode='name'):

        assert mode in ['name', 'id', 'sem_id']
        if mode == 'name':
            fn = lambda obj: obj.category.name()
        if mode == 'id':
            assert exact
            fn = lambda obj: obj.id
        if mode == 'sem_id':
            assert exact
            fn = lambda obj: obj.semantic_id
        all_objects = self.get_all_objects()
        if exact:
            return [obj for obj in all_objects if fn(obj) == name]
        else:
            return [obj for obj in all_objects if name in fn(obj)]
    
    def moves(self, actions, agent_id=0):

        for action, magnitude in actions:
            self.move(action, magnitude)
            
        observations = self.sim.get_sensor_observations(agent_id)

        return observations
    
    def move(self, action, magnitude, noise=False, agent_id=0):
        assert action in ['forward', 'rotate']
        if magnitude == 0:
            return

        curr_state = self.sim.get_agent(agent_id).get_state()
        curr_position = curr_state.position
        curr_quat = curr_state.rotation  

        theta, w = quat_to_angle_axis(curr_quat)
        if w[1] < 0:  # Fixing the condition
            theta = 2 * np.pi - theta

        new_agent_state = habitat_sim.AgentState()
        new_agent_state.position = np.copy(curr_position)  
        new_agent_state.rotation = curr_quat 

        if action == 'forward':
            local_point = np.array([0, 0, -magnitude])
        
            global_p = local_to_global(curr_position, curr_quat, local_point)
            delta = (global_p - curr_position)/10
            pos = np.copy(curr_position)
            for _ in range(10):
                if self.slide:
                    new_pos = self.sim.pathfinder.try_step(pos, pos + delta)
                else:
                    new_pos = self.sim.pathfinder.try_step_no_sliding(pos, pos + delta)
                pos = new_pos

            new_agent_state.position = pos

            
        elif action == 'rotate':
            new_theta = theta + magnitude
            new_quat = quat_from_angle_axis(new_theta, np.array([0, 1, 0]))
            new_agent_state.rotation = new_quat

        self.sim.get_agent(agent_id).set_state(new_agent_state)


    def get_goal_image(self, goal_position, goal_rotation):
        assert self.goal_image_agent
        new_agent_state = habitat_sim.AgentState()
        new_agent_state.position = goal_position
        new_agent_state.rotation = goal_rotation
        self.sim.get_agent(1).set_state(new_agent_state)
        observations = self.sim.get_sensor_observations(1)

        return observations['goal_sensor']
    

    def step(self, actions, agent_id=0):

        if actions == 'r':
            random_point = self.sim.pathfinder.get_random_navigable_point()
            random_yaw = 2 
            random_orientation = quat_from_angle_axis(random_yaw, np.array([0, 1, 0]))
            agent_state = habitat_sim.AgentState()
            agent_state.position = random_point
            agent_state.rotation = random_orientation
            self.sim.get_agent(agent_id).set_state(agent_state)
            observations = self.sim.get_sensor_observations(agent_id)

        else:
            observations = self.moves(actions, agent_id=agent_id)

    
        agent_state = self.sim.get_agent(agent_id).get_state()
    
        all_out = {'agent_state': agent_state}
        for sensor in self.sensors:
            out = {'annotations': [], 'agent_state': agent_state}
            sem_image = observations[f"semantic_sensor_{sensor}"]
            objects = []
                   
            annotated = 0
            for obj, _, local_coords in objects:
                obj_wrapped = {'obj': obj.category.name(), 'curr_local_coords': local_coords, 'obj_id': obj.semantic_id}
                if self.do_annotate_image:  
                    sucess = self.annotate_image(observations[f'color_sensor_{sensor}'], obj_wrapped)
                    if sucess:
                        out['annotations'].append(obj_wrapped)
                        annotated += 1
                    if annotated >= self.num_objects:
                        break
                else:
                    out['annotations'].append(obj_wrapped)

            if not self.headless:
                cv2.imshow(f"RGB View {sensor}", cv2.cvtColor(observations[f'color_sensor_{sensor}'], cv2.COLOR_RGB2BGR))

            if self.show_semantic:
                sem_image_visual = (sem_image % 40) * 255 / 40  # Scale semantic labels to visible range
                sem_image_visual = sem_image_visual.astype(np.uint8)
                cv2.imshow(f"Semantic View  {sensor}", sem_image_visual)
            self.steps += 1

            out['image'] = observations[f'color_sensor_{sensor}']
            all_out[f'color_sensor_{sensor}'] = out
            all_out[f'depth_sensor_{sensor}'] = observations[f'depth_sensor_{sensor}']

        return all_out

    def get_all_objects(self, filter=True, instances=None):
        if filter:
            objects = [obj for obj in self.sim.semantic_scene.objects if obj.category.name() not in self.bad_categories]
        else:
            objects = self.sim.semantic_scene.objects
        counted_categories = Counter([a.category.name() for a in objects])
        if instances is None:
            return objects
        return [obj for obj in objects if (counted_categories[obj.category.name()] >= instances[0] and counted_categories[obj.category.name()] <= instances[1])]
                
    def get_closest_objects(self, semanic_images, agent_state, num_objects=5):
        for sem_image in semanic_images:
            obj_ids = set(sem_image.flatten())
            objects = [self.sim.semantic_scene.objects[i] for i in obj_ids]
            filtered = []
            for obj in objects:
                if obj.category.name() in self.bad_categories:
                    continue
                distance = np.linalg.norm(obj.aabb.center - agent_state.position)
                filtered.append([obj, distance])
        filtered.sort(key=lambda x: x[1])
        return filtered[0:num_objects]

