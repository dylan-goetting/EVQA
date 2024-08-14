import pdb
from collections import Counter
from random import shuffle
import random
from habitat_sim.utils.common import quat_from_angle_axis, quat_to_angle_axis
import habitat_sim
import cv2
import numpy as np
import magnum as mn
from src.utils import *

class AnnotatedSimulator:

    def __init__(self, scene_path, scene_config, resolution=(720, 1280), fov=90, headless=False, show_semantic=False, 
                 verbose=False, scene_id=None, sensors=['center'], random_seed=100):

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
        backend_cfg.random_seed = random_seed

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = []
        self.fov = fov
        
        for sensor in sensors:
            
            sem_cfg = habitat_sim.CameraSensorSpec()
            sem_cfg.uuid = f"semantic_sensor_{sensor}"
            sem_cfg.sensor_type = habitat_sim.SensorType.SEMANTIC
            sem_cfg.resolution = [240, 320]

            sem_cfg.orientation = mn.Vector3([-0.4, sensor, 0])
            agent_cfg.sensor_specifications.append(sem_cfg)
            rgb_sensor_spec = habitat_sim.CameraSensorSpec()
            rgb_sensor_spec.uuid = f"color_sensor_{sensor}"
            rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            rgb_sensor_spec.resolution = self.RESOLUTION
            rgb_sensor_spec.hfov = fov            
            rgb_sensor_spec.orientation = mn.Vector3([-0.4, sensor, 0])
            agent_cfg.sensor_specifications.append(rgb_sensor_spec)

        self.focal_length = calculate_focal_length(fov, self.RESOLUTION[1])
        self.sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(self.sim_cfg)

    def filter_objects(self, sem_image, sensor_state, max_objects=5):
        obj_ids = Counter(sem_image.flatten())
        objects = [self.sim.semantic_scene.objects[i] for i in obj_ids.keys()]
        #shuffle(objects)
        filtered = []
        bad_categories = ['floor', 'wall', 'ceiling', 'Unknown', 'unknown', 'surface']
        counted_categories = Counter([a.category for a in objects])

        for obj in objects:

            if len(filtered) == max_objects:
                break
            if obj.category.name() in bad_categories:
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
                for _, (xp, yp) in filtered:
                    if abs(xp - x_p) < 300 and abs(yp - y_p) < 100:
                        valid = False
                        break
            if not valid:
                continue

            ray = habitat_sim.geo.Ray(sensor_state.position, obj.aabb.center - sensor_state.position)
            max_distance = 100.0  # Set a max distance for the ray
            raycast_results = self.sim.cast_ray(ray, max_distance)
            if raycast_results.has_hits():
                distance = np.linalg.norm(global_to_local(sensor_state.position, sensor_state.rotation, raycast_results.hits[0].point))
                com_distance = np.linalg.norm(local_pt)
                error = abs(distance-com_distance)/distance
                if error > 0.1:
                    continue

            filtered.append([obj, (x_p, y_p)])

        return filtered

    def project_2d(self, local_point):

        point_3d = [local_point[0], -local_point[1], -local_point[2]] #inconsistency between habitat camera frame and classical convention

        x = self.focal_length * point_3d[0] / point_3d[2]
        x_pixel = int(self.RESOLUTION[1] / 2 + x)

        y = self.focal_length * point_3d[1] / point_3d[2]
        y_pixel = int(self.RESOLUTION[0] / 2 + y)
        return x_pixel, y_pixel

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
    
    def move_choices(self, action, points=None):
        if points is not None:
            try:
                action = int(action)
                if action == -1:
                    print("DEFAULTING ACTION")
                    return (['forward' , 0.2],)
                if action <= len(points):
                    mag, theta = points[action-1]
                    return (['rotate', -theta], ['forward', 1.5],)
                elif action == 6:
                    return (['rotate', np.pi],)
                elif action == 7:
                    return (['forward', -1.5],)
                elif action == 8:
                    print('MODEL THINKS DONE')
                    return (['forward' , 0.2],)
            except:
                pass
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
        

        print("DEFAULTING ACTION")
        return (['forward' , 0.2],)
    
    def run_user_input(self, arrows=False, points=None, annotate_image=False):
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
            _ = self.step(actions, num_objects=2, annotate_image=annotate_image, draw_arrows=points)

        self.sim.close()
        cv2.destroyAllWindows()

    
    def move(self, action, magnitude, noise=False):
        assert action in ['forward', 'rotate']
        if noise:
            action = action*random.normalvariate(1, 0.2)

        curr_state = self.sim.get_agent(0).get_state()
        curr_position = curr_state.position
        curr_quat = curr_state.rotation  # Quaternion

        theta, w = quat_to_angle_axis(curr_quat)
        if w[1] < 0:  # Fixing the condition
            theta = 2 * np.pi - theta

        new_agent_state = habitat_sim.AgentState()
        new_agent_state.position = np.copy(curr_position)  # Copy the current position
        new_agent_state.rotation = curr_quat # Initialize with the current rotation

        if action == 'forward':
            local_point = np.array([0, 0, -magnitude])
        
            global_p = local_to_global(curr_position, curr_quat, local_point)
            #global_p = self.sim.pathfinder.snap_point(global_p)

            new_agent_state.position = self.sim.pathfinder.try_step(curr_position, global_p)
            
        elif action == 'rotate':
            new_theta = theta + magnitude
            new_quat = quat_from_angle_axis(new_theta, np.array([0, 1, 0]))
            new_agent_state.rotation = new_quat

        self.sim.get_agent(0).set_state(new_agent_state)
        observations = self.sim.get_sensor_observations()

        return observations

    def step(self, actions, num_objects=4, annotate_image=False, draw_arrows=[], objects_to_annotate=[]):

        if actions == 'r':
            random_point = self.sim.pathfinder.get_random_navigable_point()
            random_yaw = np.random.uniform(0, 2 * np.pi)
            random_orientation = quat_from_angle_axis(random_yaw, np.array([0, 1, 0]))
            agent_state = habitat_sim.AgentState()
            agent_state.position = random_point
            agent_state.rotation = random_orientation
            self.sim.get_agent(0).set_state(agent_state)
            observations = self.sim.get_sensor_observations()
    
        else:
            for a1, a2 in actions:
                observations = self.move(a1, a2)

        agent_state = self.sim.get_agent(0).get_state()
        all_out = {}
        for sensor in self.sensors:
            out = {'annotations': [], 'agent_state': agent_state}
            sem_image = observations[f"semantic_sensor_{sensor}"]

            if objects_to_annotate == []:
                objects = self.filter_objects(sem_image, agent_state.sensor_states[f'color_sensor_{sensor}'],
                                                max_objects=num_objects)
            else:
                objects = []
                for obj_id in objects_to_annotate:
                    obj = self.sim.semantic_scene.objects[obj_id]
                    local_coords = np.round(global_to_local(agent_state.sensor_states[f'color_sensor_{sensor}'].position,
                                                            agent_state.sensor_states[f'color_sensor_{sensor}'].rotation,
                                                            obj.aabb.center), 3)
                    objects.append([obj, local_coords])

            for obj, local_coords in objects:
                if annotate_image:
                    sucess = self.annotate_image(observations[f'color_sensor_{sensor}'], obj_wrapped)
                    if sucess:
                        obj_wrapped = {'obj': obj.category.name(), 'curr_local_coords': local_coords, 'obj_id': obj.id}
                        out['annotations'].append(obj_wrapped)
            if draw_arrows:
                self.draw_arrows(observations[f'color_sensor_{sensor}'], agent_state, agent_state.sensor_states[f'color_sensor_{sensor}'], points=draw_arrows)
                img = observations[f'color_sensor_{sensor}']
                if sensor == 0:
                    name = 'center'
                elif sensor > 0:
                    name = 'left'
                else:
                    name = 'right'
                if len(self.sensors) > 1:
                    cv2.putText(img, f"{name.upper()} SENSOR", (int(img.shape[1]/2 - 100), 70), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 3)

            if not self.headless:
                cv2.imshow(f"RGB View {sensor}", cv2.cvtColor(observations[f'color_sensor_{sensor}'], cv2.COLOR_RGB2BGR))

            if self.show_semantic:
                sem_image_visual = (sem_image % 40) * 255 / 40  # Scale semantic labels to visible range
                sem_image_visual = sem_image_visual.astype(np.uint8)
                cv2.imshow(f"Semantic View  {sensor}", sem_image_visual)
            self.steps += 1

            out['image'] = observations[f'color_sensor_{sensor}']
            all_out[f'color_sensor_{sensor}'] = out
        return all_out

    def get_all_objects(self, unique=True):
        objects = self.sim.semantic_scene.objects
        if unique:
            counted_categories = Counter([a.category for a in objects])
            return [obj for obj in objects if counted_categories[obj.category] == 1]
                
        return objects

    def agent_frame_to_image_coords(self, point, agent_state, camera_state):
        global_point = local_to_global(agent_state.position, agent_state.rotation, point)
        camera_point = global_to_local(camera_state.position, camera_state.rotation, global_point)
        if camera_point[2] > 0:
            return None
        xp, yp = self.project_2d(camera_point)
        return xp, yp

    def draw_arrows(self, rgb_image, agent_state, camera_state, points=None, font_scale=3, font_thickness=4):
        origin_point = [0, 0, 0]
        if points is None:
            points = [(1.75, -np.pi*0.35), (1.5, -np.pi*0.21), (1.5, 0), (1.5, 0.21*np.pi),  (1.75, 0.35*np.pi)]
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (0, 0, 0)  # Black color in BGR for the text
        circle_color = (255, 255, 255)  # White color in BGR for the circle

        start_p = self.agent_frame_to_image_coords(origin_point, agent_state, camera_state)
        action = 1
        for mag, theta in points:
            cart = [mag*np.sin(theta), 0, -mag*np.cos(theta)]
            end_p = self.agent_frame_to_image_coords(cart, agent_state, camera_state)
            if end_p is None:
                action += 1
                continue
            arrow_color = (255, 0, 0)  
            cv2.arrowedLine(rgb_image, start_p, end_p, arrow_color, font_thickness, tipLength=0.05)
            text = str(action)
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            circle_center = (end_p[0], end_p[1])
            circle_radius = max(text_width, text_height) // 2 + 15
            cv2.circle(rgb_image, circle_center, circle_radius, circle_color, -1)
            text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
            cv2.putText(rgb_image, text, text_position, font, font_scale, text_color, font_thickness)
            action += 1
