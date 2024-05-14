import pdb
from collections import Counter
from random import shuffle

from habitat_sim.utils.common import quat_from_angle_axis
import habitat_sim
import cv2
import numpy as np
import magnum as mn

from src.utils import *


class AnnotatedSimulator:

    def __init__(self, scene_path, scene_config, resolution=(720, 1280), fov=90, headless=False, show_semantic=False, verbose=False
                 ):

        self.verbose = verbose
        self.filtered_objects = []
        self.steps = 0
        self.action_mapping = {
            ord('w'): "move_forward",
            ord('a'): "turn_left",
            ord('d'): "turn_right",
            ord('q'): "stop",
            ord('r'): "random"
        }
        self.RESOLUTION = resolution
        self.show_semantic = show_semantic
        self.headless =headless
        if not self.headless:
            cv2.namedWindow("RGB View", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("RGB View", self.RESOLUTION[1], self.RESOLUTION[0])

        if show_semantic:
            cv2.namedWindow("Semantic View", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Semantic View", self.RESOLUTION[1], self.RESOLUTION[0])
        self.scene_id = None
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = scene_path
        backend_cfg.scene_dataset_config_file = scene_config
        backend_cfg.enable_physics = True
        sem_cfg = habitat_sim.CameraSensorSpec()
        sem_cfg.uuid = "semantic"
        sem_cfg.sensor_type = habitat_sim.SensorType.SEMANTIC
        sem_cfg.resolution = [240, 320]

        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = self.RESOLUTION
        rgb_sensor_spec.hfov = fov

        self.focal_length = calculate_focal_length(fov, rgb_sensor_spec.resolution[1])

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [sem_cfg, rgb_sensor_spec]

        self.sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(self.sim_cfg)
        self.camera = self.sim.agents[0]._sensors['color_sensor']

    def filter_objects(self, sem_image, sensor_state, max_objects=5):
        obj_ids = Counter(sem_image.flatten())
        objects = [self.sim.semantic_scene.objects[i] for i in obj_ids.keys()]
        shuffle(objects)
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
            if obj_ids[obj.semantic_id] < 5:
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
                    if abs(xp - x_p) < 100 and abs(yp - y_p) < 30:
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
                if error > 0.15:
                    continue
                #print(f'raytesting to {obj.category.name()}, hit {self.sim.semantic_scene.objects[raycast_results.hits[0].object_id].category.name()}, point {}, local coords are {local_pt}')
                #print(f"[Ray testing]: object {obj.category.name()}, ray distance: {distance}, com distance: {com_distance}, error: {error}")

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
        cv2.circle(img, (x_pixel, y_pixel), 4, (255, 0, 0), -1)
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.85
        font_color = (0, 0, 0)
        font_thickness = 1
        text_size, baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_x = int(x_pixel - text_size[0] // 2)
        text_y = int(y_pixel + text_size[1] + 10)
        rect_top_left = (text_x-3, text_y - text_size[1])  # Top-left corner
        rect_bottom_right = (text_x + text_size[0], text_y + 3)  # Bottom-right corner

        # Draw the rectangle to highlight the background
        cv2.rectangle(img, rect_top_left, rect_bottom_right, (255, 255, 255), -1)
        cv2.putText(img, label, (text_x, text_y), font, font_scale, font_color, font_thickness)

    def run_user_input(self):
        assert not self.headless
        while True:
            if self.steps > 0:
                key = cv2.waitKey(0)
                if key == ord("p"):
                    pdb.set_trace()

                action = self.action_mapping.get(key, "move_forward")
                if action == "stop":
                    break
                _ = self.step(action)
            else:
                _ = self.step('move_forward')

        self.sim.close()
        cv2.destroyAllWindows()

    def step(self, action, num_objects=4, annotate_image=False):
        if action == 'r':

            random_point = self.sim.pathfinder.get_random_navigable_point()
            random_yaw = np.random.uniform(0, 2 * np.pi)
            random_orientation = quat_from_angle_axis(random_yaw, np.array([0, 1, 0]))
            agent_state = habitat_sim.AgentState()
            agent_state.position = random_point
            agent_state.rotation = random_orientation
            self.sim.get_agent(0).set_state(agent_state)
            observations = self.sim.get_sensor_observations()
    
        else:
            observations = self.sim.step(action)

        agent_state = self.sim.get_agent(0).get_state()

        rgb_image = observations["color_sensor"]
        sem_image = observations["semantic"]

        if self.verbose:
            print("Agent position:", agent_state.position)
            # print("Agent rotation:", agent_state.rotation)
        self.filtered_objects = self.filter_objects(sem_image, agent_state.sensor_states['color_sensor'],
                                            max_objects=num_objects)
        out = {'annotations': [], 'agent_state': agent_state}
        for obj, _ in self.filtered_objects:
            local_coords = np.round(global_to_local(agent_state.sensor_states['color_sensor'].position,
                                                    agent_state.sensor_states['color_sensor'].rotation,
                                                    obj.aabb.center), 3)
            obj_wrapped = {'obj': obj.category.name(), 'curr_local_coords': local_coords}
            out['annotations'].append(obj_wrapped)
            if self.verbose:
                print(
                    f"[Notable Objects] Object ID: {obj.semantic_id}, Category: {obj.category.name()}, "
                    f"local coords: {local_coords}")
            if annotate_image:
                self.annotate_image(rgb_image, obj_wrapped)
        if not self.headless:
            cv2.imshow("RGB View", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

        if self.show_semantic:
            sem_image_visual = (sem_image % 40) * 255 / 40  # Scale semantic labels to visible range
            sem_image_visual = sem_image_visual.astype(np.uint8)
            cv2.imshow("Semantic View", sem_image_visual)
        self.steps += 1

        out['image'] = rgb_image
        return out
