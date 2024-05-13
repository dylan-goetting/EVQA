import os
import pdb
import pickle
import random
from matplotlib import pyplot as plt
import numpy as np
import datetime
import habitat_sim
import cv2
import ast
import h5py
import pandas as pd
from src.llavaAgent import LLaVaAgent
from PIL import Image
from collections import Counter
from src.utils import *
from src.vlmAgent import VLMAgent
from src.annoatedSimulator import AnnotatedSimulator
from torch.utils.tensorboard import SummaryWriter


class SpatialBenchmark:

    def __init__(self, sim_kwargs, vlm_agent, offline=True, data_path=None):
        self.offline = offline
        if self.offline:
            data_path = f'annotated_datasets/{data_path}.hdf5'
            self.data_file = h5py.File(data_path, 'r')
            self.dataset = self.data_file['data']

        else:
            self.annotatedSimulator = AnnotatedSimulator(**sim_kwargs)
            self.headless = sim_kwargs['headless']

        self.vlmAgent = vlm_agent
        self.score = {'x_pts': 0, 'y_pts': 0, 'z_pts': 0, 'total_pts': 0, 'possible_pts': 0, 'x_pts_weighted': 0, 'y_pts_weighted': 0, 'z_pts_weighted': 0,
                      'total_pts_weighted': 0, 'possible_pts_weighted': 0, 'accuracies': [], 'accuracies_weighted': []}
        columns = [
            'x_pts', 'x_pts_weighted', 'x_possible_pts_weighted', 'y_pts', 'y_possible_pts_weighted', 'z_pts', 'z_possible_pts_weighted', 
            'accuracy', 'accuracy_weighted', 'tokens_generated', 'num_samples', 'num_objects', 'speed'
        ]
        self.df = pd.DataFrame(columns=columns)
        self.vlm_errors = 0
        self.iterations = 0
        self.efficienty = {'tokens_generated': [], 'durations': []}
        self.run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(f'logs/{self.run_name}')

    def evaluate_vlm(self, context, num_samples, num_objects):

        obj_wrappers = context['annotations']
        image = context['image']
        prompt = ("You are a robot navigating within an 3-D environment as shown. In the image you see, there are "
                  f"{num_objects} labeled objects. You will be asked to analyze the spatial position of these labeled "
                  f"objects with relation to each other. The red dots on each object are the object's center of mass, "
                  f"which you should use when comparing the position of two objects. From your point of view, "
                  f"answer each question with the"
                  f"descriptors right/left, above/below, in front/behind. If there is no clear spatial difference "
                  f"along a given axis, you can answer 'neutral'")
        labels = []
        weights = []
        ht = set()
        while len(labels) < num_samples:
            obj1, obj2 = random.sample(obj_wrappers, 2)
            if (obj2['obj'], obj1['obj']) not in ht and (obj1['obj'], obj2['obj']) not in ht:
                ht.add((obj1['obj'], obj2['obj']))
                l, w = self.parse_diff_vector(obj2['curr_local_coords'], obj1['curr_local_coords'])
                labels.append(l)
                weights.append(self.calculate_weights(*w))
                prompt += f"\n\t{len(labels)}.) Where is the {obj2['obj']} in relation to the {obj1['obj']}?"
        weights = np.array(weights, dtype=object)
        assert weights.shape == (num_samples, 3)
        prompt += ("\nReason through the task and describe the 3d layout of the image you see, and at the very end of your response, output "
                   "a json object in the following example format:"
                   f"\n{{1: ['right', 'above', 'neutral'], 2: ['left', 'neutral', 'in front']}} Make sure there are exactly {num_samples} key-pairs and each key is the number of the question\n")

        response, performance = self.vlmAgent.call(image, prompt, num_samples)
        predictions = self.parse_response(response)
        
        im_file = Image.fromarray(image[:, :, 0:3].astype('uint8'))
        im_file.save(f'logs/{self.run_name}/iter{self.iterations}/image_prompt.png')
        with open(f'logs/{self.run_name}/iter{self.iterations}/details.txt', 'w') as file:
            file.write(f'[PROMPT]\n{prompt}\n\n')
            file.write(f'[GROUND TRUTH]\n{labels}\n\n')
            file.write(f'[MODEL OUTPUT]\n{response}\n\n')
            file.write(f'[PERFORMANCE]\n{performance}')
        
        row = { 'x_pts': 0, 'x_pts_weighted': 0, 'x_possible_pts_weighted':weights[:, 0].sum(),
                'y_pts':0, 'y_pts_weighted': 0, 'y_possible_pts_weighted':weights[:, 1].sum(), 
                'z_pts':0, 'z_pts_weighted':0, 'z_possible_pts_weighted':weights[:, 2].sum(),
                'accuracy':0, 'accuracy_weighted':0, 'tokens_generated':performance['tokens_generated'], 
                'num_samples':num_samples, 'num_objects':num_objects, 'success': 1,
                'speed': performance['tokens_generated']/performance['duration']}
        try:
            for i in range(num_samples):
                for j, axis in enumerate(['x_pts', 'y_pts', 'z_pts']):
                    if i+1 in predictions:
                        key = i+1
                    else:
                        key = str(i+1)
                    if labels[i][j] == predictions[key][j]:
                        row[axis] += 1
                        row[f'{axis}_weighted'] += weights[i][j]
            row['accuracy'] = (row['x_pts'] + row['y_pts'] + row['z_pts'])/(num_samples*3)
            row['accuracy_weighted'] = (row['x_pts_weighted'] + row['y_pts_weighted'] + row['z_pts_weighted'])/(weights.sum())

        except KeyError or IndexError as e:
            print(e)
            row['success'] = 0
            print("Error parsing VLM response, moving on")

        finally:
            row = pd.DataFrame([row])
            self.df= pd.concat([self.df, row], ignore_index=True)
        

    def calculate_weights(self, theta_x, theta_y, ratio_z):

        weight_x = theta_x**2 / (100 + theta_x**2)
        weight_y = theta_y**2/ (100 + theta_y**2)
        weight_z = ratio_z**2/ (0.005 + ratio_z**2)

        return [weight_x, weight_y, weight_z]

    def parse_diff_vector(self, obj2, obj1):
        diff_vector = obj2 - obj1
        theta_x = np.rad2deg(np.arctan(obj1[0]/obj1[2])- np.arctan(obj2[0]/obj2[2]))
        answer = []
        if theta_x >= 0:
            answer.append('right')
        elif theta_x < 0:
            answer.append('left')


        theta_y = np.rad2deg(np.arctan(obj1[1]/obj1[2]) - np.arctan(obj2[1]/obj2[2]))
        if theta_y >= 0:
            answer.append('above')
        elif theta_y < 0:
            answer.append('below')


        ratio_z = diff_vector[2]/min(obj2[2], obj1[2])
        if ratio_z >= 0:
            answer.append('behind')
        elif ratio_z < 0:
            answer.append('in front')

        return answer, [theta_x, theta_y, ratio_z]

    def parse_response(self, response):
        try:
            response_dict = ast.literal_eval(response[response.rindex('{'):response.rindex('}')+1])
        except ValueError:
            response_dict = {}
        return response_dict

    def run(self, num_objects=4, num_samples=3, num_iterations=100):
        try:
            for iter in range(num_iterations):
                
                if self.offline:
                    item = self.dataset[iter]
                    annotations = pickle.loads(item['metadata'])
                    image = item['image']
                    for i in range(num_objects):
                        image = annotate_image_offline(annotations[i], image, item['fov']) 
                    context = {'image': image, 'annotations': annotations[0:num_objects]}

                else:
                    if self.annotatedSimulator.steps > 0:
                        action = self.select_action()
                    else:
                        action = 'move_forward'
                    if action == 'stop':
                        break
                    
                    while True:
                        context = self.annotatedSimulator.step(action, num_objects=num_objects, annotate_image=True)
                        if len(context['annotations']) == num_objects:
                            break
                        else:
                            print('sampling another pose, not enough objects')

                os.mkdir(f'logs/{self.run_name}/iter{self.iterations}')

                self.evaluate_vlm(context, num_samples=num_samples, num_objects=num_objects)    
                self.iterations += 1

        finally:
            print('closing file')
            self.data_file.close()

        accs = []
        pdb.set_trace()
        for axis in ['x_accuracy', 'y_accuracy', 'z_accuracy']:
            accs.append(self.score[f'{axis[0]}_pts']*3 / self.score['possible_pts'])
            print(axis, self.score[f'{axis[0]}_pts']*3 / self.score['possible_pts'])
        accs.append(self.score['total_pts']/self.score['possible_pts'])
        print('overall accuracy', self.score['total_pts']/self.score['possible_pts'])
        self.writer.add_histogram('accuracy distribution', np.array(self.score['accuracies']), bins='auto', max_bins=10)
        self.writer.add_histogram('tokens_generated distribution', np.array(self.efficienty['tokens_generated']), bins='auto', max_bins=10)
        self.writer.add_histogram('speed distribution', np.array(self.efficienty['durations']), bins='auto', max_bins=10)

        fig, ax = plt.subplots()
        ax.bar(['x_accuracy', 'y_accuracy', 'z_accuracy', 'overall_accuracy'], accs)
        ax.set_ylabel('Accuracy')
        ax.set_title('Final Accuracies')

        self.writer.add_figure("Final Accuracies", fig)
        self.annotatedSimulator.sim.close()
        cv2.destroyAllWindows()
        self.writer.close()
        print('\nComplete')

    def select_action(self):
        if self.headless:
            return 'r'
        key = cv2.waitKey(0)
        if key == ord("p"):
            pdb.set_trace()

        action = self.annotatedSimulator.action_mapping.get(key, "move_forward")
        return action

    def visualize_results(self, df):
