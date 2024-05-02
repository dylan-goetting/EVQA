import os
import pdb
import random
from matplotlib import pyplot as plt
import numpy as np
import datetime
import habitat_sim
import cv2
import ast

from llavaAgent import LLaVaAgent
from PIL import Image
from collections import Counter
from utils import *
from vlmAgent import VLMAgent
from annoatedSimulator import AnnotatedSimulator
from torch.utils.tensorboard import SummaryWriter


class SpatialBenchmark:

    def __init__(self, sim_kwargs, vlm_agent):

        self.annotatedSimulator = AnnotatedSimulator(**sim_kwargs)
        self.vlmAgent = vlm_agent
        self.score = {'x_pts': 0, 'y_pts': 0, 'z_pts': 0, 'total_pts': 0, 'possible_pts': 0, 'accuracies': []}

        self.vlm_errors = 0
        self.iterations = 0
        self.efficienty = {'tokens_generated': [], 'durations': []}
        self.headless = sim_kwargs['headless']
        self.run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(f'logs/{self.run_name}')

    def evaluate_vlm(self, context, num_samples, num_objects):

        #agent_state = context['agent_state']
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
        ht = set()
        while len(labels) < num_samples:
            obj1, obj2 = random.sample(obj_wrappers, 2)
            if (obj2['obj'], obj1['obj']) not in ht and (obj1['obj'], obj2['obj']) not in ht:
                labels.append(self.parse_diff_vector(obj2['curr_local_coords'], obj1['curr_local_coords']))
                prompt += f"\n\t{len(labels)}.) Where is the {obj2['obj'].category.name()} in relation to the {obj1['obj'].category.name()}?"

        prompt += ("\nReason through the task and describe the 3d layout of the image you see, and at the very end of your response, output "
                   "a json object in the following example format:"
                   f"\n{{1: ['right', 'above', 'neutral'], 2: ['left', 'neutral', 'in front']}} Make sure there are exactly {num_samples} key-pairs and each key is the number of the question\n")
        im_file = Image.fromarray(image[:, :, 0:3])
        im_file.save(f'logs/{self.run_name}/iter{self.iterations}/image_prompt.png')
        
        
        #print(prompt)
        #print(labels)
        response, performance = self.vlmAgent.call(image, prompt, num_samples)
        predictions = self.parse_response(response)
        
        with open(f'logs/{self.run_name}/iter{self.iterations}/details.txt', 'w') as file:
            file.write(f'[PROMPT]\n{prompt}\n\n')
            file.write(f'[GROUND TRUTH]\n{labels}\n\n')
            file.write(f'[MODEL OUTPUT]\n{response}\n\n')
            file.write(f'[PERFORMANCE]\n{performance}')
        
        #print(predictions, labels, prompt)
        try:
            score = {'x_pts': 0, 'y_pts': 0, 'z_pts': 0, 'total_pts': 0, 'possible_pts': 0}
            for i in range(num_samples):
                for j, axis in enumerate(['x_pts', 'y_pts', 'z_pts']):
                    if labels[i][j] == predictions[i+1][j]:
                        score[axis] += 1
                        score['total_pts'] += 1
                    score['possible_pts'] += 1
                    return score, performance

        except KeyError:
            print("Error parsing VLM response, moving on")
            return None


    def parse_diff_vector(self, obj2, obj1):
        diff_vector = obj2 - obj1
        theta_x = np.rad2deg(np.arctan(obj1[0]/obj1[2])- np.arctan(obj2[0]/obj2[2]))
        answer = []
        if theta_x > 10:
            answer.append('right')
        elif theta_x < -10:
            answer.append('left')
        else:
            answer.append('neutral')

        theta_y = np.rad2deg(np.arctan(obj1[1]/obj1[2]) - np.arctan(obj2[1]/obj2[2]))
        if theta_y > 5:
            answer.append('above')
        elif theta_y < -5:
            answer.append('below')
        else:
            answer.append('neutral')

        ratio_z = diff_vector[2]/min(obj2[2], obj1[2])
        if ratio_z > 0.1:
            answer.append('behind')
        elif ratio_z < -0.1:
            answer.append('in front')
        else:
            answer.append('neutral')
        return answer

    def parse_response(self, response):

        response_dict = ast.literal_eval(response[response.rindex('{'):response.rindex('}')+1])

        return response_dict

    def run(self, num_objects=4, num_samples=3, num_iterations=100):
        for _ in range(num_iterations):
            #pdb.set_trace()
            if self.annotatedSimulator.steps > 0:
                action = self.select_action()
            else:
                action = 'move_forward'
            if action == 'stop':
                break
            
            while True:
                context = self.annotatedSimulator.step(action, num_objects=num_objects)
                if len(context['annotations']) == num_objects:
                    os.mkdir(f'logs/{self.run_name}/iter{self.iterations}')

                    score = self.evaluate_vlm(context, num_samples=num_samples, num_objects=num_objects)
                    self.iterations += 1
                    if score is None:
                        self.vlm_errors += 1
                    else:    
                        for k, v in score[0].items():
                            self.score[k] += v

                    self.efficienty['tokens_generated'].append(score[1]['tokens_generated'])
                    self.efficienty['durations'].append(score[1]['duration'])

                    self.writer.add_scalar('response error rate', self.vlm_errors/self.iterations, self.iterations)
                    self.writer.add_scalar('cum accuracy', self.score['total_pts']/self.score['possible_pts'], self.iterations)
                    self.writer.add_scalar('cum efficiency', sum(self.efficienty['tokens_generated'])/sum(self.efficienty['durations']), self.iterations)
                    break

        accs = []
        for axis in ['x_accuracy', 'y_accuracy', 'z_accuracy']:
            accs.append(self.score[f'{axis[0]}_pts']*3 / self.score['possible_pts'])
            print(axis, self.score[f'{axis[0]}_pts']*3 / self.score['possible_pts'])
        accs.append(self.score['total_pts']/self.score['possible_pts'])
        print('overall accuracy', self.score['total_pts']/self.score['possible_pts'])
        self.writer.add_histogram('accuracy distribution', np.array(self.score['accuracies']), 'auto', max_bins=10)
        self.writer.add_histogram('tokens_generated distribution', np.array(self.efficienty['tokens_generated']), 'auto', max_bins=10)
        self.writer.add_histogram('speed distribution', np.array(self.efficienty['durations']), 'auto', max_bins=10)
        self.writer.add_figure("Final Accuracies", plt.bar(['x_accuracy', 'y_accuracy', 'z_accuracy', 'overall_accuracy'], accs))

        self.annotatedSimulator.sim.close()
        cv2.destroyAllWindows()
        self.writer.close()
        print('\nComplete')

    def select_action(self):
        if self.headless:
            return 'turn_right'
        key = cv2.waitKey(0)
        if key == ord("p"):
            pdb.set_trace()

        action = self.annotatedSimulator.action_mapping.get(key, "move_forward")
        return action


if __name__ == '__main__':

    sim_kwargs = {'scene_path': 'datasets/hm3d/minival/00808-y9hTuugGdiq/y9hTuugGdiq.basis.glb',
                  'scene_config': "datasets/hm3d/minival/hm3d_annotated_minival_basis.scene_dataset_config.json",
                  'resolution': (1080, 1920), 'headless': True}
    
    vlm_kwargs = {'llm': 'mistral', 'size':'7b'}

    vlm_agent = LLaVaAgent(**vlm_kwargs)

    benchmark = SpatialBenchmark(sim_kwargs, vlm_agent)
    benchmark.run(num_objects=2, num_samples=1, num_iterations = 5)
