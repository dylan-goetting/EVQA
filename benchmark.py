import pdb
import random
from collections import Counter

import numpy as np

import habitat_sim
import cv2
from utils import *
from vlmAgent import VLMAgent
from annoatedSimulator import AnnotatedSimulator


class SpatialBenchmark:

    def __init__(self, sim_kwargs, vlm_kwargs):

        self.annotatedSimulator = AnnotatedSimulator(**sim_kwargs)
        self.vlmAgent = VLMAgent(**vlm_kwargs)
        self.score = {'x_pts': 0, 'y_pts': 0, 'z_pts': 0, 'total_pts': 0, 'possible_pts': 0}

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

        prompt += ("\nReason through the task, analyze the 3d layout of the image you see, and at the very end output "
                   "'ANSWER' followed by a json object in the following example format:"
                   "\n{1: ['right', 'above', 'neutral'], 2: ['left', 'neutral', 'in front']}\nDo not generate any "
                   "images, respond purely in text")

        print(prompt)
        print(labels)
        # predictions = self.parse_response(self.vlmAgent.call(image, prompt, num_samples), num_samples)
        predictions = self.vlmAgent.call(image, prompt, num_samples)
        #print(predictions, labels, prompt)
        score = {'x_pts': 0, 'y_pts': 0, 'z_pts': 0, 'total_pts': 0, 'possible_pts': 0}
        for i in range(num_samples):
            for j, axis in enumerate(['x_pts', 'y_pts', 'z_pts']):
                if labels[i][j] == predictions[i+1][j]:
                    score[axis] += 1
                    score['total_pts'] += 1
                score['possible_pts'] += 1

        return score

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

    def parse_response(self, response, num_samples):
        predictions = []
        for i in range(num_samples):
            pred = set()
            for kw in ['right', 'left', 'above', 'below', 'in front', 'behind']:
                if kw in response[i]:
                    pred.add(kw)
            predictions.append(pred)
        return predictions

    def run(self, num_objects=4, num_samples=3):
        while True:
            if self.annotatedSimulator.steps > 0:
                action = self.select_action()
            else:
                action = 'move_forward'
            if action == 'stop':
                break

            context = self.annotatedSimulator.step(action, num_objects=num_objects)
            if len(context['annotations']) == num_objects:

                score = self.evaluate_vlm(context, num_samples=num_samples, num_objects=num_objects)
                for k, v in score.items():
                    self.score[k] += v

        print('overall accuracy', self.score['total_pts']/self.score['possible_pts'])
        for axis in ['x_accuracy', 'y_accuracy', 'z_accuracy']:
            print(axis, self.score[f'{axis[0]}_pts']*3 / self.score['possible_pts'])

        self.annotatedSimulator.sim.close()
        cv2.destroyAllWindows()

    def select_action(self):
        key = cv2.waitKey(0)
        if key == ord("p"):
            pdb.set_trace()

        action = self.annotatedSimulator.action_mapping.get(key, "move_forward")
        return action


if __name__ == '__main__':

    sim_kwargs = {'scene_path': '../habitat-sim/data/scene_datasets/hm3d/minival/00808-y9hTuugGdiq/y9hTuugGdiq.basis.glb',
                  'scene_config': "../habitat-sim/data/scene_datasets/hm3d/minival/hm3d_annotated_minival_basis.scene_dataset_config.json",
                  'resolution': (1080, 1920)}
    benchmark = SpatialBenchmark(sim_kwargs, {})
    benchmark.run(num_objects=5, num_samples=4)
