import json
import numpy as np
import yaml
import argparse
import os
import sys
sys.path.insert(0, '..')
# print("Current Working Directory:", os.getcwd())
from src.annoatedSimulator import AnnotatedSimulator
import pickle


if __name__ == '__main__':
    
    # use argparse to add the args "num_objects, num_samples, num_iterations, headless, offline, "
    parser = argparse.ArgumentParser(description="Create a dataset from sumulator")
    parser.add_argument('--scene_ids', type=int, nargs='*', help='list of scene IDs', default=[873])
    parser.add_argument('--headless', type=bool, default=True)
    parser.add_argument('--resolution', type=list, help='camera resolution', default=(1080, 1920))
    parser.add_argument('--fov', type=int, help='fov of camera', default=90)
    parser.add_argument('--max_objects', type=int, help='number of objects to annotate per image', default=5)

    args = parser.parse_args()

    files = os.listdir('hm3d/val/')
    scene_paths = []
    sim_kwargs = []
    for f in files:
        try:
            if int(f[2:5]) in args.scene_ids:
                # pdb.set_trace()
                hsh = f[6:]
                for fov in [135]:
                    sim_kwargs.append({'scene_path': f'hm3d/val/00{f[2:5]}-{hsh}/{hsh}.basis.glb', 'sensors':[-1.8, 0, 1.8],
                                'scene_config': "hm3d/val/hm3d_annotated_val_basis.scene_dataset_config.json",
                                'resolution': args.resolution, 'headless': False, 'fov': 135, 'scene_id': f[2:5]})

        except:
            continue

    sim = AnnotatedSimulator(**sim_kwargs[0])
    #obs = sim.step('r', num_objects=2, annotate_image=True)
    #context = obs['annotations']
    #print(context)
    p1 = [(1.6, -2.7), (1.5, -1.8), (1.6, -0.9), (1.5, 0), (1.6, 0.9),  (1.5, 1.8), (1.6, 2.7)]
    # p1 = [(1.95, -np.pi*0.42), (1, 0),  (1.95, 0.41*np.pi)]

    sim.run_user_input(annotate_image=True, points=p1)

    # out = sim.step('r')
    # image = out['image']
    # i = 0
    # while True:
    
    #     i += 1
    #     path = f'logs/test'
    #     im_file = Image.fromarray(image[:, :, 0:3].astype('uint8'))
    #     im_file.save(f'{path}/image_scene{i}.png')
    #     image = sim.move('rotate', np.pi/4)

