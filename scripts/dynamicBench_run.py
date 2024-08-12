import json
import yaml
import argparse
import os
import sys
sys.path.insert(0, '/home/dylangoetting/SpatialBenchmark')
from src.benchmark import *
from src.vlmAgent import *
from src.dynamicBench import *



if __name__ == '__main__':
    
    # use argparse to add the args "num_objects, num_samples, num_iterations, headless, offline, "
    parser = argparse.ArgumentParser(description="Create a dataset from sumulator")
    parser.add_argument('--scene_ids', type=int, nargs='*', help='list of scene IDs', default=[873])
    parser.add_argument('--resolution', type=list, help='camera resolution', default=(1080, 1920))
    parser.add_argument('--fov', type=int, help='fov of camera', default=135)

    args = parser.parse_args()

    files = os.listdir('scenes/hm3d/val/')
    scene_paths = []
    sim_kwargs = []
    for f in files:
        try:
            if int(f[2:5]) in args.scene_ids:
                hsh = f[6:]
                sim_kwargs.append({'scene_path': f'scenes/hm3d/val/00{f[2:5]}-{hsh}/{hsh}.basis.glb',
                            'scene_config': "scenes/hm3d/val/hm3d_annotated_val_basis.scene_dataset_config.json",
                            'resolution': args.resolution, 'headless': True, 'fov': args.fov, 'scene_id': f[2:5]})

        except:
            continue
    
    #agent = LLaVaAgent('qwen7b', quantize=False, use_flash_attention_2=True, device_map='auto', do_sample=True)
    agent = GeminiAgent(sys_instruction="You are an embodied robotic assistant, with an RGB image sensor. You can move forwards/backwards, and rotate left/right. You observe the image and instructions given to you and output a helpful repsonse. Your textual output is converted into actions that physically move you within the environment. You cannot move through obstacles. ")
    traj = 0
    p1 = [(1.95, -np.pi*0.42), (1.6, -np.pi*0.29), (1.5, 0), (1.6, 0.29*np.pi),  (1.95, 0.41*np.pi)]
    p2 = [(1.75, -np.pi*0.37), (1.5, -np.pi*0.22), (1.5, 0), (1.5, 0.22*np.pi),  (1.75, 0.37*np.pi)]
    dbench = DynamicBench(sim_kwargs[0], agent, {'task': 'obj_nav'})
    for obj in ['UPSTAIRS BEDROOM']:
        for itr in range(8): 
            print(f'on itr number {itr}')
            
            dbench.run(log_freq=1, history=5, inner_loop=40, draw_arrows=True, points=p1, font_size=3, font_thickness=4, random_spawn=False, consistency=1)
            dbench.run(log_freq=1, history=5, inner_loop=40, draw_arrows=True, points=p1, font_size=3, font_thickness=4, random_spawn=False, consistency=3)
            dbench.run(log_freq=1, history=5, inner_loop=40, draw_arrows=True, points=p2, font_size=3, font_thickness=4, random_spawn=False, consistency=3)


    dbench.annotatedSimulator.sim.close()

