from ast import arg
import json
from random import shuffle
from matplotlib.pyplot import arrow
from torch import rand
import yaml
import argparse
import os
import sys
sys.path.insert(0, '/home/dylangoetting/SpatialBenchmark')
from src.benchmark import *
from src.vlmAgent import *
from src.dynamicBench import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a dynamic benchmark")
    parser.add_argument('--seed', type=int, help='random seed', default=11)
    parser.add_argument('--iterations', type=int, help='number of iterations', default=50)
    parser.add_argument('--scene_ids', type=int, nargs='+', help='scene ids to run', default=[])
    parser.add_argument('--task', type=str, help='task to run', default='obj_nav')
    parser.add_argument('--split', type=str, help='train or val', default='train')
    
    args = parser.parse_args()
    if args.split == 'train':
        json_file =  "scenes/hm3d/hm3d_annotated_train_basis.scene_dataset_config.json"
        with open(json_file, 'r') as f:
            data = json.load(f)
            scenes = data['stages']['paths']['.glb']
            scene_ids = set(int(s[2:5]) for s in scenes)
        files = [f for f in os.listdir('scenes/hm3d/train/') if int(f[2:5]) in scene_ids]
    else:
        files = os.listdir('scenes/hm3d/val/')

    random.seed(args.seed)
    shuffle(files)
    if args.scene_ids == []:
        args.scene_ids = [int(f[2:5]) for f in files]
    # scene_counter = pickle.load(open('scenes/hm3d/scene_counter.pickle', 'rb'))
    # obj_counter = pickle.load(open('scenes/hm3d/val/obj_counter.pickle', 'rb'))
    bench_cls = DynamicBench
    if args.task == 'obj_nav':
        bench_cls = NavBench
    outer_run_name = "" + datetime.datetime.now().strftime("%m%d") + "_seed" + str(args.seed)

    for itr in range(args.iterations):
        f = random.choice(files)
        try:
            if int(f[2:5]) in args.scene_ids:
                hsh = f[6:]
                arrow_width = 0.77
                p1 = [(1.7, -np.pi*0.42), (1.9, -np.pi*0.28), (1.9, 0), (1.8, 0.28*np.pi),  (1.7, 0.42*np.pi)]
                p7 = [(1.8, -3*arrow_width), (1.8, -2*arrow_width), (1.8, -arrow_width), (1.8, 0), (1.8, arrow_width),  (1.8, 2*arrow_width), (1.8, 3*arrow_width)]
                sim_kwargs = {'scene_path': f'scenes/hm3d/{args.split}/00{f[2:5]}-{hsh}/{hsh}.basis.glb', 'sensors':[0],
                            'scene_config': f"scenes/hm3d/hm3d_annotated_{args.split}_basis.scene_dataset_config.json",
                            'resolution': (1080, 1920), 'headless': True, 'fov': 135, 'scene_id': f[2:5], 'random_seed': args.seed}
                sim_kwargs['fov'] = 135
                # sim_kwargs['sensors'] = [2*arrow_width, 0, -2*arrow_width]
                
                if len(sim_kwargs['sensors']) > 1:
                    sys_instruction = f"You are an embodied robotic assistant, with {len(sim_kwargs['sensors'])} different RGB image sensors that each point in different directions. You observe the images and instructions given to you and output a textual repsonse, which is converted into actions that physically move you within the environment. You cannot move through obstacles or closed doors. "
                else:
                    sys_instruction = "You are an embodied robotic assistant, with an RGB image sensor. You observe the image and instructions given to you and output a textual response, which is converted into actions that physically move you within the environment. You cannot move through obstacles or closed doors. "
                agent = GeminiAgent(sys_instruction=sys_instruction)         
                #agent = LLaVaAgent('qwen7b', quantize=False, use_flash_attention_2=True, device_map='auto', do_sample=True)
                
                keys = [('kitchen', ['microwave', 'stove', 'oven', 'dishwasher', 'refrigerator']), ('living room', ['sofa', 'couch', 'tv', 'coffee table']), 
                        ('bedroom', ['bed', 'nightstand', 'dresser']), ('bathroom', ['toilet', 'shower', 'bath'])] 
                
                nbench = bench_cls(sim_kwargs, agent, {'task': 'obj_nav', 'goals': keys}, outer_run_name=outer_run_name)
                
                print('\n\nRUNNING SCENE:', f[2:5], 'ITERATION:', itr)

                nbench.run(log_freq=1, history=1, inner_loop=5, points=p1, consistency=1)
                nbench.annotatedSimulator.sim.close()
                nbench.get_costs()


        except SystemError as e:
            print(e)
            continue


