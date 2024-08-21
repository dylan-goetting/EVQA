from ast import arg
from dotenv import load_dotenv
import os
import random

load_dotenv() 
import json
from matplotlib.pyplot import arrow
import yaml
import argparse
import os
import sys
sys.path.insert(0, '/home/dylangoetting/SpatialBenchmark')
from src.vlm import *
from src.dynamicBench import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a dynamic benchmark")
    parser.add_argument('-sd', '--seed', type=int, help='random seed', default=0)
    parser.add_argument('-itr', '--iterations', type=int, help='number of iterations', default=50)
    parser.add_argument('-sid', '--scene_ids', type=int, nargs='+', help='scene ids to run', default=[])
    parser.add_argument('-t', '--task', type=str, help='task to run', default='obj_nav')
    parser.add_argument('-sp', '--split', type=str, help='train or val', default='val')
    parser.add_argument('-pa', '--priv_actions', action='store_true', help='use priv actions')
    parser.add_argument('-his', '--history', type=int, help='context_history', default=10)
    parser.add_argument('-il', '--inner_loop', type=int, help='inner loop', default=30)
    parser.add_argument('-c', '--consistency', type=int, help='inner loop', default=1)
    parser.add_argument('-msg', '--max_steps_per_goal', type=int, help='maximum steps per goal', default=10)

    args = parser.parse_args()
    if args.seed == 0:
        args.seed = random.randint(0, 10000)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # scene_counter = pickle.load(open('scenes/hm3d/scene_counter.pickle', 'rb'))
    # obj_counter = pickle.load(open('scenes/hm3d/val/obj_counter.pickle', 'rb'))
    bench_cls = DynamicBench
    if args.task == 'obj_nav':
        bench_cls = NavBench
    if args.task == 'goat':
        bench_cls = GOATBench
    outer_run_name = datetime.datetime.now().strftime("%m%d%s") + "_seed" + str(args.seed)
    

    arrow_width = 0.77
    p1 = [(1.7, -np.pi*0.42), (1.9, -np.pi*0.28), (1.9, 0), (1.8, 0.28*np.pi),  (1.7, 0.42*np.pi)]
    p7 = [(1.7, -np.pi*0.43), (1.8, -np.pi*0.34), (1.9, -np.pi*0.19), (1.9, 0), (1.9, np.pi*0.19), (1.8, 0.34*np.pi),  (1.7, 0.43*np.pi)]
    points = p7 if args.priv_actions else p1
    p7 = [(1.8, -3*arrow_width), (1.8, -2*arrow_width), (1.8, -arrow_width), (1.8, 0), (1.8, arrow_width),  (1.8, 2*arrow_width), (1.8, 3*arrow_width)]
    sim_kwargs = { 'sensors':[0], 'resolution': (1080, 1920), 'headless': True, 'fov': 120, 'random_seed': args.seed}
    sim_kwargs['sensors'] = [2*arrow_width, 0, -2*arrow_width]
    
    if len(sim_kwargs['sensors']) > 1:
        sys_instruction = f"You are an embodied robotic assistant, with {len(sim_kwargs['sensors'])} different RGB image sensors that each point in different directions. You observe the images and instructions given to you and output a textual repsonse, which is converted into actions that physically move you within the environment. You cannot move through obstacles or closed doors. "
    else:
        sys_instruction = "You are an embodied robotic assistant, with an RGB image sensor. You observe the image and instructions given to you and output a textual response, which is converted into actions that physically move you within the environment. You cannot move through obstacles or closed doors. "
    
    vlm_model = GeminiModel(sys_instruction=sys_instruction)         
    #agent = LLaVaAgent('qwen7b', quantize=False, use_flash_attention_2=True, device_map='auto', do_sample=True)
    
    goals = [('kitchen', ['microwave', 'stove', 'oven', 'dishwasher', 'refrigerator']), ('living room', ['sofa', 'couch', 'tv', 'coffee table']), 
            ('bedroom', ['bed', 'nightstand', 'dresser']), ('bathroom', ['toilet', 'shower', 'bath'])] 
    exp_kwargs={'split': args.split, 'scene_ids': args.scene_ids}
    if args.task == 'goat':
        exp_kwargs = {'split': args.split, 'num_scenes': 20}
    nbench = bench_cls(sim_kwargs=sim_kwargs, vlm_agent=vlm_model, exp_kwargs=exp_kwargs, outer_run_name=outer_run_name)
    # nbench.annotatedSimulator.priv_actions = True if args.priv_actions else False
    nbench.run_experiment(outer_loop=args.iterations, log_freq=1, history=args.history, inner_loop=args.inner_loop, points=points, consistency=args.consistency, max_steps_per_goal=args.max_steps_per_goal)

