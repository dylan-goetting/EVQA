from dotenv import load_dotenv
import os
import random
import datetime
load_dotenv() 
from matplotlib.pyplot import arrow
import argparse
import os
import sys
sys.path.insert(0, '/home/dylangoetting/SpatialBenchmark')
from src.vlm import *
from src.dynamicBench import *
from src.benchmarks import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run a dynamic benchmark")
    parser.add_argument('-sd', '--seed', type=int, help='random seed', default=0)
    parser.add_argument('-itr', '--iterations', type=int, help='number of iterations', default=50)
    parser.add_argument('-sid', '--scene_ids', type=int, nargs='+', help='scene ids to run', default=[])
    parser.add_argument('-t', '--task', type=str, help='task to run', default='obj_nav')
    parser.add_argument('-sp', '--split', type=str, help='train or val', default='val')
    parser.add_argument('-his', '--history', type=int, help='context_history', default=0)
    parser.add_argument('-il', '--inner_loop', type=int, help='inner loop', default=30)
    parser.add_argument('-c', '--consistency', type=int, help='inner loop', default=1)
    parser.add_argument('-msg', '--max_steps_per_goal', type=int, help='maximum steps per goal', default=10)
    parser.add_argument('-mu', '--multi', action='store_true', help='verbose')
    parser.add_argument('-u', '--uniform', action='store_true', help='uniform')
    parser.add_argument('-um', '--use_map', type=int, help='use map', default=0)
    parser.add_argument('-p', '--pro', action='store_true', help='pro')
    parser.add_argument('-e', '--explore_factor', type=float, help='use map', default=0)
    parser.add_argument('-st', '--success_thresh', type=float, default=2)
    parser.add_argument('-pm', '--priv_map',  action='store_true')



    args = parser.parse_args()
    if args.seed == 0:
        args.seed = random.randint(0, 10000)
    random.seed(args.seed)
    np.random.seed(args.seed)
    outer_run_name = datetime.datetime.now().strftime("%m%d%s") + "_seed" + str(args.seed) + f'_map{args.use_map}'
    

    arrow_width = 0.75
    p1 = [(1.7, -np.pi*0.42), (1.9, -np.pi*0.28), (1.9, 0), (1.8, 0.28*np.pi),  (1.7, 0.42*np.pi)]
    p7 = [(1.7, -np.pi*0.46), (1.8, -np.pi*0.36), (1.8, -np.pi*0.3), (1.9, -np.pi*0.19), (1.9, 0), (1.9, np.pi*0.19), (1.8, 0.3*np.pi), (1.8, 0.36*np.pi),  (1.7, 0.46*np.pi)]
    pm = [(2.5, -3*arrow_width), (2.5, -2*arrow_width), (2.5, -arrow_width), (2.5, 0), (2.5, arrow_width),  (2.5, 2*arrow_width), (2.5, 3*arrow_width)]
    points = p7 if args.multi else p1
    sens = [2*arrow_width, 0, -2*arrow_width] if args.multi else [0]
    fov = 125 if args.multi else 140
    sim_kwargs = { 'sensors':sens, 'resolution': (1080, 1920), 'headless': True, 'fov': fov, 'random_seed': args.seed}
    
    if len(sim_kwargs['sensors']) > 1:
        sys_instruction = f"You are an embodied robotic assistant, with {len(sim_kwargs['sensors'])} different RGB image sensors that each point in different directions. You observe the images and instructions given to you and output a textual repsonse, which is converted into actions that physically move you within the environment. You cannot move through closed doors. "
    else:
        sys_instruction = "You are an embodied robotic assistant, with an RGB image sensor. You observe the image and instructions given to you and output a textual response, which is converted into actions that physically move you within the environment. You cannot move through closed doors. "
    if args.task == 'meqa':
        sys_instruction = "You are an embodied robotic assistant, with an RGB image sensor. You observe the image and instructions given to you and output a textual response, which is converted into actions that physically move you within the environment. You are working with another agent to complete the task, and you can communicate with them. To be as efficient as possible, you should avoid exlporing the same areas as your partner. "
    model = 'gemini-1.5-pro' if args.pro else 'gemini-1.5-flash'
    vlm_model = GeminiModel(model=model, sys_instruction=sys_instruction)         
    # vlm_model = GPTModel(sys_instruction=sys_instruction)
    #agent = LLaVaAgent('qwen7b', quantize=False, use_flash_attention_2=True, device_map='auto', do_sample=True)
    
    goals = [('kitchen', ['microwave', 'stove', 'oven', 'dishwasher', 'refrigerator']), ('living room', ['sofa', 'couch', 'tv', 'coffee table']), 
            ('bedroom', ['bed', 'nightstand', 'dresser']), ('bathroom', ['toilet', 'shower', 'bath'])] 
    goals = [('bathroom', ['toilet', 'shower', 'bath'])]
    run_kwargs = {
        'outer_loop': args.iterations,
        'history': args.history,
        'inner_loop': args.inner_loop,
        'consistency': args.consistency,
        'log_freq': 1,
        'uniform': True if args.uniform else False,
        'points': points,
        'use_map': args.use_map,
        'explore_factor': args.explore_factor,
        'map_type': 'priv' if args.priv_map else 'unpriv',
    }
    exp_kwargs={'split': args.split, 'scene_ids': args.scene_ids}


    bench_cls = DynamicBench
    if args.task == 'obj_nav':
        bench_cls = NavBench
        run_kwargs['goals'] = goals

    if args.task == 'goat':
        bench_cls = GOATBench
        exp_kwargs = {'split': 'train', 'num_scenes': 20}
        run_kwargs['max_steps_per_goal'] = args.max_steps_per_goal
        run_kwargs['success_thresh'] = args.success_thresh
    if args.task == 'eqa':
        bench_cls = EQABench
        hard_qs = [451, 403, 337, 182, 209, 97, 481]
        hard_qs = None
        # hard_qs += [360, 150, 201, 119, 210, 466, 246, 66]

        exp_kwargs['scene_ids'] = hard_qs

    if args.task == 'meqa':
        bench_cls = MultiAgentEQA

    nbench = bench_cls(sim_kwargs=sim_kwargs, vlm_agent=vlm_model, exp_kwargs=exp_kwargs, outer_run_name=outer_run_name)

    nbench.run_experiment(**run_kwargs)

