from ast import arg, parse
from pdb import run
from dotenv import load_dotenv
import os
import random
import datetime
import wandb

load_dotenv() 
import argparse
import os
import sys
sys.path.insert(0, '/home/dylangoetting/SpatialBenchmark')
from src.agent import *
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
    parser.add_argument('-e', '--explore_factor', type=float, default=0)
    parser.add_argument('-piv', '--pivot',  action='store_true')
    parser.add_argument('-ca', '--catch',  action='store_true')
    parser.add_argument('-v', '--version',  type=int, default=2)
    parser.add_argument('--parts', type=int, default=10)
    parser.add_argument('--part', type=int, default=0)
    parser.add_argument('-n', '--name', type=str, default='default')
    parser.add_argument('-pa', '--parallel', action='store_true')
    parser.add_argument('-lf', '--log_freq', type=int, default=1)
    parser.add_argument('--port', type=int, default=5000, help='Port for the Flask server')
    parser.add_argument('--fov', type=int, default=140, help='Field of view')
    parser.add_argument('-de', '--depth_est', action='store_true')
    parser.add_argument('--noslide', action='store_true')
    parser.add_argument('--height', type=float, default = 1.5)
    parser.add_argument('--pitch', type=float, default = -0.45)
    parser.add_argument('--nm', action='store_true')
    parser.add_argument('-res', '--res_factor',  type=int, default=1)


    args = parser.parse_args()
    if args.name == 'default':
        args.name = f'default_{random.randint(0, 1000)}'
    log_file = f'parallel/{args.task}_{args.name}/{args.part}_of_{args.parts}.txt'
    if not os.path.exists(log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s %(levelname)s: %(message)s')
        
    if args.seed == 0:
        args.seed = 101 
    random.seed(args.seed)
    np.random.seed(args.seed)
    outer_run_name = args.name
    clip_mag = 2.5
    
    arrow_width = 0.8
    p1 = [(clip_mag, -np.pi*0.36), (clip_mag, -np.pi*0.28), (clip_mag, 0), (clip_mag, 0.28*np.pi),  (clip_mag, 0.36*np.pi)]
    p7 = [(1.7, -np.pi*0.46), (1.8, -np.pi*0.36), (1.8, -np.pi*0.3), (1.9, -np.pi*0.19), (1.9, 0), (1.9, np.pi*0.19), (1.8, 0.3*np.pi), (1.8, 0.36*np.pi),  (1.7, 0.46*np.pi)]
    pm = [(2.5, -3*arrow_width), (2.5, -2*arrow_width), (2.5, -arrow_width), (2.5, 0), (2.5, arrow_width),  (2.5, 2*arrow_width), (2.5, 3*arrow_width)]
    points = pm if args.multi else p1

    fac = args.res_factor
    resolution = (1080//fac, 1920//fac)
    sens = [2*arrow_width, 0, -2*arrow_width] if args.multi else [0]
    fov = args.fov
    sim_kwargs = { 'sensors':sens, 'resolution': resolution, 'headless': True, 'fov': fov, 'random_seed': args.seed, 
                  'height': args.height, 'pitch': args.pitch, 'slide': not args.noslide}
    
    if len(sim_kwargs['sensors']) > 1:
        sys_instruction = f"You are an embodied robotic assistant, with {len(sim_kwargs['sensors'])} different RGB image sensors that each point in different directions. You observe the images and instructions given to you and output a textual repsonse, which is converted into actions that physically move you within the environment. You cannot move through closed doors. "
    else:
        sys_instruction = "You are an embodied robotic assistant, with an RGB image sensor. You observe the image and instructions given to you and output a textual response, which is converted into actions that physically move you within the environment. You cannot move through closed doors. "
    
    model = 'gemini-1.5-pro' if args.pro else 'gemini-1.5-flash'
    if args.nm:
        model += '-002'
    vlm_model = GeminiModel(model=model, sys_instruction=sys_instruction)         

    run_kwargs = {
        'outer_loop': args.iterations,
        'inner_loop': args.inner_loop,
        'log_freq': args.log_freq,
    }
    exp_kwargs={'split': args.split, 'scene_ids': args.scene_ids, 'parts': args.parts, 'part': args.part, 'parallel': args.parallel if args.parallel else False}

    bench_cls = DynamicBench
    if args.task == 'objnav':
        agent_cls = NavAgent
        bench_cls = NavBench

    if args.task == 'hmon':
        bench_cls = HMONBench
        agent_cls = HMONAgent
        exp_kwargs.update({'split': 'val'})
        run_kwargs['success_thresh'] = 0.3
        run_kwargs['version'] = args.version

    if args.task == 'vlnce':
        bench_cls = VLNCE
        exp_kwargs.update({'split': 'val_unseen'})


    if args.task == 'goat':
        bench_cls = GOATBench
        agent_cls = GOATAgent
        exp_kwargs.update({'split': args.split})
        run_kwargs['max_steps_per_goal'] = args.max_steps_per_goal
        run_kwargs['success_thresh'] = 0.25

    if args.task == 'eqa':
        bench_cls = EQABench
        agent_cls = EQAAgent
        hard_qs = None
        # hard_qs += [360, 150, 201, 119, 210, 466, 246, 66]
        exp_kwargs['scene_ids'] = hard_qs

    agent_kwargs = {
        'vlm': vlm_model,
        'fov': fov,
        'sensors': sens,
        'resolution': resolution,
        'task': args.task,
        'outer_loop': args.iterations,
        'history': args.history,
        'inner_loop': args.inner_loop,
        'consistency': args.consistency,
        'log_freq': args.log_freq,
        'uniform': True if args.uniform else False,
        'points': points,
        'use_map': args.use_map,
        'explore_factor': args.explore_factor,
        'pivot': True if args.pivot else False,
        'clip_mag': clip_mag,
        'depth_est': True if args.depth_est else False
    }
    
    agt = agent_cls(**agent_kwargs)

    try:
        nbench = bench_cls(sim_kwargs=sim_kwargs, exp_kwargs=exp_kwargs, agent=agt, outer_run_name=outer_run_name, catch=True if args.catch else False, 
                           log_file=log_file, port=args.port)
        nbench.run_experiment(**run_kwargs)
    except Exception if args.catch else DatabaseError as e:
        tb = traceback.extract_tb(e.__traceback__)
        for frame in tb:
            logging.error(f"Frame {frame.filename} line {frame.lineno}")
        logging.error(e)
        print(e)
    finally:
        wandb.finish()