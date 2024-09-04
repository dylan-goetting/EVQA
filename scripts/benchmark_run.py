import yaml
import argparse
import sys
sys.path.insert(0, '/home/dylangoetting/SpatialBenchmark')
from src.benchmark import *
from vlm import *

DEFAULT_CONFIG = {
    'headless': True,
    'resolution': (1080, 1920),
    'num_objects': {'max': 2, 'min': 2},
    'num_samples': {'max': 1, 'min': 1},
    'num_iterations': 200,
    'offline': True,
    'vlm_cls': 'LlavaAgent',
    'vlm_kwargs': {'name': '8b'},
    'log_freq': 10,
    'icl': {'max':0, 'min': 0},
    'shuffle': False,
    'dynamic': False,
    'inner_loop': 1
}

if __name__ == '__main__':
    # use argparse to add the args "num_objects, num_samples, num_iterations, headless, offline, "
    parser = argparse.ArgumentParser(description="Load a YAML config file for model training.")
    parser.add_argument('config_path', type=str, help='Path to the configuration YAML file')
    
    args = parser.parse_args()
    with open(f'configs/{args.config_path}.yaml', 'r') as file:
        config = yaml.safe_load(file)
        config = {**DEFAULT_CONFIG, **config}

    sim_kwargs = {'scene_path': 'datasets/hm3d/minival/00808-y9hTuugGdiq/y9hTuugGdiq.basis.glb',
                  'scene_config': "datasets/hm3d/minival/hm3d_annotated_minival_basis.scene_dataset_config.json",
                  'resolution': config['resolution'], 'headless': config['headless']}
    vlm_cls = locals()[config['vlm_cls']]
    vlm_agent = vlm_cls(**config['vlm_kwargs'])

    benchmark = SpatialBenchmark(sim_kwargs, vlm_agent, offline=config['offline'], data_path=config['data_path'])
    benchmark.run(objects=config['num_objects'], samples=config['num_samples'], 
                  num_iterations = config['num_iterations'], log_freq = config['log_freq'], 
                  icl=config['icl'], shuffle=config['shuffle'], dynamic=config['dynamic'], 
                  inner_loop=config['inner_loop'])