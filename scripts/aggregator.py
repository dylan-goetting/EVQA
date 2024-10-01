import re
import traceback
from cv2 import exp
from flask import Flask, request, jsonify
import threading
import numpy as np
import pandas as pd
import scipy as sp
import wandb
import time
import argparse
import sys
sys.path.insert(0, '/home/dylangoetting/SpatialBenchmark')
from src.utils import *

app = Flask(__name__)

# Aggregated metrics
episode_data = []
episodes_completed = set()
cumulative_metrics = {'episodes_completed': 0}
total_episodes = [1]
spend = {}
lock = threading.Lock()
terminate_event = threading.Event()
task_log = {}
rows = []

@app.route('/terminate', methods=['POST'])
def terminate():
    with lock:
        print("Received termination signal.")
    terminate_event.set()
    logging_thread.join()  # Wait for the logging thread to finish

    shutdown_server()
    return jsonify({'status': 'terminating'}), 200



@app.route('/log', methods=['POST'])
def log_metrics():
    data = request.json
    required_keys = ['instance', 'data_ndx', 'total_episodes', 'spend', 'task', 'task_data']
    for required_key in required_keys:
        if required_key not in data:
            print('missing key', required_key)
            return jsonify({'status': 'error', 'message': f'Missing key {required_key} in data'}), 400
        
    with lock:
        instance = data['instance']
        spend[instance] = data['spend']
        total_episodes[0] = data['total_episodes']

        if 'task_data' in data:
            if data['task'] not in task_log:
                task_log[data['task']] = []
            task_log[data['task']].append(data['task_data'])

        if data['data_ndx'] not in episodes_completed:
            episodes_completed.add(data['data_ndx'])
            episode_data.append(data)
            cumulative_metrics['episodes_completed'] += 1

            for k, v in data.items():
                if k in required_keys:
                    continue
                if k not in cumulative_metrics:
                    cumulative_metrics[k] = 0
                cumulative_metrics[k] += v
            
    return jsonify({'status': 'success'}), 200

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        print("Not running with the Werkzeug Server. Unable to shut down the server.")
        return
    func()

def log_task_data():
    for task, data in task_log.items():
        if task == 'objnav':
            if 'explore_curve' in data[0]:
                
                mean_curve = bucket_normalized_timesteps([d['task_data']['explore_curve'] for d in episode_data])
                xs = np.linspace(0, 1, len(mean_curve), endpoint=False) + (0.5 / len(mean_curve))
                data = [[x, y] for (x, y) in zip(xs, mean_curve)]
                table = wandb.Table(data=data, columns=["x", "y"])
                wandb.log(
                    {
                        "explore_curve": wandb.plot.line(
                            table, "x", "y", title="Exploration curve"
                        )
                    }
                )

                xs = [a['navigable_area'] for a in episode_data]
                ys = [a['explored'] for a in episode_data]
                data = [[x, y] for (x, y) in zip(xs, ys)]
                table = wandb.Table(data=data, columns = ["x", "y"])
                wandb.log({"navigability scatter" : wandb.plot.scatter(table, "x", "y",
                                                title="Navigability vs Explored Area")})

        if task == 'goat':
            rows = []
            for d in episode_data:
                rows += d['task_data']['goal_data']
            columns = list(rows[0].keys())
            data = [[row[col] for col in columns] for row in rows]
            # df = pd.DataFrame(expanded_data)
            # table = wandb.Table(dataframe=dataframe[0])

            my_table = wandb.Table(columns=columns, data=data)
            wandb.log({"goat_table": my_table})

            out_log = {
                'goals_completed': len(rows),
                'success_rate': sum([1 for r in rows if r['goal_reached']]) / len(rows),
                'spl': sum([r['spl'] for r in rows]) / len(rows),
            }
            wandb.log(out_log)


def wandb_logging(sleep):
    while not terminate_event.is_set():
        time.sleep(sleep)  # Log every 10 seconds
        with lock:
                
            total_spend = sum(spend.values())
            out_data = {
                'total_spend': total_spend,
                'episodes_completed': cumulative_metrics['episodes_completed'],
                'progress': cumulative_metrics['episodes_completed'] / total_episodes[0],
            }
            print(out_data)
            for key, value in cumulative_metrics.items():
                if key == 'episodes_completed' or cumulative_metrics['episodes_completed'] == 0:
                    continue
                out_data[key] = value/cumulative_metrics['episodes_completed']
            wandb.log(out_data)
            try:
                log_task_data()
            except Exception as e:
                tb = traceback.extract_tb(e.__traceback__)
                for frame in tb:
                    print(f"Frame {frame.filename} line {frame.lineno}")
                # logging.error(e)
                print(e)
                print(f"Error logging task data: {e}")

            print(f"Logged to wandb")

    time.sleep(1)
    with lock:
        total_spend = sum(spend.values())
        out_data = {
            'total_spend': total_spend,
            'episodes_completed': cumulative_metrics['episodes_completed'],
            'progress': cumulative_metrics['episodes_completed'] / total_episodes[0],
        }
        for key, value in cumulative_metrics.items():
            if value > 0 and key != 'episodes_completed':
                out_data[key] = value/cumulative_metrics['episodes_completed']
            wandb.log(out_data)
        print(f"final log to wandb")

    print("WandB logging thread terminating.")
    wandb.finish()
    print("Aggregator has shut down.")

    exit(0)
  
if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Aggregator for Parallel Workers')
    parser.add_argument('--name', type=str, required=True, help='Name for the wandb run group')
    parser.add_argument('--port', type=int, default=5000, help='Port for the Flask server')
    parser.add_argument('--sleep', type=int, default=10, help='Time to sleep between logging')
    args = parser.parse_args()

    # Initialize WandB
    task = args.name.split('_')[0]
    wandb.init(project='embodied_navigation', group=task, name=args.name)
    print('initialized wandb')
    # Start wandb logging in a separate thread
    logging_thread = threading.Thread(target=wandb_logging, daemon=True, args=(args.sleep,))
    logging_thread.start()
    
    # Run Flask app
    try:
        app.run(host='0.0.0.0', port=args.port)
    except KeyboardInterrupt:
        print("Aggregator received KeyboardInterrupt. Shutting down.")
    finally:
        terminate_event.set()
        logging_thread.join()
        print("Aggregator has shut down.")
        exit(0)