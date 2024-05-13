import json
import yaml
import argparse
import os
import sys
sys.path.insert(0, '/home/dylangoetting/SpatialBenchmark')
# print("Current Working Directory:", os.getcwd())
from src.benchmark import *
from src.llavaAgent import *
import h5py
import pickle

if __name__ == '__main__':
    
    # use argparse to add the args "num_objects, num_samples, num_iterations, headless, offline, "
    parser = argparse.ArgumentParser(description="Create a dataset from sumulator")
    parser.add_argument('name', type=str, help='Name of dataset')
    parser.add_argument('--scene_id', type=int, help='number of scene', default=800)
    parser.add_argument('--headless', type=bool, default=True)
    parser.add_argument('--resolution', type=list, help='camera resolution', default=(1080, 1920))
    parser.add_argument('--fov', type=int, help='fov of camera', default=90)
    parser.add_argument('--size', type=int, help='number of items in dataset', default=50)
    parser.add_argument('--max_objects', type=int, help='number of objects to annotate per image', default=5)

    args = parser.parse_args()

    files = os.listdir('scenes/hm3d/minival/')
    scene_paths = 'scenes/hm3d/minival/00808-y9hTuugGdiq/y9hTuugGdiq.basis.glb'
    for f in files:
        try:
            if int(f[2:5]) == args.scene_id:
                hsh = f[6:] 
                scene_path = f'scenes/hm3d/minival/00{args.scene_id}-{hsh}/{hsh}.basis.glb'
        except:
            continue


    sim_kwargs = {'scene_path': scene_path,
                  'scene_config': "scenes/hm3d/minival/hm3d_annotated_minival_basis.scene_dataset_config.json",
                  'resolution': args.resolution, 'headless': args.headless, 'fov': args.fov}
    # pdb.set_trace()
    asim = AnnotatedSimulator(**sim_kwargs)
    custom_dtype = np.dtype([
        ('image', np.float32, (args.resolution[0], args.resolution[1], 4)),
        ('metadata', 'S5000'),  # JSON string, up to 1000 bytes
        ('fov', int),
        ('label', int)
    ])
    with h5py.File(f'annotated_datasets/{args.name}.hdf5', 'w') as f:
    # Create a group for images
    
        dataset = f.create_dataset("data", shape=(args.size,), dtype=custom_dtype)   

        for iter in range(args.size):
            while True:
                out = asim.step('r', num_objects=args.max_objects, annotate_image=False)
                if len(out['annotations']) == args.max_objects:
                    break
            image = out['image']
            image_float = image.astype('float32')
            mdata = out['annotations']
            json_mdata = pickle.dumps(mdata)
            item = np.zeros((), dtype=custom_dtype)
            item['image'] = image_float
            item['metadata'] = json_mdata
            item['label'] = iter
            item['fov'] = args.fov
            dataset[iter] = item



    file_path = f'annotated_datasets/{args.name}.hdf5'

    # Open the HDF5 file in read mode
    with h5py.File(file_path, 'r') as f:
        # Access the dataset
        dataset = f[args.name]

        # Iterate over the dataset
        for i in range(len(dataset)):
            # Read the image data
            image_data = dataset[i]['image']
            # Deserialize the metadata
            metadata = pickle.loads(dataset[i]['metadata'])

            # Read the label
            label = dataset[i]['label']

            # Process the data (for demonstration, just print some info)
            print(f"Entry {i}:")
            print(f"Label: {label}")
            print(f"Metadata: {metadata}")
            print(f"Image shape: {image_data.shape}")