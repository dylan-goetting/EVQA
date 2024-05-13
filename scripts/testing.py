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


dtype = np.dtype([
        ('image', np.float32, (1080, 1920, 3)),
        #('metadata', 'S5000'),  # JSON string, up to 1000 bytes
        ('label', int)
    ])
f =  h5py.File(f't.hdf5', 'w')
# Create a group for images

dataset = f.create_dataset('t', shape=(10,), dtype=dtype)   

for iter in range(10):
    data = np.zeros((), dtype=dtype)
    data['label'] = iter
    data['image'] = np.ones((1080, 1920, 3), dtype=np.float32)
    dataset[iter] = data



file_path = f't.hdf5'

# Open the HDF5 file in read mode
f1 = h5py.File(file_path, 'r')
    # Access the dataset
dataset = f1['t']

    # Iterate over the dataset
for i in range(len(dataset)):
    # Read the image data
    # Read the label
    label = dataset[i]['label']
    im = dataset[i]['image']
    print(f"Label: {label}")
