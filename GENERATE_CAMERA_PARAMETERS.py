'''
THE GOAL OF THIS PYTHON SCRIPT IS TO CONVERT THE CAMERA PARAMETERS EXTRACTED USING "COLMAP" TO A FORMAT WHICH
CAN BE USED FOR TRAINING A NEURAL RADIANCE FIELD (NeRF) MODEL.

NOTE: THE OUTPUT OF THIS SCRIPT WILL BE IN .JSON FORMAT
'''

import numpy as np
import pandas as pd
import os
import json
from scipy.spatial.transform import Rotation as R

# Function to convert quaternion to rotation matrix
def quat2mat(quat):
    r = R.from_quat(quat)
    return r.as_matrix()

# Functions to read the COLMAP output files
def read_images_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Extract camera parameters for each image
    images = {}
    for i in range(4, len(lines), 2): # Skip the first 4 lines, then read 2 lines at a time
        line = lines[i].split()
        image_id = int(line[0])
        qvec = np.array(list(map(float, line[1:5]))) # Quaternion describing the camera rotation
        tvec = np.array(list(map(float, line[5:8]))) # Camera position
        camera_id = int(line[8])
        images[image_id] = {'qvec': qvec.tolist(), 'tvec': tvec.tolist(), 'camera_id': camera_id}

    return images

def read_cameras_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Extract intrinsic parameters for each camera
    cameras = {}
    for line in lines[3:]: # Skip the first 3 lines
        line = line.split()
        camera_id = int(line[0])
        model = line[1]
        width = int(line[2])
        height = int(line[3])
        params = np.array(list(map(float, line[4:])))
        cameras[camera_id] = {'model': model, 'width': width, 'height': height, 'params': params.tolist()}

    return cameras

def normalize_positions(images):
    # Collect all camera positions
    positions = np.array([image['tvec'] for image in images.values()])

    # Calculate the mean and maximum absolute value
    mean = np.mean(positions, axis=0)
    max_abs = np.max(np.abs(positions - mean))

    # Normalize the positions
    for image in images.values():
        image['tvec'] = ((image['tvec'] - mean) / max_abs).tolist()

def convert_colmap_to_nerf(images, cameras):
    # Convert the data to the format used by the NeRF model
    data = {}
    for image_id, image in images.items():
        camera = cameras[image['camera_id']]
        data[image_id] = {'image_path': os.path.join('images', str(image_id) + '.jpg'),  # .jpg format
                          'camera_pose': np.concatenate([quat2mat(np.array(image['qvec'])).flatten(), np.array(image['tvec'])]).tolist(),  # Convert quaternion to rotation matrix
                          'camera_intrinsics': camera['params'],
                          'width': camera['width'],
                          'height': camera['height']}
    return data

# Paths to the COLMAP output files
images_txt_path = 'WITHOUT_METADATA/OUTPUT_PARAMETERS/images.txt'
cameras_txt_path = 'WITHOUT_METADATA/OUTPUT_PARAMETERS/cameras.txt'

# Read the COLMAP output files
images = read_images_txt(images_txt_path)
cameras = read_cameras_txt(cameras_txt_path)

# Normalize the camera positions
normalize_positions(images)

# Convert the data to the format used by the NeRF model
data = convert_colmap_to_nerf(images, cameras)

# Save the data to a JSON file
with open('nerf_data.json', 'w') as f:
    json.dump(data, f)

# End of the Python Script
print("TASK COMPLETED SUCCESSFULLY")