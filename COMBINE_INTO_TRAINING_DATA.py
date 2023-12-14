'''
In this Python Script, we combine the (i) 3D coordinates, (ii) viewing directions, (iii) RGB color values, and
(iv) density values into one dataset. This dataset will consist of input-output pairs where each input consists of
3D coordinates and a viewing direction, and each output consists of RGB color values and a density value.
'''





# Importing the essential libraries
import numpy as np
import json
import os
import cv2
import pickle
from sklearn.model_selection import train_test_split





# Python Function 0: Save the output in .pkl format
def save_to_file(training_data, filename='training_data.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(training_data, f)
    print(f"Data saved to {filename}")





# Python Function 1: Map 2D Pixels to 3D Coordinates
def load_camera_params(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return list(data.values())

def back_project(pixel_data, camera_params):
    focal_length = np.array(camera_params['camera_intrinsics'][:2])  # Extracting fx and fy
    principal_point = np.array(camera_params['camera_intrinsics'][2:])  # Extracting cx and cy

    camera_pose = np.array(camera_params['camera_pose'])
    rotation_matrix = camera_pose[:9].reshape(3, 3)
    translation_vector = camera_pose[9:]

    normalized_pixel_data = (pixel_data - principal_point) / focal_length

    homogeneous_pixel_data = np.concatenate([normalized_pixel_data, np.ones((normalized_pixel_data.shape[0], 1))], axis=1)

    camera_coordinates = np.einsum("ij,aj->ai", np.linalg.inv(rotation_matrix),
                                   homogeneous_pixel_data - translation_vector)

    return camera_coordinates

def map_2d_pixels_to_3d_coords(image_dir, camera_params_list):
    pixel_data_list = extract_pixels_from_images(image_dir)
    all_camera_coordinates = []

    for pixel_data, camera_params in zip(pixel_data_list, camera_params_list):
        # Ensure pixel_data has the right shape
        pixel_data = pixel_data.reshape(-1, 2)

        camera_coordinates = back_project(pixel_data, camera_params)
        all_camera_coordinates.append(camera_coordinates)

    return all_camera_coordinates





# Python Function 2: Extract Pixels from Images
def extract_pixels_from_images(image_dir):
    """
    Function to generate the pixel coordinates for each pixel in the image.
    The output is a list of 2D numpy arrays where each row in an array is the (x, y) coordinate of a pixel.
    """

    # Ensure only .jpg files are processed, case-insensitive
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')])

    # Initialize an empty list to hold the pixel data
    pixel_data = []

    # Loop over each image file
    for i, image_file in enumerate(image_files, start=1):
        print("Image Number: ", i)
        print("Image File: ", image_file)  # Print out image file name

        # Create the full image path
        image_path = os.path.join(image_dir, image_file)

        # Get the image shape by loading it with OpenCV
        image_shape = cv2.imread(image_path).shape[:2]

        # Check if image is loaded correctly
        if image_shape is None:
            print(f"Could not load image: {image_path}")
            continue

        # Generate a grid of (x, y) coordinates that correspond to each pixel in the image
        x, y = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
        coordinates = np.stack([x, y], axis=-1).reshape(-1, 2)

        # Append the 2D coordinates to the list
        pixel_data.append(coordinates)

    return pixel_data





# Python Function 3: Create Viewing Directions
def create_viewing_directions(camera_params_list, all_camera_coordinates):
    viewing_directions = []

    for camera_coordinates, camera_params in zip(all_camera_coordinates, camera_params_list):
        # Extracting the camera's position from the camera parameters
        camera_position = np.array(camera_params['camera_pose'][9:])

        # Extend the camera_position to have the same number of rows as camera_coordinates
        camera_position = np.tile(camera_position, (camera_coordinates.shape[0], 1))

        # Now camera_position and camera_coordinates have the same shape and can be subtracted
        # to compute the viewing direction for each 3D point
        viewing_direction = camera_position - camera_coordinates

        # Normalizing the viewing direction vectors
        norm = np.linalg.norm(viewing_direction, axis=1, keepdims=True)
        viewing_direction = viewing_direction / norm

        viewing_directions.append(viewing_direction)

    return viewing_directions





# Python Function 4: Extract 2D Pixels from Images
def extract_2d_pixels_from_images(image_dir):
    """
    Function to read the images and convert them into a pixel array.
    The output is a list of flattened pixel data from each image.
    """

    # Ensure only .jpg files are processed, case-insensitive
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')])

    # Initialize an empty list to hold the pixel data
    pixel_data = []

    # Loop over each image file
    for i, image_file in enumerate(image_files, start=1):
        print("Image Number: ", i)

        # Create the full image path
        image_path = os.path.join(image_dir, image_file)

        # Load the image using OpenCV
        image = cv2.imread(image_path)

        # Check if image is loaded correctly
        if image is None:
            print(f"Could not load image: {image_path}")
            continue

        # Convert the image from BGR (default in OpenCV) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Flatten the 2D array (height x width, channels=3) into 1D and append it to the pixel_data list
        pixel_data.append(image.reshape(-1, 3))

    return pixel_data





# Python Function 5: Initialize Density Values
def initialize_density_values(all_camera_coordinates, initial_value=1.0):
    density_values = []

    for camera_coordinates in all_camera_coordinates:
        density = np.ones((camera_coordinates.shape[0], 1)) * initial_value
        density_values.append(density)

    return density_values





# Final Main Function: Combine Python Function 1, 2, 3, 4, and 5
def combine_into_training_data(all_camera_coordinates, all_view_dirs, all_rgb_values, all_density_values):
    training_data = []

    for camera_coordinates, view_dirs, rgb_values, density_values in zip(all_camera_coordinates,
                                                                         all_view_dirs,
                                                                         all_rgb_values,
                                                                         all_density_values):

        print("camera_coordinates shape inside loop:", camera_coordinates.shape)
        print("view_dirs shape inside loop:", view_dirs.shape)

        # Inputs are 3D coordinates and viewing directions
        inputs = np.hstack([camera_coordinates.reshape(-1, 3), view_dirs.reshape(-1, 3)])

        # Ensure density_values have the right shape
        density_values = density_values.reshape(-1, 1)

        # Ensure rgb_values have the right shape
        rgb_values = rgb_values.reshape(-1, 3)

        # Outputs are RGB color values and density values
        outputs = np.hstack([rgb_values, density_values])

        # Combine inputs and outputs into a single dictionary for this image
        data = {'inputs': inputs, 'outputs': outputs}

        # Append this data dictionary to the list of all training data
        training_data.append(data)

    return training_data





# Step 1: Calling the variable "all_camera_coordinates"
json_file = 'nerf_data.json'
image_dir = 'WITHOUT_METADATA/OUTPUT_IMAGES_2/'

camera_params_list = load_camera_params(json_file)
all_camera_coordinates = map_2d_pixels_to_3d_coords(image_dir, camera_params_list)

# Step 2: Calling the variable "all_view_dirs"
all_view_dirs = create_viewing_directions(camera_params_list, all_camera_coordinates)

# Step 3: Calling the variable "all_rgb_values"
all_rgb_values = extract_2d_pixels_from_images(image_dir)

# Step 4: Calling thr variable "initialize_density_values
initial_density_values = initialize_density_values(all_camera_coordinates)





# print("all_camera_coordinates shape:", np.shape(all_camera_coordinates))
# print("all_view_dirs shape:", np.shape(all_view_dirs))
# print("all_rgb_values shape:", np.shape(all_rgb_values))
# print("initial_density_values shape:", np.shape(initial_density_values))





# Time to Run the Function
training_data = combine_into_training_data(all_camera_coordinates, all_view_dirs, all_rgb_values, initial_density_values)




# # UNCOMMENT IF YOU WANT TO SAVE THE OUTPUT AS A .pkl FILE
# # Save the training data to a pickle file
# save_to_file(training_data, filename='training_data_2.pkl')





for data in training_data:
    print("Inputs:", data['inputs'])
    print("Outputs:", data['outputs'])





# First, we split the data into training and the temporary set (30:70 split)
train_data, temp_data = train_test_split(training_data, test_size = 0.3, random_state = 42)

# Then, we split the temporary set into validation and testing set (50:50 split)
val_data, test_data = train_test_split(temp_data, test_size = 0.5, random_state = 42)

print(f"Training data: {len(train_data)} images")
print(f"Validation data: {len(val_data)} images")
print(f"Testing data: {len(test_data)} images")





# End of the Python Script
print("TASK COMPLETED SUCCESSFULLY")