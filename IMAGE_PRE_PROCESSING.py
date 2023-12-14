'''
THE GOAL OF THIS PYTHON SCRIPT IS TO PREPROCESS THE 41 IMAGES AS PART OF THE NeRF MODEL TRAINING
BY RESIZING / DOWNSAMPLING THEM TO ENSURE THEY ARE NOT TOO HUGE FOR MODEL TRAINING. THE ORIGINAL
SIZE OF EVERY IMAGE IS 4032 x 3024 WHICH WILL BE DOWNSAMPLED USING THE CODE BELOW
'''

from PIL import Image
import os

input_folder = 'INPUT_IMAGES'
output_folder = 'WITH_METADATA/OUTPUT_IMAGE_2'
# output_folder = 'WITH_METADATA/OUTPUT_IMAGE_4'
# output_folder = 'WITH_METADATA/OUTPUT_IMAGE_8'
# output_folder = 'WITH_METADATA/OUTPUT_IMAGE_16'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def resize_image(img, target_size):
    img.thumbnail(target_size, Image.ANTIALIAS)
    return img

target_size = (2016, 1512)
# target_size = (1008, 756)
# target_size = (504, 378)
# target_size = (252, 189)

a = 0

# Process each image in the input folder
for image_file in os.listdir(input_folder):
    a =  a + 1
    print("PRE-PROCESSED IMAGE NUMBER: ", a)
    if image_file.lower().endswith(('.jpg', '.jpeg')):
        # Read the image
        input_path = os.path.join(input_folder, image_file)
        img = Image.open(input_path)

        # Resize the image (optional)
        img_resized = resize_image(img, target_size)

        # Save the preprocessed image to the output folder
        output_file = os.path.join(output_folder, image_file)
        img_resized.save(output_file, format = 'JPEG', quality = 95, exif = img.info['exif'])

print("TASK COMPLETED SUCCESSFULLY")