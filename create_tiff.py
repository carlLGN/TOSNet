import numpy as np
import os
from PIL import Image
import tifffile
from datetime import datetime
now = datetime.now()
formatted_date_time = now.strftime('%Y-%m-%d-%H-%M')


def create_tiff(threshold=0.5, min_size=0):
    # Specify the directory containing the PNG files
    directory = './threed_results'

    # Initialize a list to hold the image arrays
    image_arrays = []

    # Loop through all png files in the directory and append each array to the list
    for i in range(500):
        filename = f'Layer{i}.png'
        filepath = os.path.join(directory, filename)
        image = Image.open(filepath)
        image_array = np.array(image)
        image_arrays.append(image_array)

    # Stack arrays to get a single 3D array (assuming all images have the same dimensions)
    image_stack = np.stack(image_arrays, axis=0)

    # Save the 3D array as a TIFF file
    tifffile.imwrite(f'narwhal__thres{threshold}_minsize{min_size}_fulldata_{formatted_date_time}.tif', image_stack)
    
if __name__ == '__main__':
    create_tiff(threshold=0.5, min_size=10)
