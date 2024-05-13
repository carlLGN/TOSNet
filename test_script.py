# Author: Carl Nørlund & Malthe Bresler
# Date: 2024-03-06


import tifffile as tif
import numpy as np
import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


from threed_helpers.preprocess import load_data
from threed_helpers.preprocess import load_data_threshold
from threed_helpers.preprocess import save_connected_components
from threed_helpers.preprocess import create_list_of_images_txt
from threed_helpers.preprocess import instance_dict
from create_tiff import create_tiff

if __name__ == '__main__':
    
    TEST_REAL_LABELS = True
    

    #Load Data
    image_path = './threed_data/scan_07_crop.tif'
    mask_path = './threed_data/scan_07_crop_mask.tif' # If it is anders mask then roll axis in load_data. If threshold dont roll axis (load_data_threshold)
    img, mask = load_data(image_path, mask_path)
    
    path_to_test = 'test.py'
    
    threshold = 0.5
    layers = [5, 85]
    min_size = [50]
    
    
    M, N, C = img.shape

    if not TEST_REAL_LABELS:
        for m in min_size:
            for i in layers:
                num_connected_components, file_names = save_connected_components(mask[i], f'./data/narwhal/masks/', min_size=m)
                create_list_of_images_txt('./data/narwhal/', file_names)
                instance_dict('./data/narwhal/', file_names)
                
                if os.path.exists('./data/narwhal/images'):
                    shutil.rmtree('./data/narwhal/images')
                os.makedirs('./data/narwhal/images')
                for j in range(num_connected_components):
                    tif.imwrite("./data/narwhal/images/"+file_names[j]+".jpg", np.array([img[i,:,:]]*3))


                with open(path_to_test) as file:
                    code = file.read()
                exec(code)

                
                
                image_result_array = np.zeros((N, M))
                for k in range(num_connected_components):
                    image_result_path = './results/narwhal/'+file_names[k]+'-255.png'
                    image_result = Image.open(image_result_path)
                    image_result_array += np.array(image_result)
                
                image_result_array = np.where(image_result_array > threshold*255, 255, 0)
                
                plt.imsave(f'threed_results/Layer{i}.png', image_result_array, cmap='gray')  # Saving the array as a PNG file
            
            # create_tiff(threshold, m)




        


    else:
        labels_path = './threed_data/unet/'
        for m in min_size:
            for i in layers:
                label = np.array(Image.open(labels_path+f'mask_{i}.png'))
                num_connected_components, file_names = save_connected_components(label, f'./data/narwhal/masks/', min_size=m)
                create_list_of_images_txt('./data/narwhal/', file_names)
                instance_dict('./data/narwhal/', file_names)
                
                if os.path.exists('./data/narwhal/images'):
                    shutil.rmtree('./data/narwhal/images')
                os.makedirs('./data/narwhal/images')
                for j in range(num_connected_components):
                    tif.imwrite("./data/narwhal/images/"+file_names[j]+".jpg", np.array([img[i,:,:]]*3))


                with open(path_to_test) as file:
                    code = file.read()
                exec(code)

                
                
                image_result_array = np.zeros((N, M))
                for k in range(num_connected_components):
                    image_result_path = './results/narwhal/'+file_names[k]+'-255.png'
                    image_result = Image.open(image_result_path)
                    image_result_array += np.array(image_result)
                
                image_result_array = np.where(image_result_array > threshold*255, 255, 0)
                
                plt.imsave(f'threed_results/Layer{i}.png', image_result_array, cmap='gray')







    






