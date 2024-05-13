'''
Author: Carl Nørlund
DTU Bachelor Project


Description of file:

Given a 500x500x500 image and mask in a .tif format, prepare train and trainval data for tosnet in this format:

-> Slices/
    -> images/
    -> list/
        train.txt
        trainval.txt
    -> masks/
    train_instances.pkl
    trainval_instances.pkl

where images contain the original images in slices and 
masks contain the mask for that slice and component in slice
formatted as such:

            slice_i_component_j
            
where i is #slice and j is #component
'''

import numpy as np
import tifffile as tif
import pickle
import cv2
import shutil
import os
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

#Select slices in tif
def select_slices(img, indexes):
    return np.array([img[indexes[i],:,:] for i in range(len(indexes))])

#Load tifs for img and mask
def load_data(img_path, mask_path):
    img = tif.imread(img_path)
    mask = np.rollaxis(tif.imread(mask_path), axis=1)
    return img, mask

#Create list of images
def create_list_of_images_txt(dir, names):
    f = open(dir, "w")
    for i in range(len(names)):
        f.write(names[i]+"\n")


def instance_dict(dir, names):
    dict = {names[i]: {0:1} for i in range(len(names))}
    with open(dir, 'wb') as file:
        pickle.dump(dict, file)
        
        
def save_connected_components(mask, output_dir, min_size=20):

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    original_num_labels, original_labels = cv2.connectedComponents(mask)
    num_labels = 0
    labels = np.zeros(original_labels.shape)
    for label_num in range(1, original_num_labels):  # Start from 1 to ignore the background

        if np.sum(original_labels==label_num) >= min_size:
            num_labels += 1
            labels[original_labels==label_num] = num_labels 
    
    for label_num in range(1, num_labels):  # Start from 1 to ignore the background
        # Create a mask for the current component
        component_mask = np.where(labels == label_num, 255, 0).astype('uint8')
        
        # Save the component as a TIFF image
        filename = os.path.join(output_dir, f"Component{label_num-1}.png")
        cv2.imwrite(filename, component_mask)
    return num_labels-1, [f"Component{label_num-1}" for label_num in range(1, num_labels)]


def invert_and_dilate(arr):
    arr = cv2.dilate(arr, kernel=np.ones((7,7)))
    arr = 255 - arr
    return arr

def load_slice(path):
    # Loads image slice and return a 2D NumPy array
    img = Image.open(path)
    return np.array(img)

def save_slice(arr, path):
    img = Image.fromarray(arr, mode='L')
    img.save(path)
    
    

#%%

if __name__ == "__main__":

    # Clear all existing relevant folders and files
    MAIN_DIRECTORY = './generating_data/data/Slices/'
    if os.path.exists(MAIN_DIRECTORY):
        shutil.rmtree(MAIN_DIRECTORY)
    os.makedirs(MAIN_DIRECTORY)
    os.makedirs(MAIN_DIRECTORY+'narwhal_train/images/')
    os.makedirs(MAIN_DIRECTORY+'narwhal_train/masks/')
    os.makedirs(MAIN_DIRECTORY+'narwhal_train/list/')
    
    # Make dirs for thin regions
    os.makedirs(MAIN_DIRECTORY+'thin_regions/narwhal_train_train/eval_mask/')
    os.makedirs(MAIN_DIRECTORY+'thin_regions/narwhal_train_train/gt_thin/')
    os.makedirs(MAIN_DIRECTORY+'thin_regions/narwhal_train_val/eval_mask/')
    os.makedirs(MAIN_DIRECTORY+'thin_regions/narwhal_train_val/gt_thin/')
        
    
    tifs = [2,3,4,5]
    train_names = []
    trainval_names = []
    for t in tifs:
        # Load tif files
        img_path = f'./generating_data/data/data_0{t}_v2.tif'
        mask_path = f'./generating_data/data/mask_0{t}_v2.tif'
        print(type(img_path))
        img, mask = load_data(img_path, mask_path)
        
        # Define train and validation slices
        all_slices = [i for i in range(500)]
        train_slice_idxs, trainval_slice_idxs = train_test_split(all_slices, test_size=0.1, random_state=0)
        train_slices_img = select_slices(img, train_slice_idxs)
        train_slices_mask = select_slices(mask, train_slice_idxs)
        trainval_slices_img = select_slices(img, trainval_slice_idxs)
        trainval_slices_mask = select_slices(mask, trainval_slice_idxs)
        
        # Components directory
        cc_dir = './generating_data/data/connected_components/'
        
        # Images directory
        img_dir = './generating_data/data/Slices/narwhal_train/images/'
        
        # Masks directory
        mask_dir = './generating_data/data/Slices/narwhal_train/masks/'
        
        # Thin-regions directory
        thin_dir = './generating_data/data/Slices/thin_regions/'
        
        
        '''
        TRAIN
        '''
        
        # Loop over train slices 
        for i in tqdm(range(len(train_slices_img))):
            
            # Find connected components for that slice and save to ./data/connected_components/ (cc_dir)
            num_cc, components = save_connected_components(train_slices_mask[i], cc_dir, min_size=20)

            # Save these images to ./data/Slice/masks/
            for j in range(num_cc):
                mask_slice_component = load_slice(cc_dir+f'Component{j}.png')
                save_slice(mask_slice_component, mask_dir + f'tif{t}_slice{i}_component{j}.png')
                save_slice(train_slices_img[i], img_dir + f'tif{t}_slice{i}_component{j}.jpg')
                train_names.append(f'tif{t}_slice{i}_component{j}.png')
                
                # Save to thin regions
                save_slice(mask_slice_component, thin_dir + f'narwhal_train_train/gt_thin/tif{t}_slice{i}_component{j}.png-0.png')
                
                # Invert and dilate mask and save it to eval_mask
                inv_and_dil_mask = invert_and_dilate(mask_slice_component)
                save_slice(inv_and_dil_mask, thin_dir + f'narwhal_train_train/eval_mask/tif{t}_slice{i}_component{j}.png-0.png')
                
                
    
        
        '''
        TRAINVAL
        '''
        
        # Loop over train slices 
        for i in tqdm(range(len(trainval_slices_img))):
            
            # Find connected components for that slice and save to ./data/connected_components/ (cc_dir)
            num_cc, components = save_connected_components(trainval_slices_mask[i], cc_dir, min_size=20)

            # Save these images to ./data/Slice/masks/
            for j in range(num_cc):
                mask_slice_component = load_slice(cc_dir+f'Component{j}.png')
                save_slice(mask_slice_component, mask_dir + f'tif{t}_slice{i}_component{j}.png')
                save_slice(train_slices_img[i], img_dir + f'tif{t}_slice{i}_component{j}.jpg')
                trainval_names.append(f'tif{t}_slice{i}_component{j}.png')
                
                # Save to thin regions
                save_slice(mask_slice_component, thin_dir + f'narwhal_train_val/gt_thin/tif{t}_slice{i}_component{j}.png-0.png')
                
                # Invert and dilate mask and save it to eval_mask
                inv_and_dil_mask = invert_and_dilate(mask_slice_component)
                save_slice(inv_and_dil_mask, thin_dir + f'narwhal_train_val/eval_mask/tif{t}_slice{i}_component{j}.png-0.png')
        
        
    
    # Create list of images 
    create_list_of_images_txt('./generating_data/data/Slices/narwhal_train/list/train.txt', train_names)

    # Create pickle file
    instance_dict('./generating_data/data/Slices/narwhal_train/train_instances.pkl', train_names)

    # Create list of images 
    create_list_of_images_txt('./generating_data/data/Slices/narwhal_train/list/trainval.txt', trainval_names)

    # Create pickle file
    instance_dict('./generating_data/data/Slices/narwhal_train/trainval_instances.pkl', trainval_names)
    

