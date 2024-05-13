#%%
import numpy as np
import tifffile as tif
import pickle
#%%
#Select slices in tif
def select_slices(img, indexes):

    return np.array([img[indexes[i],:,:] for i in range(len(indexes))])

#Load tifs for img and mask
def load_data(path):
    img = np.rollaxis(tif.imread(path), axis=1)
    mask = np.rollaxis(tif.imread(path[:-4]+"_mask.tif"), axis=1)

    return img, mask

#Create list of images
def create_list_of_images_txt(data_path, names):
    f = open(data_path+"Slices/list/test.txt", "w")
    for i in range(len(names)):
        f.write(names[i]+"\n")


def instance_dict(data_path, names):
    dict = {names[i]+".png": {0:1} for i in range(len(names))}
    with open(data_path+"Slices/test_instances.pkl", 'wb') as file:
        pickle.dump(dict, file)
    

#%%

if __name__ == "__main__":

    data_path = "C:/Users/malth/Documents/DTU/Sjette Semester/Bachelor/Data/"
    tif_name = "scan_07_crop.tif"

    img, mask = load_data(data_path+tif_name)
    
    #slice_idxs = [79, 159, 262]
    slice_idxs = [400, 410, 420]
    names = [f"slice_{slice_idxs[i]}" for i in range(len(slice_idxs))]

    slices_img = select_slices(img, slice_idxs)
    slices_mask = select_slices(mask, slice_idxs)

    for i in range(len(slices_img)):
        tif.imwrite(data_path+"Slices/images/"+names[i]+".jpg", np.array([slices_img[i],slices_img[i],slices_img[i]]))
        tif.imwrite(data_path+"Slices/masks/"+names[i]+".png", slices_mask[i])

    create_list_of_images_txt(data_path, names)

    instance_dict(data_path, names)