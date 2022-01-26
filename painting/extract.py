import os

import numpy as np

from dataset import Dataset
from descriptor import compute_feature
from utils import TRAIN_FOLDER, TEST_FOLDER, RETRIEVAL_FOLDER, FEATURES_FOLDER 




def compute_descriptor(dataset: Dataset, descriptor_name):
    """
    :param dataset: Dataset instance
    :param descriptor_name: the feature to compute for each image
    """

    N = dataset.length()
    folder_path = os.path.join(FEATURES_FOLDER, descriptor_name)

    for idx, img in enumerate(dataset.images()):
        f = compute_feature(img, descriptor_name)
        # print(f"{img.shape} -> {f.shape}")
        file_name = ds._image_list[idx][ds._image_list[idx].rfind('/')+1:]

        if not os.path.exists( folder_path ):
            os.makedirs( folder_path )
        np.save(os.path.join(folder_path, file_name), f)

        if descriptor_name == 'ccv':
            print( str(idx) +"/"+ str(N) )
        else:
            if idx % 1000 == 0:
                print( str(idx) +"/"+ str(N) )
        
        
    print( str(N) +"/"+ str(N) )


if __name__ == "__main__":

    ds = Dataset(RETRIEVAL_FOLDER, (512, 512))

    list_of_features = [
        "hog", "hsv_hist", "lbp", 
        "rgb_hist", "bow_sift",
        "dct"] #"ccv"

    for feature in list_of_features:
       print("Computing: " + feature)
    
