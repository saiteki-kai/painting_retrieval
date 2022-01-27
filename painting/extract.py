import os

import numpy as np

from dataset import Dataset
from descriptor import compute_feature
from utils import TRAIN_FOLDER, TEST_FOLDER, RETRIEVAL_FOLDER, FEATURES_FOLDER 
from utils import STANDARD_FEATURES_SIZE
from utils import LIST_OF_FEATURES_IMPLEMENTED




def compute_descriptor(dataset: Dataset, descriptor_name, vgg_level=1):
    """
    :param dataset: Dataset instance
    :param descriptor_name: the feature to compute for each image
    """
    if descriptor_name == 'vgg':
        compute_feature(dataset, descriptor_name, vgg_level=vgg_level)
        return

    N = dataset.length()
    folder_path = os.path.join(FEATURES_FOLDER, descriptor_name)

    for idx, img in enumerate(dataset.images()):
        f = compute_feature(img, descriptor_name)
        # print(f"{img.shape} -> {f.shape}")
        file_name = ds._image_list[idx][ds._image_list[idx].rfind('/')+1:]

        if not os.path.exists( folder_path ):
            os.makedirs( folder_path )
        np.save(os.path.join(folder_path, file_name), f)

        #Just to track the progress
        if descriptor_name == 'ccv':
            print( str(idx) +"/"+ str(N) )
        else:
            if idx % 1000 == 0:
                print("{} / {} " .format(idx, N))
        
    #Just to tell if finished or not
    print("{} / {} " .format(N, N))


if __name__ == "__main__":

    ds = Dataset(RETRIEVAL_FOLDER, STANDARD_FEATURES_SIZE)

    # We avoit to do ccv for now (too slow)
    list_of_features = [x for x in LIST_OF_FEATURES_IMPLEMENTED if (x != 'ccv' and x!= 'vgg')]

    for feature in list_of_features:
       print("Computing: " + feature)
       compute_descriptor(ds, feature)

    # VGG features are special for now, we want to be able to specify the level
    # Once a level is fixed we can modify it and it will be the same of
    # the others
    ds_vgg = Dataset(RETRIEVAL_FOLDER, (224, 224))
    print("Computing: vgg")
    compute_descriptor(ds_vgg, "vgg", vgg_level=3)
