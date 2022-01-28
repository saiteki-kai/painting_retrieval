from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image as image_vgg
import numpy as np
from tensorflow.keras.models import Model

# should clear keras models
from keras.backend import clear_session as clear_session_keras 
import gc

import cv2 as cv
import os
from dataset import Dataset
from utils import FEATURES_FOLDER 


def get_vgg(image_path=None, dataset:Dataset=None, level=1):
  """
    image:
    dataset: A Dataset type. Doesn't matter the image_size of dataset. \
      If image_path not None it won't be readed.
    level:
    save_path:
  """
  #Clear memory
  clear_session_keras()
  gc.collect()

  if image_path is not None:
    print('Image resized into (224,224).' )
    #input_size = (224, 224)
    #image = cv.resize(image, input_size)

    image = image_vgg.load_img( image_path, target_size=(224, 224, 3) )
    image = image_vgg.img_to_array(image)
    image = np.array([image])

  elif dataset is not None:
    dim_dataset = dataset.length()
    print('Dataset dimension is: ' + str(dim_dataset) )

  """## Models"""

  #This give us all the model but the last layer
  base_model = VGG16(weights='imagenet')

  """
  If we apply multiple cut we have multiple features. \
  We have to find where to cut to extract usefull information. \
  Use base_model.summary() to consult.
  """
  if level == 1:
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
  elif level == 2:
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
  else:
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
  

  if image_path is not None:
    prediction = model.predict(image)
    
    #Clear memory
    clear_session_keras()
    gc.collect()
    del model

    return prediction.flatten()

  elif dataset is not None:
    for i in range(dim_dataset):
      im = dataset.get_image_by_index_vgg(i)

      if i % 100 == 0:
        print("{} / {} " .format(i+1, dim_dataset))

      im = model.predict(im)
      im = im.reshape(im.shape[1:])
      
      file_name = dataset._image_list[i][dataset._image_list[i].rfind('/')+1:]

      folder_path = os.path.join(FEATURES_FOLDER, 'vgg')

      if not os.path.exists( folder_path ):
        os.makedirs( folder_path )
      np.save(os.path.join(folder_path, file_name), im.flatten() )
    """ We just have to read the numpy.narray using numpy.load() """

    print("{} / {} " .format(dim_dataset, dim_dataset))
  
  #Clear memory
  clear_session_keras()
  gc.collect()
  del model

  return None