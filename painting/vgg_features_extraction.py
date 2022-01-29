from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image as image_vgg

# should clear keras models
from keras.backend import clear_session as clear_session_keras 
import gc

import cv2 as cv
from PIL import Image
import numpy as np
import os
from dataset import Dataset
from utils import FEATURES_FOLDER 

def preprocess_cv2_image(image):
  #IT WORKS BUT WITH DIFFERENT RESULTS
  image = cv.resize(image, (224, 224))
  image =  cv.cvtColor(image, cv.COLOR_BGR2RGB)
  image = Image.fromarray(image)
  #image = image.resize((224, 224))
  image = image_vgg.img_to_array(image)
  image = np.expand_dims(image, axis = 0)
  return preprocess_input(image)

def preprocess_vgg_standard(image):
  #STANDARD METHOD
  image = image_vgg.load_img(image, target_size=(224, 224, 3) )
  image = image_vgg.img_to_array(image)
  image = np.array([image])
  return preprocess_input(image)

def get_vgg(image=None, dataset:Dataset=None, level=1):
  """
    image:
    dataset: A Dataset type. Doesn't matter the image_size of dataset. \
      If image not None it won't be readed. 
    level: level=1 cut at block4_pool,\
          level=1 cut at block5_pool,\
          else cut at fc2. 
  """
  #Clear memory
  clear_session_keras()
  gc.collect()

  if image is not None:
    image = preprocess_cv2_image(image)
    print('Image resized into (224,224).' )

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
  

  if image is not None:
    prediction = model.predict(image)
    
    #Clear memory
    clear_session_keras()
    gc.collect()
    del model

    return prediction.flatten()

  elif dataset is not None:
    for i in range(dim_dataset):
      im = dataset.get_image_by_index(i)
      im = preprocess_cv2_image(im)

      if i % 100 == 0:
        print("{} / {} " .format(i+1, dim_dataset))

      im = model.predict(im)
      #im = im.reshape(im.shape[1:])
      
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