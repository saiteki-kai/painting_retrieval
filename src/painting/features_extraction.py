import os

import cv2 as cv
import numpy as np
from keras.models import load_model
import tensorflow_addons as tfa #Used in saved_model
from keras.applications.resnet import ResNet50
from keras.applications.resnet import \
    preprocess_input as preprocess_input_resnet

# should clear keras models
from keras.backend import clear_session as clear_session_keras
import gc

from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image as image_f

from ..painting.dataset import Dataset
from ..painting.utils import FEATURES_FOLDER, MODEL_FOLDER


def preprocess_cv2_image_resnet(image):
  image = cv.resize(image, (224, 224))
  image =  cv.cvtColor(image, cv.COLOR_BGR2RGB)
  image = Image.fromarray(image)
  image = image_f.img_to_array(image)
  image = np.expand_dims(image, axis = 0) #maybe not
  return preprocess_input_resnet(image)

def get_resnet50(image=None, dataset:Dataset=None, model_name='resnet_model'):
  """
    image: An image opened with OpenCV2. \
      Doesn't matter the size of the image. 
    dataset: A Dataset type. Doesn't matter the image_size of dataset. \
      If image not None it won't be readed. 
  """
  #Clear memory
  clear_session_keras()
  gc.collect()

  if image is not None:
    image = preprocess_cv2_image_resnet(image)
    print('Image resized into (224,224).' )

  elif dataset is not None:
    dim_dataset = dataset.length()
    print('Dataset dimension is: ' + str(dim_dataset) )

  """## Models"""
  #This give us all the model but the last layer
  #base_model = ResNet50(weights='imagenet')
  base_model = load_model( os.path.join(MODEL_FOLDER, model_name) )

  #base_model.summary() # Summary 
  #print([layer.name for layer in base_model.layers]) # Layer's name

  #model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
  model = Model(inputs=base_model.input, outputs=base_model.get_layer('dense').output)
  

  if image is not None:
    prediction = model.predict(image)
    
    #Clear memory
    clear_session_keras()
    gc.collect()
    del base_model
    del model

    return prediction.flatten()

  elif dataset is not None:
    for i in range(dim_dataset):
      im = dataset.get_image_by_index(i)
      im = preprocess_cv2_image_resnet(im)

      if i % 100 == 0:
        print("{} / {} " .format(i+1, dim_dataset))

      im = model.predict(im)
      #im = im.reshape(im.shape[1:])
      
      file_name = dataset.get_image_filename(i)

      folder_path = os.path.join(FEATURES_FOLDER, 'resnet50')

      if not os.path.exists( folder_path ):
        os.makedirs( folder_path )
      np.save(os.path.join(folder_path, file_name), im.flatten() )
      """ We just have to read the numpy.narray using numpy.load() """

    print("{} / {} " .format(dim_dataset, dim_dataset))
  
  #Clear memory
  clear_session_keras()
  gc.collect()
  del base_model
  del model

  return None
