import os

import cv2 as cv
import numpy as np
from PIL import Image
from keras.applications.resnet import preprocess_input as preprocess_input_resnet

# Clear keras memory
from keras.backend import clear_session as clear_session_keras
import gc

from keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image as image_f
import tensorflow_addons as tfa  # Used in saved_model (don't delete)
from src.config import FEATURES_FOLDER, MODEL_FOLDER
from src.painting.dataset import Dataset
from src.painting.models import get_classification_model


def preprocess_cv2_image_resnet(image):
    image = cv.resize(image, (224, 224))
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = image_f.img_to_array(image)
    image = np.expand_dims(image, axis=0)  # maybe not
    return preprocess_input_resnet(image)


def predictions_gen(dataset: Dataset, model):
    for im in dataset.images():
        im = preprocess_cv2_image_resnet(im)
        im = model.predict(im)
        yield im.flatten()


def get_resnet50(image=None, dataset: Dataset = None):
    """
    image: An image opened with OpenCV2.
        Doesn't matter the size of the image.
    dataset: A Dataset type. Doesn't matter the image_size of dataset.
        If image not None it won't be readed.
    """

    model = get_classification_model()
    print(model)

    if image is not None:
        image = preprocess_cv2_image_resnet(image)
        print("Image resized into (224,224)")
        prediction = model.predict(image)
        return prediction.flatten()

    elif dataset is not None:
        print(f"Dataset dimension is: {dataset.length()}")
        return predictions_gen(dataset, model)

    return None
