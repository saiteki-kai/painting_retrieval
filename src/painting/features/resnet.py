import cv2 as cv
import numpy as np
from keras.applications.resnet import preprocess_input as preprocess_input_resnet
import tensorflow_addons as tfa  # Used in saved_model (don't delete)
from src.painting.dataset import Dataset
from src.painting.models import get_classification_model, get_prediction_model


def preprocess_cv2_image_resnet(image):
    if image.shape[:2] != (224, 224):
        image = cv.resize(image, (224, 224))

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)
    return preprocess_input_resnet(image)


def predictions_dataset(dataset: Dataset, model):
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
    
    output: 1024 long features array
    """

    model = get_classification_model()

    if image is not None:
        image = preprocess_cv2_image_resnet(image)
        prediction = model.predict(image)
        return prediction.flatten()

    elif dataset is not None:
        print(f"Dataset dimension is: {dataset.length()}")
        return predictions_dataset(dataset, model)

    return None


def prediction_resnet50(image):
    base_model = get_prediction_model()

    if image is not None:
        image = preprocess_cv2_image_resnet(image)
        prediction = base_model.predict(image)
        return prediction.flatten()

    return None
