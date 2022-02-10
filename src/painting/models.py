import gc
import os

import tensorflow_addons as tfa  # Used in saved_model (don't delete)
from tensorflow import keras
from keras.backend import clear_session as clear_session_keras
from keras.models import load_model
from src.config import MODEL_FOLDER
from tensorflow.keras.models import Model
import segmentation_models as sm
from joblib import load as load_joblib

sm.set_framework("tf.keras")

# clear previous sessions
clear_session_keras()
gc.collect()

prediction_model = None
classification_model = None
segmentation_model = None

# For BOW
kmeans_model = None
scaler_model = None

def load_classification_model(model_name="resnet_model"):
    base_model = load_model(os.path.join(MODEL_FOLDER, model_name))
    model = Model(inputs=base_model.input, outputs=base_model.layers[-3].output)  # dense_4 (layer -3)
    del base_model

    global classification_model
    classification_model = model


def load_prediction_model(model_name="resnet_model"):
    base_model = load_model(os.path.join(MODEL_FOLDER, model_name))

    global prediction_model
    prediction_model = base_model


def load_segmentation_model(backbone="resnet34"):
    model = sm.Unet(backbone, encoder_weights="imagenet", encoder_freeze=True)
    model.load_weights(os.path.join(MODEL_FOLDER, "segmentation", "best_model_resnet34.h5"))

    global segmentation_model
    segmentation_model = model


def load_kmeans_model():

    kmean_path = os.path.join(MODEL_FOLDER, 'KMeans_BOW.joblib')
    model = load_joblib(kmean_path) 

    global kmeans_model
    kmeans_model = model



def load_scaler_model():
    scaler_path = os.path.join(MODEL_FOLDER, 'Scaler_BOW.joblib')
    model = load_joblib(scaler_path) 

    global scaler_model
    scaler_model = model


def load_models():
    load_classification_model()
    get_prediction_model()
    load_segmentation_model()


def get_classification_model():
    if classification_model is None:
        load_classification_model()
    return classification_model


def get_prediction_model():
    if prediction_model is None:
        load_prediction_model()
    return prediction_model


def get_segmentation_model():
    if segmentation_model is None:
        load_segmentation_model()
    return segmentation_model

def get_kmeans_model():
    if kmeans_model is None:
        load_kmeans_model()

    return kmeans_model

def get_scaler_model():
    if scaler_model is None:
        load_scaler_model()

    return scaler_model
