import cv2
import numpy as np 
import os
import time
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from src.painting.dataset import Dataset
from src.config import LIST_GENRE, MODEL_FOLDER
from src.painting.models import get_kmeans_model, get_scaler_model

TRAIN_SIZE = (128, 128)
NUM_CLASSES = len(LIST_GENRE)
N_CLUSTERS_DEFAULT = 200

def getDescriptors(sift, img):
    kp, des = sift.detectAndCompute(img, None)
    return des

def vstackDescriptors(descriptor_list):
    descriptors = np.array(descriptor_list[0])
    for descriptor in descriptor_list[1:]:
        descriptors = np.vstack((descriptors, descriptor)) 

    return descriptors

def clusterDescriptors(descriptors, n_clusters):
    kmeans = MiniBatchKMeans(n_clusters = n_clusters).fit(descriptors)
    kmean_path = os.path.join(MODEL_FOLDER, 'KMeans_BOW.joblib')
    dump(kmeans, kmean_path) 
    return kmeans

def extractFeatures(kmeans, descriptor_list, image_count, n_clusters, verbose=None):
    im_features = np.array([np.zeros(n_clusters) for i in range(image_count)])
    for i in range(image_count):
        if (verbose is not None) and (i % 10 == 0):
            print(f"{i+1} / {image_count}")
        for j in range(len(descriptor_list[i])):
            feature = descriptor_list[i][j]
            feature = feature.reshape(1, 128)
            idx = kmeans.predict(feature)
            im_features[i][idx] += 1
    if verbose is not None:
        print(f"{image_count} / {image_count}")

    return im_features

def normalizeFeatures(scale, features):
    return scale.transform(features)

def trainModelBOW(ds:Dataset, n_clusters=N_CLUSTERS_DEFAULT, verbose=None):

    sift =  cv2.SIFT_create() 
    descriptor_list = []
    image_count = ds.length()
    
    actual_image_count = 0
    if verbose is not None:
        start = time.time()
    for n_img in range(image_count):    

        img = ds.get_image_by_index(n_img)
        if img.shape[:2] != TRAIN_SIZE:
            img = cv2.resize(img, TRAIN_SIZE)
        
        if img is not None:
            des = getDescriptors(sift, img)
            if des is not None:
                descriptor_list.append(des)
                actual_image_count += 1

        if (verbose is not None) and (n_img % 1000 == 0):
            print(f"{n_img+1} / {image_count}")

    if verbose is not None:
        print(f"{image_count} / {image_count}")
        end = time.time()
        print(f"Sift detectAndCompute time: {end - start}")

    image_count = actual_image_count

    descriptors = vstackDescriptors(descriptor_list)
    if verbose is not None:
        print("Descriptors vstacked.")
        print("Descriptors start clustering.")

    kmeans = clusterDescriptors(descriptors, n_clusters)
    if verbose is not None:
        print("Descriptors clustered.")
        print("Images starting features extraction.")
        start = time.time()

    im_features = extractFeatures(kmeans, descriptor_list, image_count, n_clusters, verbose)
    if verbose is not None:
        end = time.time()
        print(f"Images features extracted in {end - start}.")
        print(f"Average of {(end - start)/image_count} per image.")
        print("Train images start normalizing.")
    
    scale = StandardScaler().fit(im_features)
    scaler_path = os.path.join(MODEL_FOLDER, 'Scaler_BOW.joblib')
    dump(scale, scaler_path) 
    im_features = scale.transform(im_features)
    if verbose is not None:
        print("Train images normalized.")

    return kmeans, scale, im_features

def featuresBOW(img, n_clusters=N_CLUSTERS_DEFAULT, kmeans=None, scale:StandardScaler=None, verbose=None):
    if verbose is not None:
        print(f"Test image size { img.shape }")
        if img.shape[:2] != TRAIN_SIZE:
            print(f"Reshaped into {TRAIN_SIZE}.")

    if img.shape[:2] != TRAIN_SIZE:
        img = cv2.resize(img, TRAIN_SIZE)

    if kmeans is None:
        if verbose is not None:
            print("Getting KMeans model")
        kmeans = get_kmeans_model()
    if scale is None:
        if verbose is not None:
            print("Getting Scaler model")
        scale = get_scaler_model()

    sift = cv2.SIFT_create()

    if verbose is not None:
        print("Getting descriptors of test.")

    descriptor_list = []
    des = getDescriptors(sift, img)
    if des is None:
        return [0] * n_clusters
    descriptor_list.append(des)

    if verbose is not None:
        print("Descriptors of test done.")
    if verbose is not None:
        print("Images starting features extraction.")

    count = 1
    im_features = extractFeatures(kmeans, descriptor_list, count, n_clusters)
    if verbose is not None:
        print("Images features extracted.")
        print("Scaler transforming.")

    im_features = scale.transform(im_features)
    
    if verbose is not None:
        print("Execution done.")
    
    return im_features[0]



