import cv2
import numpy as np 
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from src.painting.dataset import Dataset
from src.config import LIST_GENRE, MODEL_FOLDER, DATASET_FOLDER


NUM_CLASSES = len(LIST_GENRE)

def getDescriptors(sift, img):
    kp, des = sift.detectAndCompute(img, None)
    return des

def vstackDescriptors(descriptor_list):
    descriptors = np.array(descriptor_list[0])
    for descriptor in descriptor_list[1:]:
        descriptors = np.vstack((descriptors, descriptor)) 

    return descriptors

def clusterDescriptors(descriptors, n_clusters):
    kmeans = KMeans(n_clusters = n_clusters).fit(descriptors)
    kmean_path = os.path.join(MODEL_FOLDER, 'KMeans_BOW.joblib')
    dump(kmeans, kmean_path) 
    return kmeans

def extractFeatures(kmeans, descriptor_list, image_count, n_clusters):
    im_features = np.array([np.zeros(n_clusters) for i in range(image_count)])
    for i in range(image_count):
        if i % 10 == 0:
            print(f"{i+1} / {image_count}")
        for j in range(len(descriptor_list[i])):
            feature = descriptor_list[i][j]
            feature = feature.reshape(1, 128)
            idx = kmeans.predict(feature)
            im_features[i][idx] += 1
    print(f"{image_count} / {image_count}")

    return im_features

def normalizeFeatures(scale, features):
    return scale.transform(features)

def trainModel(ds:Dataset, n_clusters, verbose=None):

    #images = ds.images()
    sift =  cv2.SIFT_create() 
    descriptor_list = []
    image_count = ds.length()
    
    actual_image_count = 0
    
    for n_img in range(image_count):    

        img = ds.get_image_by_index(n_img)

        if img is not None:
            des = getDescriptors(sift, img)
            if des is not None:
                descriptor_list.append(des)
                actual_image_count += 1

        if (verbose is not None) and (n_img % 1000 == 0):
            print(f"{n_img+1} / {image_count}")

    if verbose is not None:
        print(f"{image_count} / {image_count}")

    image_count = actual_image_count

    descriptors = vstackDescriptors(descriptor_list)
    if verbose is not None:
        print("Descriptors vstacked.")
        print("Descriptors start clustering.")

    kmeans = clusterDescriptors(descriptors, n_clusters)
    if verbose is not None:
        print("Descriptors clustered.")
        print("Images starting features extraction.")

    im_features = extractFeatures(kmeans, descriptor_list, image_count, n_clusters)
    if verbose is not None:
        print("Images features extracted.")
        print("Train images start normalizing.")
    
    scale = StandardScaler().fit(im_features)
    scaler_path = os.path.join(MODEL_FOLDER, 'Scaler_BOW.joblib')
    dump(scale, scaler_path) 
    im_features = scale.transform(im_features)
    if verbose is not None:
        print("Train images normalized.")

    return kmeans, scale, im_features

def featuresBOW(img, n_clusters, kmeans=None, scale:StandardScaler=None, verbose=None):
    if verbose is not None:
        print(f"Test image size { img.shape }")

    if kmeans is None:
        if verbose is not None:
            print("Getting KMeans model")
        kmean_path = os.path.join(MODEL_FOLDER, 'KMeans_BOW.joblib')
        kmeans = load(kmean_path) 
    if scale is None:
        if verbose is not None:
            print("Getting Scaler model")
        scaler_path = os.path.join(MODEL_FOLDER, 'Scaler_BOW.joblib')
        scale = load(scaler_path)  

    sift = cv2.SIFT_create()

    if verbose is not None:
        print("Getting descriptors of test.")

    descriptor_list = []
    des = getDescriptors(sift, img)
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



