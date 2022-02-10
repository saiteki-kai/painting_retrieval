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

def readImage(img_path):
    img = cv2.imread(img_path, 0)
    return cv2.resize(img,(150,150)) #Speed up, just for now

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
    
    for n_img in range(image_count):    

        img = ds.get_image_by_index(n_img)

        if img is not None:
            des = getDescriptors(sift, img)
            if des is not None:
                descriptor_list.append(des)

        if (verbose is not None) and (n_img % 1000 == 0):
            print(f"{n_img+1} / {image_count}")

    if verbose is not None:
        print(f"{image_count} / {image_count}")

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
    dump(kmeans, scaler_path) 
    im_features = scale.transform(im_features)
    if verbose is not None:
        print("Train images normalized.")

    return kmeans, scale, im_features

def featuresBOW(img, n_clusters, kmeans=None, scale:StandardScaler=None, verbose=None):

    if kmeans is None:
        kmean_path = os.path.join(MODEL_FOLDER, 'KMeans_BOW.joblib')
        kmeans = load(kmean_path) 
    if scale is None:
        scaler_path = os.path.join(MODEL_FOLDER, 'Scaler_BOW.joblib')
        scale = load(scaler_path)  

    sift = cv2.SIFT_create()

    descriptor = getDescriptors(sift, img)
    descriptors = vstackDescriptors([descriptor])

    count = 1
    features = extractFeatures(kmeans, [descriptor], count, n_clusters)

    features = scale.transform(features)
    
    if verbose is not None:
        print("Execution done.")
    
    return descriptors, features[0]



