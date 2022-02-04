import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform
import numpy as np
import os
from dataset import Dataset
from utils import TRAIN_FOLDER, FEATURES_FOLDER
from utils import STANDARD_FEATURES_SIZE



def compute_score(matches, n_inliers):
    return (n_inliers/len(matches))

def matching(kp1, des1, kp2, des2, T=0.75):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x : x.distance)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,2)

    model, inliers = ransac((src_pts, dst_pts), AffineTransform, min_samples=4, residual_threshold=8, max_trials=100) #10000
    n_inliers = np.sum(inliers)

    return compute_score(matches, n_inliers) > T

def preprocess_orb():
    ds = Dataset(DATASET_FOLDER, STANDARD_FEATURES_SIZE)
    compute_descriptor(ds, '', (500, ))
    compute_descriptor(ds, '', (500, 32))

#preprocess_orb()

def exact_matching():
    query = cv2.imread('/home/lfx/Desktop/Painting Detection/Data/images/37.jpg')
    query = cv2.resize(query, STANDARD_FEATURES_SIZE)
    query_gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    kp1, des1 = compute_orb(query_gray)

    orb_kps = load_features('orb_kps')
    orb_desc = load_features('orb_desc')
    orb = zip(orb_kps, orb_desc)

    for kp2, des2 in enumerate(orb):
        if exact_matching(kp1, des1, kp2, des2, T=0.8):
            print('An exact macth!!!!')

