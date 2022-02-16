import os
import cv2
import joblib
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import ransac
from skimage.transform import AffineTransform

from src.config import FEATURES_FOLDER, DATASET_FOLDER
from src.painting.dataset import Dataset

# global matcher
#
#
# def get_matcher():
#     return matcher
#
#
# def init_matcher():
#     global matcher
#     FLANN_INDEX_KDTREE = 0
#
#     index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#
#     search_params = dict(checks=50)
#     matcher = cv2.FlannBasedMatcher(index_params, search_params)
#
#     # sift_descriptors = joblib.load(os.path.join(FEATURES_FOLDER, "sift.npy"), mmap_mode="r+")
#     # sift_descriptors = list([np.array(desc.reshape(-1, 128)) for desc in sift_descriptors])
#     #
#     # matcher.add(sift_descriptors)
#     # matcher.train()  # not enough space for indexing D:


def compute_score(real_matches, all_matches):
    return len(real_matches) / len(all_matches)


def flann_matching(des1, des2, lowe_ratio=0.7):
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = matcher.knnMatch(des1, des2, k=2)

    # Filter matches using the Lowe's ratio test
    ratio_thresh = lowe_ratio
    good_matches = []
    for i, values in enumerate(knn_matches):
        if len(values) == 2:
            if values[0].distance < ratio_thresh * values[1].distance:
                good_matches.append(values[0])

    return good_matches, knn_matches


def method2(kp1, des1, kp2, des2):
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf_matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.array([kp1[m.queryIdx].pt for m in matches], np.float32).reshape(-1, 2)
    dst_pts = np.array([kp2[m.trainIdx].pt for m in matches], np.float32).reshape(-1, 2)

    MIN_SAMPLES = 50

    if len(src_pts) < MIN_SAMPLES:
        print("Not enough points")
        return None

    _, inliers = ransac((src_pts, dst_pts), AffineTransform, min_samples=MIN_SAMPLES,
                        residual_threshold=8, max_trials=100)  # 10000


def exact_matching(img, threshold=0.35, lowe_ratio=0.7):
    img = cv2.resize(img, (512, 512))
    des1 = compute_sift(img, dense=False)
    des1 = des1.reshape((-1, 128))

    # real_matches, all_matches = flann_matching(des1, None)
    # score = compute_score(real_matches, all_matches)
    # print(score)
    #
    # if score > threshold:
    #     return score

    sift_descriptors = joblib.load(os.path.join(FEATURES_FOLDER, "sift.npy"), mmap_mode="r+")

    max_score = (-1, 0)
    for i, des2 in enumerate(sift_descriptors):
        if des2 is not None:
            des2 = des2[~np.isnan(des2)]
            if des2.shape[0] > 128:
                des2 = des2.reshape((-1, 128))
                real_matches, all_matches = flann_matching(des1, des2, lowe_ratio=lowe_ratio)
                score = compute_score(real_matches, all_matches)

                if score > max_score[1]:
                    max_score = (i, score)

    print(max_score)
    if max_score[1] > threshold:
        ds = Dataset(DATASET_FOLDER)
        return max_score[1], ds.get_image_filepath(max_score[0])

    return None, None


def compute_dense_keypoints(img, stride):
    """Define a grid of keypoints"""
    keypoints = [cv2.KeyPoint(x, y, stride)
                 for y in range(stride, img.shape[0] - stride, stride)
                 for x in range(stride, img.shape[1] - stride, stride)]
    return keypoints


def compute_sift(img, stride=32, dense=True, return_keypoints=False):
    sift = cv2.SIFT_create(100)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if dense:
        kp = compute_dense_keypoints(gray, stride)
        _, des = sift.compute(img, np.asarray(kp))
    else:
        kp, des = sift.detectAndCompute(img, None)
    des = np.asarray(des).flatten()

    if len(des) > 100 * 128:
        des = des[:100 * 128]

    if return_keypoints:
        return des.flatten(), kp

    return des.flatten()
