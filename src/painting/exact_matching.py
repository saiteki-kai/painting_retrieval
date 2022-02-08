import glob
import os.path

import cv2
import numpy as np
import pickle
from joblib import Parallel, delayed, load, dump
from skimage.measure import ransac
from skimage.transform import AffineTransform

from src.config import FEATURES_FOLDER
from src.painting.dataset import Dataset
from src.painting.utils import resize_with_max_ratio


def compute_score(matches, n_inliers):
    return n_inliers / len(matches)


def matching(kp1, des1, kp2, des2, threshold=0.8):
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

    if inliers is not None and len(inliers) > 0:
        n_inliers = np.sum(inliers)
        score = compute_score(matches, n_inliers)

        if score > threshold:
            return score

    return None


def exact_matching(img):
    # img = resize_with_max_ratio(img, 512, 512)
    img = cv2.resize(img, (512, 512))
    kp1, des1 = compute_orb(img)

    if len(kp1) == 0:
        return None

    orb_files = glob.glob(os.path.join(FEATURES_FOLDER, "orb", "*.pickle"))

    for filepath in orb_files:
        idx = int(os.path.splitext(os.path.split(filepath)[-1])[0])
        with open(filepath, "rb") as f:
            orb = pickle.load(f)

            kps = []
            des = []
            for o in orb:
                kp = cv2.KeyPoint(x=o[0][0], y=o[0][1], size=o[1], angle=o[2], response=o[3], octave=o[4],
                                  class_id=o[5])
                kps.append(kp)
                des.append(o[6])
            kp2 = np.asarray(kps)
            des2 = np.asarray(des)

            match = matching(kp1, des1, kp2, des2, threshold=0.8)

            if match is not None:
                return idx, match

    return None


def compute_orb(img, n_features=500):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=n_features)
    kp, des = orb.detectAndCompute(gray, None)

    kp = np.asarray(kp)
    des = np.asarray(des)

    return kp, des


def compute_orb_descriptor(dataset: Dataset):
    for idx, img in enumerate(dataset.images()):
        kp, desc = compute_orb(img)
        if len(kp) > 0 and len(desc) > 0:
            orb = [(p.pt, p.size, p.angle, p.response, p.octave, p.class_id, d) for (p, d) in zip(kp, desc)]
            with open(os.path.join(FEATURES_FOLDER, "orb", f"{idx}.pickle"), 'wb') as f:
                pickle.dump(orb, f)
        else:
            print(idx)
