import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import AffineTransform

from src.config import STANDARD_FEATURES_SIZE
from src.painting.descriptor import compute_orb
from src.painting.utils import load_features


def compute_score(matches, n_inliers):
    return n_inliers / len(matches)


def matching(kp1, des1, kp2, des2, threshold=0.75):
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf_matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.array([kp1[m.queryIdx].pt for m in matches], np.float32).reshape(-1, 2)
    dst_pts = np.array([kp2[m.trainIdx].pt for m in matches], np.float32).reshape(-1, 2)

    _, inliers = ransac((src_pts, dst_pts), AffineTransform, min_samples=4,
                        residual_threshold=8, max_trials=100)  # 10000
    n_inliers = np.sum(inliers)
    score = compute_score(matches, n_inliers)

    return score, score > threshold


def exact_matching(img):
    img = cv2.resize(img, STANDARD_FEATURES_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp1, des1 = compute_orb(gray)

    orb_kps = load_features('orb_kps')
    orb_desc = load_features('orb_desc')
    orb = zip(orb_kps, orb_desc)

    for i, (kp2, des2) in enumerate(orb):
        score, match = matching(kp1, des1, kp2, des2, threshold=0.8)

        if match:
            return i, score

    return None
