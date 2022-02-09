import os

import cv2 as cv
import joblib
import matplotlib.pyplot as plt

from src.config import LIST_OF_FEATURES_IMPLEMENTED, FEATURES_FOLDER


def load_features(feature):
    if feature != "combined" and feature not in LIST_OF_FEATURES_IMPLEMENTED:
        raise ValueError(f"unrecognized feature: '{feature}'")

    filepath = os.path.join(FEATURES_FOLDER, f"{feature}.npy")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"feature '{feature}' not computed")

    return joblib.load(filepath, mmap_mode="r+")


def plot_image(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # displaying image
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def resize_with_max_ratio(image, max_h, max_w):
    if len(image.shape) > 2:
        w, h, ch = image.shape
    else:
        w, h = image.shape

    if (h > max_h) or (w > max_w):
        rate = max_h / h
        rate_w = w * rate
        if rate_w > max_h:
            rate = max_h / w
        image = cv.resize(
            image, (int(h * rate), int(w * rate)), interpolation=cv.INTER_CUBIC
        )
    return image


def load_images_from_folder(folder):
    filenames = os.listdir(folder)

    images = []
    for filename in filenames:
        image = cv.imread(os.path.join(folder, filename))
        if image is not None:
            images.append(image)
    return images


def load_images_and_gt_from_folder(folder):
    images_path = os.path.join(folder, 'images')
    gt_path = os.path.join(folder, 'masks')
    filenames1 = os.listdir(gt_path)

    images = []
    masks = []
    for i in range(len(filenames1)):
        image = cv.imread(os.path.join(images_path, filenames1[i]).replace('.png', '.jpg'))
        mask = cv.imread(os.path.join(gt_path, filenames1[i]))
        if image is not None:
            images.append(image)
        if mask is not None:
            masks.append(mask)
    return images, masks
