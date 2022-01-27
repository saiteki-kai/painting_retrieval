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


def example():
    # Load the image
    image1 = cv2.imread('./64.jpg')
    test_image = cv2.imread('./15.jpg')
    #test_image = image1

    training_image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    training_gray = cv2.cvtColor(training_image, cv2.COLOR_RGB2GRAY)

    num_rows, num_cols = test_image.shape[:2]
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 1)
    test_image = cv2.warpAffine(test_image, rotation_matrix, (num_cols, num_rows))

    test_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)



    # Display traning image and testing image
    fx, plots = plt.subplots(1, 2, figsize=(8,4))

    plots[0].set_title("Training Image")
    plots[0].imshow(training_image)

    plots[1].set_title("Testing Image")
    plots[1].imshow(test_image)
    plt.show()


    orb = cv2.ORB_create()

    train_keypoints, train_descriptor = orb.detectAndCompute(training_gray, None)
    test_keypoints, test_descriptor = orb.detectAndCompute(test_gray, None)

    keypoints_without_size = np.copy(training_image)
    keypoints_with_size = np.copy(training_image)

    cv2.drawKeypoints(training_image, train_keypoints, keypoints_without_size, color = (0, 255, 0))
    cv2.drawKeypoints(training_image, train_keypoints, keypoints_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display image with and without keypoints size
    fx, plots = plt.subplots(1, 2, figsize=(8,4))
    plots[0].set_title("Train keypoints With Size")
    plots[0].imshow(keypoints_with_size, cmap='gray')
    plots[1].set_title("Train keypoints Without Size")
    plots[1].imshow(keypoints_without_size, cmap='gray')
    plt.show()

    print("Number of Keypoints Detected In The Training Image: ", len(train_keypoints))
    print("Number of Keypoints Detected In The Query Image: ", len(test_keypoints))

    # Brute Force Matcher / Lowie's test / FLANN based Matcher?
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = bf.match(train_descriptor, test_descriptor)
    matches = sorted(matches, key = lambda x : x.distance)

    result = cv2.drawMatches(training_image, train_keypoints, test_gray, test_keypoints, matches, test_gray, flags = 2)
    plt.rcParams['figure.figsize'] = [8.0, 4.0]
    plt.title('Best Matching Points')
    plt.imshow(result)
    plt.show()

    ## extract the matched keypoints
    src_pts = np.float32([train_keypoints[m.queryIdx].pt for m in matches]).reshape(-1,2)
    dst_pts = np.float32([test_keypoints[m.trainIdx].pt for m in matches]).reshape(-1,2)

    print(np.asarray(src_pts).shape)
    print(np.asarray(dst_pts).shape)

    # Ransac
    model, inliers = ransac((src_pts, dst_pts), AffineTransform, min_samples=4, residual_threshold=8, max_trials=10000)
    n_inliers = np.sum(inliers)

    inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
    inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
    placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
    result = cv2.drawMatches(training_image, inlier_keypoints_left, test_image, inlier_keypoints_right, placeholder_matches, None)

    # Display the best matching points
    plt.rcParams['figure.figsize'] = [8.0, 4.0]
    plt.title('Best Matching Points')
    plt.imshow(result)
    plt.show()

    # Print total number of matching points between the training and query images
    print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(matches))

    print(compute_score(matches, n_inliers))



def compute_score(matches, n_inliers):
    return (n_inliers/len(matches))

def exact_matching(img1, img2, T=0.75):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x : x.distance)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,2)

    model, inliers = ransac((src_pts, dst_pts), AffineTransform, min_samples=4, residual_threshold=8, max_trials=100) #10000
    n_inliers = np.sum(inliers)

    return compute_score(matches, n_inliers) > T

def compute_orb(img, n_features=500):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=n_features)
    kp1, des1 = orb.detectAndCompute(gray, None)
    kp1 = np.asarray(kp1)
    des1 = np.asarray(des1)
    return kp1, des1

def compute_descriptor(dataset: Dataset, feature_shape, descriptor_fn):
    """
    :param dataset: Dataset instance
    :param feature_shape: feature shape
    :param descriptor_fn: function that computes the feature for each image
    :return features matrix
    """

    if not isinstance(feature_shape, tuple):
        raise ValueError("'features_size' must be a tuple")

    if not callable(descriptor_fn):
        raise ValueError("'descriptor_fn' must be callable")

    N = dataset.length()
    features = np.zeros((N, *feature_shape))
    for idx, img in enumerate(dataset.images()):
        f = descriptor_fn(img)
        features[idx, :] = f
        # print(f"{img.shape} -> {f.shape}")
        if idx % 1000 == 0:
            print(idx)
    return features

def preprocess_orb():
    ds = Dataset('/home/lfx/Desktop/Painting Detection/train/images/', STANDARD_FEATURES_SIZE)
    features_kps = compute_descriptor(ds, (500, ), lambda x : compute_orb(x, n_features=500)[0])
    features_des = compute_descriptor(ds, (500, 32), lambda x : compute_orb(x, n_features=500)[1])
    np.save(os.path.join('.', "orb_kps"), features_kps)
    np.save(os.path.join('.', "orb_des"), features_des)


query = cv2.imread('./3.jpg')
query = cv2.resize(query, (128, 128))
ds = Dataset('/home/lfx/Desktop/Painting Detection/train/images/', (128, 128))
for idx, img in enumerate(ds.images()):
    if idx % 10 == 0:
        print(idx)

    if exact_matching(query, img, T=0.5):
        plt.title('An exact macth!!!!')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
