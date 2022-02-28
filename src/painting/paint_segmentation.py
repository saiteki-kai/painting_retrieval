import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure as exposure
from medpy.filter.smoothing import anisotropic_diffusion
from skimage.morphology import convex_hull_image
from utils import resize_with_max_ratio

from src.config import MODEL_FOLDER, OUTPUT_FOLDER

from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
import segmentation_models as sm

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing('resnet34')


def false_colors(image, nb_components):
    # Create false color image
    colors = np.random.randint(0, 255, size=(nb_components , 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]
    #colors[0] = [207, 59, 0]
    return colors[image]

def hough_transform(image, rho=1, theta=np.pi / 180, threshold=30):
    '''
    parameters:
    @rho: Distance resolution of the accumulator in pixels.
    @theta: Angle resolution of the accumulator in radians.
    @threshold: Only lines that are greater than threshold will be returned.
    '''
    return cv2.HoughLines(image, rho=rho, theta=theta, threshold=threshold)

def draw_lines(image, lines, color=[255, 0, 0], thickness=2):
    line_length=100000 #impostare uguale alla diagonale
    for line in lines:
        for rho, theta in line:
            t = theta * 180 / np.pi
            t = np.mod(t, 90)
            k = 30

            if t >= 45 - k and t <= 45 + k:
                continue

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + line_length * (-b))
            y1 = int(y0 + line_length * (a))
            x2 = int(x0 - line_length * (-b))
            y2 = int(y0 - line_length * (a))
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image

def corner_detection(image, T=0.01, min_dist=50):
    corners_points = []
    corners_image = np.zeros_like(image)

    # detect corners
    pts_copy = cv2.goodFeaturesToTrack(image, 4, T, min_dist)
    pts_copy = np.int0([pt[0] for pt in pts_copy])

    # compute the distance from each corner to every other corner
    euclidian_dist = lambda pt1, pt2 : np.sqrt((pt2[1] - pt1[1]) ** 2 + (pt2[0] - pt1[0]) ** 2)

    # if the points are not 4 we return zero points!
    if len(pts_copy) == 4:

        # sort coordinates_tuples (tl, tr, bl, br)
        [h, w] = image.shape
        tl_index = np.asarray([euclidian_dist(pt, (0, 0)) for pt in pts_copy]).argmin()
        tr_index = np.asarray([euclidian_dist(pt, (w, 0)) for pt in pts_copy]).argmin()
        bl_index = np.asarray([euclidian_dist(pt, (0, h)) for pt in pts_copy]).argmin()
        br_index = np.asarray([euclidian_dist(pt, (w, h)) for pt in pts_copy]).argmin()
        corners_points = np.asarray([pts_copy[tl_index], pts_copy[tr_index], pts_copy[br_index], pts_copy[bl_index]])

        for pt in corners_points:
            cv2.circle(corners_image, tuple(pt), 10, 255, -1)

    return corners_points, corners_image

def warp_image(img, src):
    # invert points coordinates
    src = np.array([[point[0], point[1]] for point in src], dtype="float32")
    (tl, tr, br, bl) = src

    w1 = int(np.hypot(bl[0] - br[0], bl[1] - br[1]))
    w2 = int(np.hypot(tl[0] - tr[0], tl[1] - tr[1]))

    h1 = int(np.hypot(tl[0] - bl[0], tl[1] - bl[1]))
    h2 = int(np.hypot(tr[0] - br[0], tr[1] - br[1]))

    max_w = np.max([w1, w2])
    max_h = np.max([h1, h2])

    dst = np.array([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(img, M, (max_w, max_h))

    return warp


def hough_based_segmentation(image, folder, save=False):
    # resize
    resized = resize_with_max_ratio(image, 1024, 1024)

    # grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # denoise
    blur = anisotropic_diffusion(gray, kappa=30, gamma=0.25, option=1, niter=10).astype(np.uint8) # parametri paper

    # edge detection
    sx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)

    sx = cv2.convertScaleAbs(sx)
    sy = cv2.convertScaleAbs(sy)

    #normalize / scale the image 0 to 255
    out_x = cv2.normalize(sx, None, 0, 255, cv2.NORM_MINMAX)
    out_y = cv2.normalize(sy, None, 0, 255, cv2.NORM_MINMAX)

    #round up and type cast to int from float
    out_x = np.round(out_x).astype(np.uint8)
    out_y = np.round(out_y).astype(np.uint8)

    #use a threshold value
    _, out_x = cv2.threshold(out_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, out_y = cv2.threshold(out_y, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # horizontal hough transform
    h_lines = hough_transform(out_x, threshold=210)
    h_hough_out = draw_lines(np.zeros(sx.shape), h_lines, color=255, thickness=3)

    # vertical hough transform
    v_lines = hough_transform(out_y, threshold=210)
    v_hough_out = draw_lines(np.zeros(sy.shape), v_lines, color=255, thickness=3)

    # both horizontal and vertical hough
    out = np.zeros(resized.shape)
    out[:,:,0] = blur
    out[:,:,1] = blur
    out[:,:,2] = blur
    hough_out = draw_lines(out, v_lines, color=[0,0,255], thickness=2)
    hough_out = draw_lines(hough_out, h_lines, color=[0,0,255], thickness=2)

    if save:
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, folder, '0_resized.jpg'), image)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, folder, '1_gray.jpg'), gray)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, folder, '2_blur.jpg'), blur)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, folder, '3_sobelx.jpg'), out_x)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, folder, '4_sobely.jpg'), out_y)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, folder, '5_h_hough.jpg'), h_hough_out*255)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, folder, '6_v_hough_out.jpg'), v_hough_out*255)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, folder, '7_hough_out.jpg'), hough_out)

    return hough_out

def cnn_based_segmentation(image, model, folder, save=False):
    x = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (224, 224))
    x = img_to_array(x) / 255.0
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x, batch_size=1)
    mask = (pred.squeeze() >= 0.5).astype(np.uint8)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # labeling of the connected components [Note: range() starts from 1 since 0 is the background label.]
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])
    max_bw_label = (output == max_label).astype(np.uint8)

    # masked
    masked = cv2.bitwise_and(image, image, mask=max_bw_label)

    if save:
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, folder, '1_rgb.jpg'), image)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, folder, '2_mask.png'), mask * 255)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, folder, '3_labeling_connected_components.png'), false_colors(output, nb_components))
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, folder, '4_max_bw_label.png'), max_bw_label * 255)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, folder, '5_masked.png'), masked)

    return masked, max_bw_label


def paint_segmentation_pipeline(image, model=None, folder='.', save=False):

    # resize
    resized = resize_with_max_ratio(image, 1024, 1024)

    # paint segmentation (hough based or semantic segmentation)
    if model:
        _, mask = cnn_based_segmentation(resized, model, folder)
    else:
        _, mask = hough_based_segmentation(resized)

    # morphology operations
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=30, borderValue=0)

    # convex hull
    convex_hull = convex_hull_image(closing).astype(np.uint8)

    # centroid of my mask edited with morphological operations and convexhull
    x,y,w,h = cv2.boundingRect(convex_hull)
    centroid = (x+round(w/2), y+round(h/2))

    # edge detection
    (T, _) = cv2.threshold(convex_hull, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    canny = cv2.Canny(convex_hull, T * 0.5, T)

    # hough transform
    hough_lines = hough_transform(canny, threshold=30)
    hough_out = draw_lines(np.zeros(mask.shape), hough_lines, color=255, thickness=3)

    # invert hough mask
    invert_hough = 255 - hough_out
    invert_hough = (invert_hough > 0).astype(np.uint8)

    # labeling of the connected components [Note: range() starts from 1 since 0 is the background label.]
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(invert_hough, connectivity=4)
    dist = lambda p1, p2 : np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    min_label, min_dist = min([(i, dist(centroids[i], centroid)) for i in range(1, nb_components)], key=lambda x: x[1])
    min_dist_bw_label = (output == min_label).astype(np.uint8)

    _,_,w1,h1 = cv2.boundingRect(min_dist_bw_label)
    min_dist = min(w1, h1) * 0.75

    # corner detection
    corners_points, corners_image = corner_detection(min_dist_bw_label, T=0.01, min_dist=min_dist)

    if save:
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, folder, '6_close.png'), closing * 255)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, folder, '8_convex_hull.png'), convex_hull * 255)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, folder, '9_canny.png'), canny)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, folder, '10_hough.png'), hough_out * 255)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, folder, '11_invert_hough.png'), invert_hough * 255)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, folder, '12_labeling_connected_components.png'), false_colors(output, nb_components))
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, folder, '13_min_dist_bw_label.png'), min_dist_bw_label*255)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, folder, '14_harris.png'), corners_image)

    # if the corners detected are not 4 the warped image is not corrected!
    if len(corners_points) != 4:
        x,y,w,h = cv2.boundingRect(min_dist_bw_label)
        not_unwarped = cv2.bitwise_and(resized, resized, mask=min_dist_bw_label)
        not_unwarped = not_unwarped[y:y+h, x:x+w]

        if save:
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, folder, 'not_un_warped.jpg'), not_unwarped)

        return not_unwarped, min_dist_bw_label
    else:
        # image distortion correction
        un_warped = warp_image(resized, corners_points)
        
        if save:
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, folder, 'un_warped.jpg'), un_warped)

        return un_warped, min_dist_bw_label
