import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure as exposure
from medpy.filter.smoothing import anisotropic_diffusion
from skimage.morphology import convex_hull_image

from src.config import MODEL_FOLDER, OUTPUT_FOLDER

from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
import segmentation_models as sm

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing('resnet34')

def sobel(gray, ksize):
    # apply sobel derivatives
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    # optionally normalize to range 0 to 255 for proper display
    sobelx_norm = exposure.rescale_intensity(sobelx, in_range='image', out_range=(0, 255)) \
        .clip(0, 255).astype(np.uint8)
    sobely_norm = exposure.rescale_intensity(sobelx, in_range='image', out_range=(0, 255)) \
        .clip(0, 255).astype(np.uint8)

    # square
    sobelx2 = cv2.multiply(sobelx, sobelx)
    sobely2 = cv2.multiply(sobely, sobely)

    # normalize to range 0 to 255 and clip negatives
    sobelx2 = exposure.rescale_intensity(sobelx2, in_range='image', out_range=(0, 255)) \
        .clip(0,255).astype(np.uint8)
    sobely2 = exposure.rescale_intensity(sobely2, in_range='image', out_range=(0, 255)) \
        .clip(0,255).astype(np.uint8)

    # add together and take square root
    sobel_magnitude = cv2.sqrt(sobelx2 + sobely2)
    sobel_magnitude = exposure.rescale_intensity(sobel_magnitude, in_range='image', out_range=(0, 255)) \
        .clip(0,255).astype(np.uint8)

    return sobelx2, sobely2, sobel_magnitude

def sobel_operator(gray, ksize=3):
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    out_x = np.clip(sobel_x, 0, 255).astype(np.uint8)
    mean = np.mean(sobel_x)
    out_x[out_x <= mean] = 0

    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    out_y = np.clip(sobel_y, 0, 255).astype(np.uint8)
    mean = np.mean(sobel_y)
    out_y[out_y <= mean] = 0

    return out_x, out_y

def fillhole(input_image):
    '''
    input gray binary image  get the filled image by floodfill method
    Note: only holes surrounded in the connected regions will be filled.
    :param input_image:
    :return:
    '''
    im_flood_fill = input_image.copy()
    h, w = input_image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    im_flood_fill = im_flood_fill.astype("uint8")
    cv2.floodFill(im_flood_fill, mask, (0, 0), 255)
    im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
    img_out = input_image | im_flood_fill_inv
    return img_out

def hough_transform(image, rho=1, theta=np.pi / 180, threshold=30):
    '''
    parameters:
    @rho: Distance resolution of the accumulator in pixels.
    @theta: Angle resolution of the accumulator in radians.
    @threshold: Only lines that are greater than threshold will be returned.
    '''
    return cv2.HoughLines(image, rho=rho, theta=theta, threshold=threshold)

def draw_lines(image, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for rho, theta in line:
            t = theta * 180 / np.pi
            t = np.mod(t, 90)
            k = 40

            if t >= 45 - k and t <= 45 + k:
                continue

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 100000 * (-b))
            y1 = int(y0 + 100000 * (a))
            x2 = int(x0 - 100000 * (-b))
            y2 = int(y0 - 100000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image

def harris_corner_detection(image, T=50):
    dst = cv2.cornerHarris(image, 2, 5, 0.04)
    dst = cv2.dilate(dst, None)

    empty = np.zeros_like(image)
    empty[dst > 0.01 * dst.max()] = 255
    coordinates = np.argwhere(empty)
    coordinates_list = [l.tolist() for l in list(coordinates)]
    coordinates_tuples = [tuple(l) for l in coordinates_list]

    # Compute the distance from each corner to every other corner. 
    def distance(pt1, pt2):
        (x1, y1), (x2, y2) = pt1, pt2
        dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dist

    coordinates_tuples_copy = coordinates_tuples
    i = 1
    for pt1 in coordinates_tuples:
        for pt2 in coordinates_tuples[i::1]:
            if (distance(pt1, pt2) < T):
                coordinates_tuples_copy.remove(pt2)
        i += 1

    for pt in coordinates_tuples:
        cv2.circle(empty, tuple(reversed(pt)), 10, 255, -1)

    return coordinates_tuples, empty

def get_destination_points(corners):
    w1 = np.sqrt((corners[0][0] - corners[1][0]) ** 2 + (corners[0][1] - corners[1][1]) ** 2)
    w2 = np.sqrt((corners[2][0] - corners[3][0]) ** 2 + (corners[2][1] - corners[3][1]) ** 2)
    w = max(int(w1), int(w2))

    h1 = np.sqrt((corners[0][0] - corners[2][0]) ** 2 + (corners[0][1] - corners[2][1]) ** 2)
    h2 = np.sqrt((corners[1][0] - corners[3][0]) ** 2 + (corners[1][1] - corners[3][1]) ** 2)
    h = max(int(h1), int(h2))

    destination_corners = np.float32([(0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0)])

    print('\nThe destination points are: \n')
    for index, c in enumerate(destination_corners):
        character = chr(65 + index) + "'"
        print(character, ':', c)

    print('\nThe approximated height and width of the original image is: \n', (h, w))
    return destination_corners, h, w

def warp_image(img, src):
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


def hough_based_segmentation(image):

    # grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # denoise
    blur = anisotropic_diffusion(gray, kappa=30, gamma=0.25, option=1, niter=10).astype(np.uint8) # parametri paper

    # edge detection
    sx, sy = sobel_operator(blur)

    # hough transform (tuning parametri di hough da fare)
    h_lines = hough_transform(sx, threshold=200)
    v_lines = hough_transform(sy, threshold=200)

    out = draw_lines(gray, h_lines, color=255, thickness=3)
    out = draw_lines(out, v_lines, color=255, thickness=3)
    return out

def cnn_based_segmentation(image, model):
    x = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (224, 224))
    x = img_to_array(x) / 255.0
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x, batch_size=1)
    mask = (pred.squeeze() >= 0.5).astype(np.uint8)
    mask = cv2.resize(mask, image.shape[:2])
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked, mask

def paint_segmentation_pipeline(image, model=None):

    # paint segmentation (hough based or semantic segmentation)
    if model:
        masked, mask = cnn_based_segmentation(image, model)
    else:
        masked, mask = hough_based_segmentation(image)

    cv2.imwrite(os.path.join(OUTPUT_FOLDER, '1_rgb.jpg'), image)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, '2_mask.png'), mask * 255)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, '3_masked.png'), masked)

    # morphology operations
    kernel = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(mask, kernel, iterations = 15)
    erode = cv2.erode(dilate, kernel, iterations = 15)
    fillholes = (fillhole(erode) > 0).astype(np.uint8)

    cv2.imwrite(os.path.join(OUTPUT_FOLDER, '4_dilate.png'), dilate * 255)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, '5_erode.png'), erode * 255)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, '6_fillholes.png'), fillholes * 255)

    # convex hull
    convex_hull = convex_hull_image(mask).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, '7_convex_hull.png'), convex_hull * 255)

    # edge detection
    (T, _) = cv2.threshold(convex_hull, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    canny = cv2.Canny(convex_hull, T * 0.5, T)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, '8_canny.png'), canny)

    # hough transform
    hough_lines = hough_transform(canny, threshold=100)
    hough_out = draw_lines(np.zeros(mask.shape), hough_lines, color=255, thickness=3)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, '9_hough.png'), hough_out * 255)

    # invert hough mask
    invert_hough = 255 - hough_out
    invert_hough = (invert_hough > 0).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, '10_invert_hough.png'), invert_hough * 255)

    # labeling of the connected components [Note: range() starts from 1 since 0 is the background label.]
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(invert_hough, connectivity=4)

    # Create false color image
    colors = np.random.randint(0, 255, size=(nb_components , 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # for cosmetic reason we want the background black
    false_colors = colors[output]

    cv2.imwrite(os.path.join(OUTPUT_FOLDER, '11_labeling_connected_components.png'), false_colors)

    max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])
    max_bw_label = (output == max_label).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, '12_max_bw_label.png'), max_bw_label * 255)

    # harris corner detection
    corners_points, corners_image = harris_corner_detection(max_bw_label)
    src = np.array([[point[1], point[0]] for point in corners_points], dtype="float32")
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, '13_harris.png'), corners_image)

    # image distortion correction
    un_warped = warp_image(masked, src)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, '14_un_warped.jpg'), un_warped)

    return un_warped

