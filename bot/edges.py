import os
import skimage
from skimage.io import imread


def fileparts(fn):
    (dirName, fileName) = os.path.split(fn)
    (fileBaseName, fileExtension) = os.path.splitext(fileName)
    return dirName, fileBaseName, fileExtension


def edges(fn):
    img = skimage.io.imread(fn)
    img = skimage.img_as_ubyte(img)
    img = skimage.color.rgb2gray(img)
    gx = skimage.feature.canny(img)
    gx = skimage.img_as_ubyte(gx)
    skimage.io.imsave(fn, gx)
