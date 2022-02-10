import PIL.Image
import cv2 as cv
from src.painting.dataset import Dataset
from src.config import DATASET_FOLDER

ds = Dataset(DATASET_FOLDER, image_size=None)

corrupted = []
for i, img in enumerate(ds.images()):
    if img is None:
        print("None")
        corrupted.append(ds.get_image_filename(i))
    print(i)

print(len(corrupted), corrupted)

corrupted = ['69008.jpg', '121.jpg', '38324.jpg', '97976.jpg', '84772.jpg', '77094.jpg', '85232.jpg', '80945.jpg',
             '32150.jpg', '1262.jpg', '32577.jpg', '43658.jpg', '65430.jpg', '95897.jpg', '83271.jpg', '84021.jpg',
             '32192.jpg', '50789.jpg', '38922.jpg']
