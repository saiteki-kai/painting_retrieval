import os
from pathlib import Path

ROOT_FOLDER = Path(__file__).parent.parent

DATA_FOLDER = os.path.join(ROOT_FOLDER, "data")
DATASET_FOLDER = os.path.join(ROOT_FOLDER, "data", "raw", "dataset")
FEATURES_FOLDER = os.path.join(ROOT_FOLDER, "data", "features")
OUTPUT_FOLDER = os.path.join(ROOT_FOLDER, "out")
MODEL_FOLDER = os.path.join(ROOT_FOLDER, "model")

STANDARD_FEATURES_SIZE = (512, 512)

# For semplicity let ccv the last (not mandatory)
LIST_OF_FEATURES_IMPLEMENTED = [
    "rgb_hist",
    "local_rgb_hist",
    "hsv_hist",
    "local_hsv_hist",
    "lbp",
    "hog",
    "dct",
    "resnet50",
    "edge_hist",
    "ccv",
    "orb",
    "combined"
]

SIMILARITY_DISTANCES = [
    "euclidean",
    "cosine",
    "manhattan",
    "chebyshev"
]

LIST_GENRE = [
    'abstract', 
    'allegorical painting',
    'animal painting',
    'cityscape',
    'design',
    'figurative',
    'flower painting',
    'genre painting',
    'illustration',
    'landscape',
    'marina',
    'mythological painting',
    'nude painting (nu)',
    'other',
    'portrait',
    'religious painting',
    'self-portrait',
    'sketch and study',
    'still life',
    'symbolic painting'
]

LIST_F1_PER_GENRE_ON_TEST = [
    0.6918767507002801, 
    0.0,
    0.005899705014749263,
    0.4549180327868853,
    0.2971285892634207,
    0.0,
    0.34355828220858897,
    0.37193763919821826,
    0.37184873949579833,
    0.6533383628819314,
    0.17842323651452283,
    0.0,
    0.11091854419410746,
    0.015,
    0.6084861960348341,
    0.3143872113676732,
    0.0,
    0.4332603938730853,
    0.23668639053254437,
    0.0
]

