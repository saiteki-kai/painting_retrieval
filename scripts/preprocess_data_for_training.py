"""
  It takes long time.
  It resize all the data into (224,224,3) so that resnet50 can read them.
  It also puts the resized files into folders to identify the pictorial genre to which they belong.
"""

import pandas as pd
import cv2 as cv
import os

base_dir = './data/'
data_folder = base_dir + 'raw/dataset/'
train_folder = data_folder + 'train/'
test_folder = data_folder + 'test/'


def clear():
    # Clear terminal output
    os.system('cls' if os.name == 'nt' else 'clear')


def dataframe_generator():
    df = pd.read_csv(data_folder + "all_data_info.csv")
    df.rename(columns={"new_filename": "filename"}, inplace=True)
    df.drop(columns=["pixelsx", "pixelsy", "size_bytes", "artist_group", "source"], inplace=True)
    df.drop(columns=["artist", "style", "date", "title"], inplace=True)
    df.dropna(subset=["genre"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # save memory
    df["genre"] = df["genre"].astype("category")

    df.reset_index(drop=True, inplace=True)
    df.to_csv(data_folder + "not_all_data_info.csv")
    return df


# df = dataframe_generator() # To generate it
df = pd.read_csv(data_folder + "not_all_data_info.csv")  # If we have it already

from collections import Counter

c_train = Counter(df["genre"])
c_test = Counter(df["genre"])

list_other_genre = []
for key, value in c_train.items():
    if (value / len(df) * 100) < 1:
        print(f"{key:25s}\t{(value / len(df) * 100):.3f}% \t{(c_test[key] / len(df) * 100):.3f}% ")
        list_other_genre.append(key)

"""
  Many pictorial genres appear less than 1% of the time in the train set.
  We decided to merge them into a single genre called 'other'.

  GENRE                     TRAIN   TEST
  bird-and-flower painting 	0.117% 	0.117% 
  history painting         	0.862% 	0.862% 
  literary painting        	0.547% 	0.547% 
  interior                 	0.657% 	0.657% 
  poster                   	0.280% 	0.280% 
  advertisement            	0.080% 	0.080% 
  panorama                 	0.019% 	0.019% 
  cloudscape               	0.204% 	0.204% 
  quadratura               	0.022% 	0.022% 
  caricature               	0.226% 	0.226% 
  capriccio                	0.231% 	0.231% 
  veduta                   	0.228% 	0.228% 
  battle painting          	0.351% 	0.351% 
  calligraphy              	0.157% 	0.157% 
  vanitas                  	0.031% 	0.031% 
  pastorale                	0.123% 	0.123% 
  wildlife painting        	0.321% 	0.321% 
  miniature                	0.116% 	0.116% 
  yakusha-e                	0.090% 	0.090% 
  tessellation             	0.182% 	0.182% 
  shan shui                	0.032% 	0.032% 
  bijinga                  	0.093% 	0.093% 
  urushi-e                 	0.001% 	0.001% 
"""


def get_image_index(filename):
    for index in range(df.shape[0]):  # n_row
        if (df["filename"][index] == filename):
            return index
    return -1


def get_genre_by_filename(filename):
    index = get_image_index(filename)
    return df["genre"][index]


train_dir = os.path.join(data_folder, "resized_train")
test_dir = os.path.join(data_folder, "resized_test")


def separate_image_in_folders(folder, save_folder_name):
    save_folder = os.path.abspath(os.path.join(folder, os.pardir))
    save_folder = os.path.join(save_folder, save_folder_name)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    N = len(os.listdir(folder))
    n_file = 0

    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        try:
            img = cv.resize(img, (224, 224))
        except Exception as e:
            print(str(e))
            print("Problem with image: " + filename)
            img = None

        if n_file % 10 == 0:
            clear()
            print("{} / {} ".format(n_file, N))
        n_file = n_file + 1

        if img is not None:
            index = get_image_index(filename)
            if index != -1:
                genre = df["genre"][index]
                if genre in list_other_genre:
                    genre = 'other'
                temp_path = os.path.join(save_folder, genre)
                if not os.path.exists(temp_path):
                    os.makedirs(temp_path)
                cv.imwrite(os.path.join(temp_path, filename), img)

    clear()
    print("{} / {} ".format(N, N))
    return True


"""
  It takes long time.
  De-comment before run.
"""
# separate_image_in_folders( train_folder, "refactored_train")
# separate_image_in_folders( test_folder, "refactored_test")


print(f"There are {len(list_other_genre)} pictorial genres that appear less than 1%.")
print(f"Now you just have to consider {42 - len(list_other_genre) + 1} classes!")
