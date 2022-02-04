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
  #Clear terminal output
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

#df = dataframe_generator() # To generate it
df = pd.read_csv(data_folder + "not_all_data_info.csv") # If we have it already

def get_image_index(filename):
  for index in range(df.shape[0]): #n_row
    if( df["filename"][index] == filename ):
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

  if not os.path.exists( save_folder ):
    os.makedirs( save_folder )

  N = len( os.listdir(folder) )
  n_file = 0

  for filename in os.listdir(folder):
    img = cv.imread(os.path.join(folder,filename))
    try:
      img = cv.resize(img, (224, 224))
    except Exception as e:
      print(str(e))
      print("Problem with image: " +filename)
      img = None


    if n_file % 10 == 0:
      clear()
      print("{} / {} " .format(n_file, N))
    n_file = n_file + 1 

    if img is not None:
      index = get_image_index(filename)
      if index != -1:
        genre = df["genre"][index]
        temp_path = os.path.join(save_folder, genre)
        if not os.path.exists( temp_path ):
          os.makedirs( temp_path )
        cv.imwrite( os.path.join(temp_path, filename) , img)
  
  clear()
  print("{} / {} " .format(N, N))
  return True


"""
  It takes long time.
  De-comment before run.
"""
#separate_image_in_folders( train_folder, "refactored_train") 
#separate_image_in_folders( test_folder, "refactored_test") 