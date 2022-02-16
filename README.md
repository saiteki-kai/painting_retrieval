# Painting retrieval
The aim of the project is to provide a telegram bot that can recognize a photographed painting and that can recommend similar ones.

To run the program just run the `main.py` file, after running the files in the `script` folder

## data
This folder is intended to contain the [original](https://www.kaggle.com/c/painter-by-numbers/) and processed dataset.


## model
This folder contains all the trained models.
- `resnet_model.zip` : ResNet50 model finetuned.
- `KMeans_BOW.joblib` : KMeans model, used in BOW features
- `Scaler_BOW.joblib` : Scaler, used in BOW features


## scripts
This folder contains the code that must be executed offline to set up the system before execution.

## src
### painting
  This folder contains the code concerning the paintings. It ranges from the calculation of the features to the matching.
 ### bot
  This folder contains the code concerning the telegram bot.
