# Painting retrieval

The aim of the project is to provide a telegram bot that can recognize a photographed painting and that can recommend
similar ones.

## Preview

![bot example](./out/bot-preview.png)


## Install

```
virtualenv venv
source venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

Create an .env file in the root folder and insert your bot's token

```
TOKEN = 110201543:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw
```

## Configuration

Download `all_data_info.csv`, `train.zip` and `test.zip` 
from this [dataset](https://www.kaggle.com/c/painter-by-numbers/)
and put them into `painting_retrieval/data/raw/dataset/`.

Unzip `model/resnet_model.zip` in the same directory.

Then, from the root, run the following script
```
pip install -e "/PATH_TO_PAINTING_RETREIVAL/painting_retrieval"
```
Now, from `/PATH_TO_PAINTING_RETREIVAL/painting_retrieval/`, run
```
python3 scripts/generate_dataframe.py
python3 scripts/extract_features.py 
python3 scripts/index_features.py
```


## Execution

Run the bot

```
python3 main.py
```

Run any scripts

```
python3 scripts/{script-name}.py
```

## Folders

### data

This folder is intended to contain the [original](https://www.kaggle.com/c/painter-by-numbers/) and processed dataset.

### model

This folder contains all the trained models.

- `resnet_model.zip` : ResNet50 model finetuned.
- `KMeans_BOW.joblib` : KMeans model, used in BOW features
- `Scaler_BOW.joblib` : Scaler, used in BOW features

### scripts

This folder contains the code that must be executed offline to set up the system before execution.

### src

#### painting

This folder contains the code concerning the paintings. It ranges from the calculation of the features to the matching.

#### bot

This folder contains the code concerning the telegram bot.
