# Painting retrieval

## Install

```
virtualenv venv
source venv/bin/activate

pip install -r requirements.txt
pip install .
```

Create an .env file in the root folder and insert your bot's token

```
TOKEN = 110201543:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw
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

### bot
This folder contains the code concerning the telegram bot.

### data
This folder contains the data.

### model
This folder contains the ResNet50 model finetuned.

### painting
This folder contains the code concerning the paintings. It ranges from the calculation of the features to the matching.

### scripts
This folder contains the code that must be executed offline to set up the system before execution.