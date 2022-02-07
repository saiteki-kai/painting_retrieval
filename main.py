from src.bot.bot import start_bot
from src.painting.models import load_models

if __name__ == '__main__':
    # preload models
    load_models()

    # start the bot
    start_bot()
