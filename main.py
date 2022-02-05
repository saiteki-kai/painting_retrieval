import gc
from keras.backend import clear_session
from src.bot.bot import start_bot

if __name__ == '__main__':
    # clear keras memory
    clear_session()
    gc.collect()

    # start the bot
    start_bot()
