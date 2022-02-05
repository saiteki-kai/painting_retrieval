from src.bot.bot import start_bot

if __name__ == '__main__':
    # Clear keras memory
    from keras.backend import clear_session as clear_session_keras
    import gc

    clear_session_keras()
    gc.collect()

    # Start the bot
    start_bot()
