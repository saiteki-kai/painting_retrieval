import os
import tempfile
import time

from dotenv import load_dotenv
from telegram import Bot
from telegram.ext import Updater, Filters, MessageHandler

from edges import edges


def photo_handler(update, ctx):
    """Photo Handler

    processes the original image and send back the result
    """
    chat_id = update.effective_chat.id

    file = ctx.bot.getFile(update.message.photo[-1].file_id)
    _, ext = os.path.splitext(file.file_path)

    with tempfile.NamedTemporaryFile(mode="w", suffix=ext) as tmp:
        file.download(tmp.name)

        ctx.bot.sendMessage(chat_id, "Hi, please wait until the image is ready")

        start_time = time.perf_counter()
        edges(tmp.name)
        elapsed = time.perf_counter() - start_time

        caption = f"The execution took {elapsed:.3f} seconds"

        with open(tmp.name, 'rb') as photo:
            ctx.bot.sendPhoto(chat_id, photo, caption=caption)


if __name__ == '__main__':
    load_dotenv(".env")
    TOKEN = os.getenv("TOKEN")

    bot = Bot(TOKEN)

    updater = Updater(TOKEN)
    updater.dispatcher.add_handler(MessageHandler(Filters.photo, photo_handler))
    updater.start_polling()
    updater.idle()
