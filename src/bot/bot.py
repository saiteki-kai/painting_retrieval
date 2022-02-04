import os
import tempfile
import time

import cv2 as cv
from dotenv import load_dotenv
from telegram import Bot, InputMediaPhoto
from telegram.ext import Updater, Filters, MessageHandler

from ..painting.dataset import Dataset
from ..painting.retrieval import retrieve_images
from ..painting.utils import DATASET_FOLDER, STANDARD_FEATURES_SIZE


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

        img = cv.imread(tmp.name)
        retrieved_img_paths, retrieve_time, dists = retrieve_images(img, "rgb_hist")

        elapsed = time.perf_counter() - start_time

        caption = f"The execution took {elapsed:.3f} seconds"

        images = []
        for i, f in enumerate(retrieved_img_paths):
            with open(f, "rb") as img:
                images.append(InputMediaPhoto(img, caption=f"Image: {i + 1}, Distance: {dists[i]}"))

        ctx.bot.sendMediaGroup(chat_id, images)
        ctx.bot.sendMessage(chat_id, caption)


def start_bot():
    load_dotenv(".env")
    TOKEN = os.getenv("TOKEN")

    bot = Bot(TOKEN)

    print("Bot started")

    updater = Updater(TOKEN)
    updater.dispatcher.add_handler(MessageHandler(Filters.photo, photo_handler))
    updater.start_polling()
    updater.idle()
