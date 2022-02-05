import os
import tempfile
import time

import cv2 as cv
from telegram import InputMediaPhoto, Update
from telegram.ext import CallbackContext

from src.painting.retrieval import retrieve_images

# TODO: add exact matching


def photo_handler(update: Update, ctx: CallbackContext):
    """Photo Handler

    processes the original image and send back the result
    """
    chat_id = update.effective_chat.id

    file = ctx.bot.getFile(update.message.photo[-1].file_id)
    _, ext = os.path.splitext(file.file_path)

    with tempfile.NamedTemporaryFile(mode="w", suffix=ext) as tmp:
        file.download(tmp.name)

        ctx.bot.sendMessage(chat_id, "Searching...")

        start_time = time.perf_counter()

        img = cv.imread(tmp.name)
        retrieved_img_paths, _, dists = retrieve_images(
            img,
            feature=ctx.chat_data["settings"]["feature"],
            similarity=ctx.chat_data["settings"]["similarity"],
            n_results=ctx.chat_data["settings"]["results"],
        )

        elapsed = time.perf_counter() - start_time

        caption = f"The execution took {elapsed:.3f} seconds"

        images = []
        for i, filepath in enumerate(retrieved_img_paths):
            with open(filepath, "rb") as img:
                images.append(
                    InputMediaPhoto(
                        img, caption=f"Image: {i + 1}, Distance: {dists[i]}"
                    )
                )

        ctx.bot.sendMediaGroup(chat_id, images)
        ctx.bot.sendMessage(chat_id, caption)
