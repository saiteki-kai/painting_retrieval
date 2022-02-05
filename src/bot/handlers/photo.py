import os
import tempfile
import time

import cv2 as cv
from telegram import InputMediaPhoto, Update
from telegram.ext import CallbackContext

# from src.painting.exact_matching import exact_matching
from src.painting.retrieval import retrieve_images


def photo_handler(update: Update, ctx: CallbackContext):
    chat_id = update.effective_chat.id

    file = ctx.bot.getFile(update.message.photo[-1].file_id)
    _, ext = os.path.splitext(file.file_path)

    with tempfile.NamedTemporaryFile(mode="w", suffix=ext) as tmp:
        file.download(tmp.name)
        img = cv.imread(tmp.name)

        update.message.reply_text("Matching...")

        result = None # exact_matching(img)

        if result is None:
            update.message.reply_text("No matches found")
        else:
            _, score = result
            update.message.reply_text(f"Found a match {score}")

        update.message.reply_text("Searching...")

        start_time = time.perf_counter()

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
