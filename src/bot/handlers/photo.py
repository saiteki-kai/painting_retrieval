import os
import tempfile
import time

import cv2 as cv
from telegram import InputMediaPhoto, Update
from telegram.ext import CallbackContext

from src.painting.exact_matching import exact_matching
from src.painting.retrieval import retrieve_images
from src.painting.paint_segmentation import paint_segmentation_pipeline
from src.painting.models import get_segmentation_model


def photo_handler(update: Update, ctx: CallbackContext):
    chat_id = update.effective_chat.id

    if len(update.message.photo) > 0:
        file = ctx.bot.getFile(update.message.photo[-1].file_id)
    elif update.message.document:
        file = ctx.bot.getFile(update.message.document.file_id)
    else:
        update.message.reply_text("The file must be an image")

    _, ext = os.path.splitext(file.file_path)

    with tempfile.NamedTemporaryFile(mode="w", suffix=ext) as tmp:
        file.download(tmp.name)
        img = cv.imread(tmp.name)

        if ctx.chat_data["settings"]["segmentation"]:
            update.message.reply_text("Segmentation...")
            segmented_img, _ = paint_segmentation_pipeline(img, model=get_segmentation_model(), folder="" )

            with tempfile.NamedTemporaryFile(mode="wb", suffix=ext) as tmp_segmented:
                cv.imwrite(tmp_segmented.name, segmented_img)
                update.message.reply_photo(open(tmp_segmented.name, "rb"))

            img = segmented_img

        update.message.reply_text("Matching...")
        start_time = time.time()
        result, matched_img_fp = exact_matching(img)
        matching_time = time.time() - start_time
        update.message.reply_text(f"Matching took {matching_time:.3f}s")

        if result is None:
            update.message.reply_text("No matches found")
        else:
            update.message.reply_photo(open(matched_img_fp, "rb"))
            update.message.reply_text(f"Found a match, score: {result}")

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
            print(filepath)
            with open(filepath, "rb") as img:
                images.append(
                    InputMediaPhoto(
                        img, caption=f"Image: {i + 1}, Distance: {dists[i]:.2f}"
                    )
                )

        ctx.bot.sendMediaGroup(chat_id, images)
        ctx.bot.sendMessage(chat_id, caption)
