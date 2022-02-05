from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ParseMode, Update
from telegram.ext import CallbackContext


def set_feature_handler(update: Update, ctx: CallbackContext):
    ctx.chat_data["command"] = "set_feature"

    keyboard = [
        [InlineKeyboardButton("ResNet50", callback_data="resnet50")],
        [InlineKeyboardButton("RGB histogram", callback_data="rgb_hist")],
        [InlineKeyboardButton("HSV histogram", callback_data="hsv_hist")],
        [InlineKeyboardButton("HOG", callback_data="hog")],
    ]

    update.message.reply_text(
        "Choice a feature:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


def set_similarity_handler(update: Update, ctx: CallbackContext):
    ctx.chat_data["command"] = "set_similarity"

    keyboard = [
        [InlineKeyboardButton("Euclidean", callback_data="euclidean")],
        [InlineKeyboardButton("Cosine", callback_data="cosine")],
        [InlineKeyboardButton("Manhattan", callback_data="manhattan")],
        [InlineKeyboardButton("Chebyshev", callback_data="chebyshev")],
    ]

    update.message.reply_text(
        "Choice a similarity distance:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


def set_results_handler(update: Update, ctx: CallbackContext):
    ctx.chat_data["command"] = "set_results"

    keyboard = [
        [
            InlineKeyboardButton("1", callback_data=1),
            InlineKeyboardButton("3", callback_data=3),
            InlineKeyboardButton("5", callback_data=5),
            InlineKeyboardButton("10", callback_data=10),
        ]
    ]

    update.message.reply_text(
        "Choice the number of images to retrieve:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


def query_handler(update: Update, ctx: CallbackContext):
    answer = update.callback_query.data
    command = ctx.chat_data["command"]

    if command == "set_similarity":
        ctx.chat_data["settings"]["similarity"] = answer
        # update.callback_query.answer(f"similarity set to {answer}")
        update.callback_query.edit_message_text(
            f"similarity set to *{answer}*",
            parse_mode=ParseMode.MARKDOWN
        )
    elif command == "set_results":
        ctx.chat_data["settings"]["results"] = int(answer)
        # update.callback_query.answer(f"number of images to retrieve set to {answer}")
        update.callback_query.edit_message_text(
            f"number of images to retrieve set to *{answer}*",
            parse_mode=ParseMode.MARKDOWN,
        )
    elif command == "set_feature":
        ctx.chat_data["settings"]["feature"] = answer
        # update.callback_query.answer(f"feature set to **{answer}**")
        update.callback_query.edit_message_text(
            f"feature set to *{answer}*",
            parse_mode=ParseMode.MARKDOWN
        )


def get_settings_handler(update: Update, ctx: CallbackContext):
    n_results = ctx.chat_data["settings"]["results"]
    feature = ctx.chat_data["settings"]["feature"]
    similarity = ctx.chat_data["settings"]["similarity"]

    update.message.reply_text(
        f"feature: *{feature}*\nsimilarity: *{similarity}*\nresults: *{n_results}*",
        parse_mode=ParseMode.MARKDOWN)
