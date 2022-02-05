import logging
import os

from dotenv import load_dotenv
from telegram import Bot, Update
from telegram.ext import Filters, MessageHandler, Updater, CallbackContext, CommandHandler, \
    CallbackQueryHandler, PicklePersistence

from src.bot.handlers.photo import photo_handler
from src.bot.handlers.settings import set_feature_handler, set_similarity_handler, \
    set_results_handler, query_handler, get_settings_handler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def start_handler(update: Update, ctx: CallbackContext):
    # initialize chatbot data
    ctx.chat_data["settings"] = {
        "feature": "rgb_hist",
        "similarity": "euclidean",
        "results": 5,
    }

    update.message.reply_text("Welcome")


def error_callback(update: Update, ctx: CallbackContext):
    logging.warning('Update "%s" caused error "%s"', update, ctx.error)


def start_bot():
    load_dotenv(".env")
    TOKEN = os.getenv("TOKEN")

    bot = Bot(TOKEN)
    print("Bot started")

    persistence = PicklePersistence(filename="botdata")

    updater = Updater(TOKEN, persistence=persistence, use_context=True)

    updater.dispatcher.add_handler(MessageHandler(Filters.photo, photo_handler))
    updater.dispatcher.add_handler(CommandHandler("start", start_handler))
    updater.dispatcher.add_handler(CommandHandler("set_feature", set_feature_handler))
    updater.dispatcher.add_handler(CommandHandler("set_results", set_results_handler))
    updater.dispatcher.add_handler(CommandHandler("set_similarity", set_similarity_handler))
    updater.dispatcher.add_handler(CommandHandler("get_settings", get_settings_handler))
    updater.dispatcher.add_handler(CallbackQueryHandler(query_handler))
    updater.dispatcher.add_error_handler(error_callback)

    updater.start_polling()
    updater.idle()
