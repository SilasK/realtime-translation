from flask import Flask, render_template, jsonify
from src.translation.translation import WebOutputStream
from threading import Thread
from src.translation import server
import signal

CONFIG_FILE = "translation_server_config.yaml"

# from src.utils.logging import add_other_loggers,logger
# import sys

# ## Set up logging
# logger.remove()
# logger.add(sys.stderr, level="DEBUG", format="{time:%H:%M:%S.%f}<level>[{level}]: {message}</level>", colorize=True)

# logger.add("logs/translation_server.log", level="DEBUG", format="{time:%H:%M:%S.%f}[{level}]: {message}", colorize=True)

# add_other_loggers(["src.whisper_streaming.online_asr", "sr.translation.translation", "src.whisper.audio","src.whisper.timestamped_words"], level="DEBUG")




args = server.load_config(CONFIG_FILE)

target_languages = args.target_languages

translation_server_arguments = server.initialize(args,log_to_console=False,log_to_web=True)

def start_translation_pipeline():
    server.main_loop(args,*translation_server_arguments)


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translations/<language>')
def get_translations(language):
    stream = WebOutputStream.get_stream(language)
    if not stream:
        return jsonify({'text': ''})
    return jsonify({'text': stream.get_new_content()})

if __name__ == '__main__':

    signal.signal(signal.SIGINT, server.signal_handler)

    # Start translation pipeline in separate thread
    translation_thread = Thread(target=start_translation_pipeline, daemon=True)
    translation_thread.start()

    app.run(        host='0.0.0.0',  # Allow external connections
        port=5000,
        debug=True       # Set to False in production
        )