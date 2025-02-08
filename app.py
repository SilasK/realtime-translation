from flask import Flask, render_template, jsonify, request
from threading import Thread
import signal

from src.translation.translation import WebOutputStream, LanguageName
from src.translation import server  # assumed to contain translation pipeline code

CONFIG_FILE = "translation_server_config.yaml"


args = server.load_config(CONFIG_FILE)
# subset only to the languages that are supported by the translation pipeline
LanguageName = {l: LanguageName[l] for l in [args.lan] + args.target_languages}

translation_server_arguments = server.initialize(args, log_to_console=False, log_to_web=True)

def run_translation_pipeline():
    server.main_loop(args,*translation_server_arguments)

app = Flask(__name__, static_folder="static", template_folder="templates")

@app.route('/')
def index():
    # Pass the LanguageName dict to template for button generation
    return render_template('index.html', language_names=LanguageName)

@app.route('/translate/<language>')
def translate(language):
    return render_template('translate.html', language=language, language_name=LanguageName.get(language, language))

@app.route('/translations/<language>')
def get_translations(language):
    stream = WebOutputStream.get_stream(language)
    if not stream:
        return jsonify({'text': ''})
    
    # If ?full=true is provided, return the full buffer.
    if request.args.get('full', 'false').lower() == 'true':
        data = stream.get_full_content()
    else:
        data = stream.get_new_content()
    return jsonify({'text': data})

if __name__ == '__main__':
    signal.signal(signal.SIGINT, server.signal_handler)
    
    # Start the translation pipeline on a separate thread.
    translation_thread = Thread(target=run_translation_pipeline, daemon=True)
    translation_thread.start()
    
    app.run(
        host='0.0.0.0',  # For local testing use localhost; change if needed.
        port=5000,
        debug=True
    )