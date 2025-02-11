from dataclasses import dataclass
import time
import queue
import threading
import numpy as np
from typing import Optional, Dict, List, Set, Callable
import logging
from abc import ABC, abstractmethod
from pathlib import Path


logger = logging.getLogger(__name__)


from transformers import M2M100Config, M2M100ForConditionalGeneration, M2M100Tokenizer


import torch
import signal
def define_torch_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
    

TRANSLATION_MODEL = "facebook/m2m100_1.2B" 
#"facebook/m2m100_418M"

TORCH_DEVICE = define_torch_device()
logger.info(f"Using torch device: {TORCH_DEVICE}")


# @dataclass
# class TranslationResult:
#     original_text: str
#     translation: str
#     target_lang: str
#     segment_start_time: float
#     segment_end_time: float
#     transcribed_time: float
#     translated_time: float 


class TextOutputStreamBase(ABC):
    def __init__(self, language: str):
        self.language = language

    @abstractmethod
    def write(self, translated_text: str, is_complete: bool):
        pass


    def stop(self):
        pass




class ConsoleOutputStream(TextOutputStreamBase):
    def __init__(self, language: str,console_color: int = 93):
        super().__init__(language)
        self.color = console_color


    def write(self, translated_text: str, is_complete: bool):
        if is_complete:
            logger.info(f"\033[{self.color}m[{self.language}]\033[0m: {translated_text}")
        else:
            logger.info(f"\033[{self.color}m[{self.language}]\033[0m: \033[31m{translated_text}\033[0m")

class FileOutputStream(TextOutputStreamBase):
    def __init__(self, file_path: Path | str, language: str):

        super().__init__(language)
        try:
            self.outfile = open(file_path, 'w', encoding='utf-8')
        except FileExistsError as e:
            logger.error(f"Cannot save translated text to {file_path}:\n{e}")
            raise
            

        self.outfile.write(f"---\nlanguage: {self.language}\n---\n\n")
        self.sep=" "

    
    def write(self, translated_text: str, is_complete: bool):
        if is_complete:
            self.outfile.write(translated_text)
            self.outfile.write(self.sep)
            self.outfile.flush()

    def stop(self):
        self.outfile.close()

from queue import Queue
from threading import Lock

class WebOutputStream(TextOutputStreamBase):
    # Class-level storage for all language streams
    _streams: Dict[str, 'WebOutputStream'] = {}
    _lock = Lock()

    def __init__(self, language: str,sep=" "):
        super().__init__(language)
        self.queue = Queue()
        self.buffer = []
        self.sep = sep
        self.incomplete_buffer=""
        
        # Register this stream instance
        with self._lock:
            WebOutputStream._streams[language] = self
            logger.debug(f"Created WebOutputStream for language: {language}")

    def write(self, translated_text: str, is_complete: bool):
        """Write translated text to queue and buffer"""
        if is_complete:
            self.queue.put(translated_text)
            self.buffer.append(translated_text)

        else:   
            self.incomplete_buffer = translated_text
            

        logger.debug(f"WebOutputStream for language {self.language} send new content: {translated_text}")


    def get_new_content(self) -> str:
        """Get new content since last check"""
        content = []
        while not self.queue.empty():
            content.append(self.queue.get())
        return self.sep.join(content)

    def get_full_content(self) -> str:
        """Get all content from buffer"""
        return self.sep.join(self.buffer)
    
    def get_incomplete_content(self) -> str:
        return self.incomplete_buffer

    @classmethod
    def get_stream(cls, language: str) -> Optional['WebOutputStream']:
        """Get stream for specific language"""
        return cls._streams.get(language)

    @classmethod
    def get_available_languages(cls) -> list[str]:
        """Get list of available language streams"""
        return list(cls._streams.keys())

    def stop(self):
        """Clean up resources"""
        with self._lock:
            if self.language in self._streams:
                del self._streams[self.language]
        logger.debug(f"Stopped WebOutputStream for language: {self.language}")

class OnlineTranslator():
    
    def __init__(self, model,src_lang,tgt_lang,
                 output_file: Optional[Path | str] = None,
                 log_to_console: bool = True,
                 write_to_web: bool = False,
                 **inference_ksw):
        
        self.model = model  # Just store the reference to the model
    
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.output_streams = []

        if output_file is not None:
            self.output_streams.append(FileOutputStream(output_file,language=tgt_lang) )
        if log_to_console:
            self.output_streams.append(ConsoleOutputStream(tgt_lang))
        if write_to_web:
            self.output_streams.append(WebOutputStream(tgt_lang))
        
        assert len(self.output_streams) > 0, "No output stream defined"

        self.tokenizer = M2M100Tokenizer.from_pretrained(TRANSLATION_MODEL, src_lang=src_lang,tgt_lang=tgt_lang)

        self.inference_kwargs = inference_ksw
        self.inference_kwargs.setdefault("forced_bos_token_id", self.tokenizer.get_lang_id(self.tgt_lang))



    def tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt").to(TORCH_DEVICE)


    def translate_tokenized_text(self, tokenized_text: torch.Tensor) -> str:
        try:
            
            before_inference = time.time()
            generated_tokens = self.model.generate(**tokenized_text,
                                                    **self.inference_kwargs)
            
            translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

            logger.debug(f"Translating text to {self.tgt_lang} took {time.time()-before_inference:.2f}s")

            return translated_text

            

        except Exception as e:
            logger.error(f"Error translating to {self.tgt_lang}\nError:\n\n{e}")
            return "[ Translation Error ]" 
        
    def translate_to_output(self, tokenized_text: torch.Tensor, is_complete: bool) -> None:
        translation = self.translate_tokenized_text(tokenized_text)

        for output_stream in self.output_streams:
            output_stream.write(translation, is_complete)

        

    def stop(self):
        for output_stream in self.output_streams:
            output_stream.stop()




        


class TranslationPipeline():

    def __init__(self,src_lang,target_languages: List[str],output_folder: Optional[Path | str ] = None,log_to_console: bool = True,log_to_web: bool = False):


        self.should_run = False

        signal.signal(signal.SIGINT, lambda s, f: self.stop())

        # Load model
        
        logger.info(f"Loading model '{TRANSLATION_MODEL}'")
        self.model = M2M100ForConditionalGeneration.from_pretrained(TRANSLATION_MODEL).to(TORCH_DEVICE)

        # Self tokenizer no target-lang
        self.src_lang = src_lang
        self.tokenizer = M2M100Tokenizer.from_pretrained(TRANSLATION_MODEL, src_lang=self.src_lang)


        self.original_output_streams = []
        if log_to_console:
            self.original_output_streams.append(ConsoleOutputStream(src_lang,console_color=36))

        if log_to_web:
            self.original_output_streams.append(WebOutputStream(src_lang))

        # Create Output folder if specified
        if output_folder is not None:
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
 

            self.original_output_streams.append( FileOutputStream(output_folder / f"original_{src_lang}.md",src_lang))




        self.translators = []
        for lang in target_languages:
 

            if output_folder is not None:
                output_file = output_folder / f"translation_{lang}.md"
            else:
                output_file = None

                

            self.translators.append(OnlineTranslator(self.model,
                                                    src_lang=src_lang,
                                                 tgt_lang=lang,
                                                 output_file=output_file,
                                                    log_to_console=log_to_console,
                                                    write_to_web=log_to_web
                                                 ))

        # Set up translation queue
        self.translation_queue = queue.Queue()


        # set up keybord interrupt handling
        # signal.signal(signal.SIGINT, lambda s, f: self.stop())

        logger.debug("Initialized multi-language translation pipeline")

    def __del__(self):
        self.stop()
        


    def _translation_thread(self):
        """
        Get tokenized text from the queue and translate it to all target languages
        Only translate incomplete text if the queue is empty.
        """
        while self.should_run:
            try:
                is_complete,Tokenizers, tokenized_text = self.translation_queue.get(timeout=1)
            except queue.Empty:
                logger.debug("Translation queue is empty")
                time.sleep(0.5)
                continue
            
            queue_size = self.translation_queue.qsize()
            logger.debug(f"Translation queue size: {queue_size}")
            

            if is_complete:
                for T in Tokenizers:
                    T.translate_to_output(tokenized_text,is_complete)
            elif self.translation_queue.not_empty:
                logger.warning("Skipping incomplete translation as queue is not empty")
                continue
            else:
                for T in Tokenizers:
                    T.translate_to_output(tokenized_text,is_complete)




    def start(self):
        # Start one thread for all translators as they are using the same model
        logger.debug("Starting translation queue")
        self.should_run = True
        self.translation_thread = threading.Thread(target=self._translation_thread)
        self.translation_thread.start()


    def stop(self):

        if not self.should_run:
            logger.warning("You already asked to stop the translation pipeline")

        else:

            logger.info("Stopping translation pipeline. Waiting for threads to finish...")

            for output_stream in self.original_output_streams:
                output_stream.stop()
            
            self.should_run = False
            self.translation_thread.join()

            for T in self.translators:
                T.stop()

            

            logger.info("Translation pipeline stopped")
            


    def put_text(self,text:str, incomplete_text: str):

        # Complete text first
        for output_stream in self.original_output_streams:
            output_stream.write(text, is_complete=True)
            output_stream.write(incomplete_text, is_complete=False)
        

        tokenized_text = self.tokenizer(text, return_tensors="pt").to(TORCH_DEVICE)
        
        self.translation_queue.put((True,self.translators,tokenized_text))
        
        # Incoplete text

        tokenized_incomplete_text = self.tokenizer(incomplete_text, return_tensors="pt").to(TORCH_DEVICE)

        self.translation_queue.put((False,self.translators,tokenized_incomplete_text))

        

LanguageName = {
    "af": "Afrikaans",
    "am": "አማርኛ",
    "ar": "العربية",
    "ast": "Asturianu",
    "az": "Azərbaycan",
    "ba": "Башҡортса",
    "be": "Беларуская",
    "bg": "Български",
    "bn": "বাংলা",
    "br": "Brezhoneg",
    "bs": "Bosanski",
    "ca": "Català",
    "ceb": "Cebuano",
    "cs": "Čeština",
    "cy": "Cymraeg",
    "da": "Dansk",
    "de": "Deutsch",
    "el": "Ελληνικά",
    "en": "English",
    "es": "Español",
    "et": "Eesti",
    "fa": "فارسی",
    "ff": "Pulaar",
    "fi": "Suomi",
    "fr": "Français",
    "fy": "Frysk",
    "ga": "Gaeilge",
    "gd": "Gàidhlig",
    "gl": "Galego",
    "gu": "ગુજરાતી",
    "ha": "Hausa",
    "he": "עברית",
    "hi": "हिन्दी",
    "hr": "Hrvatski",
    "ht": "Kreyòl Ayisyen",
    "hu": "Magyar",
    "hy": "Հայերեն",
    "id": "Bahasa Indonesia",
    "ig": "Igbo",
    "ilo": "Ilokano",
    "is": "Íslenska",
    "it": "Italiano",
    "ja": "日本語",
    "jv": "Basa Jawa",
    "ka": "ქართული",
    "kk": "Қазақ тілі",
    "km": "ខ្មែរ",
    "kn": "ಕನ್ನಡ",
    "ko": "한국어",
    "lb": "Lëtzebuergesch",
    "lg": "Luganda",
    "ln": "Lingála",
    "lo": "ລາວ",
    "lt": "Lietuvių",
    "lv": "Latviešu",
    "mg": "Malagasy",
    "mk": "Македонски",
    "ml": "മലയാളം",
    "mn": "Монгол",
    "mr": "मराठी",
    "ms": "Bahasa Melayu",
    "my": "မြန်မာ",
    "ne": "नेपाली",
    "nl": "Nederlands",
    "no": "Norsk",
    "ns": "Sesotho sa Leboa",
    "oc": "Occitan",
    "or": "ଓଡ଼ିଆ",
    "pa": "ਪੰਜਾਬੀ",
    "pl": "Polski",
    "ps": "پښتو",
    "pt": "Português",
    "ro": "Română",
    "ru": "Русский",
    "sd": "سنڌي",
    "si": "සිංහල",
    "sk": "Slovenčina",
    "sl": "Slovenščina",
    "so": "Soomaali",
    "sq": "Shqip",
    "sr": "Српски",
    "ss": "SiSwati",
    "su": "Basa Sunda",
    "sv": "Svenska",
    "sw": "Kiswahili",
    "ta": "தமிழ்",
    "th": "ไทย",
    "tl": "Tagalog",
    "tn": "Setswana",
    "tr": "Türkçe",
    "uk": "Українська",
    "ur": "اردو",
    "uz": "Oʻzbek",
    "vi": "Tiếng Việt",
    "wo": "Wolof",
    "xh": "isiXhosa",
    "yi": "ייִדיש",
    "yo": "Yorùbá",
    "zh": "中文",
    "zu": "isiZulu"
}



if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    logger.info("Testing translation pipeline")

    Test_sentences = [
        "Bonjour à tous.",
        "Aujourd'hui, nous allons parler de la traduction en temps réel.",
        "C'est un sujet très intéressant."
    ]

    pipeline = TranslationPipeline("fr",["en","uk","de"])
    pipeline.start()

    for sentence in Test_sentences:
        pipeline.put_text(sentence)
    pipeline.stop()

    # Send to files

    temp_folder= Path("temp")

    pipeline = TranslationPipeline("fr",["en","uk","de"],output_folder=temp_folder)
    pipeline.start()
    

    for sentence in Test_sentences:
        pipeline.put_text(sentence)
    pipeline.stop()



    for file in temp_folder.glob("*.md"):
        print(f"## Translation to {file.stem}\n")
        print(file.read_text())
        print("\n\n")

    
    # delete temp folder
    for file in temp_folder.glob("*.md"):
        file.unlink()
    temp_folder.rmdir()


    # Test with interruption

    pipeline = TranslationPipeline("fr",["en","uk","de"])
    pipeline.start()
    try:
        pipeline.put_text("Une phrase avant l'interruption.")
        logger.debug("Waiting before interruption")
        time.sleep(2)
        KeyboardInterrupt()
    except KeyboardInterrupt:
        logger.debug("Keyboard interruption")
        pass





    logger.info("End of test")