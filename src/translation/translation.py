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
    

TRANSLATION_MODEL = "facebook/m2m100_418M"
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
    def write(self, translated_text: str):
        pass


    def stop(self):
        pass




class ConsoleOutputStream(TextOutputStreamBase):
    def __init__(self, language: str,console_color: int = 93):
        super().__init__(language)
        self.color = console_color


    def write(self, translated_text: str):
        print(f"\033[{self.color}m[{self.language}]\033[0m: {translated_text}")

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

    
    def write(self, translated_text: str):
        self.outfile.write(translated_text)
        self.outfile.write(self.sep)
        self.outfile.flush()

    def stop(self):
        self.outfile.close()


class OnlineTranslator():
    
    def __init__(self, model,src_lang,tgt_lang,
                 output_file: Optional[Path | str] = None,
                 **inference_ksw):
        
        self.model = model  # Just store the reference to the model
    
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        if output_file is None:
            self.output_stream = ConsoleOutputStream(tgt_lang)
        else:
            self.output_stream = FileOutputStream(output_file,tgt_lang)


        self.tokenizer = M2M100Tokenizer.from_pretrained(TRANSLATION_MODEL, src_lang=src_lang,tgt_lang=tgt_lang)

        self.inference_kwargs = inference_ksw
        self.inference_kwargs.setdefault("forced_bos_token_id", self.tokenizer.get_lang_id(self.tgt_lang))



    def tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt").to(TORCH_DEVICE)


    def translate_tokenized_text(self, tokenized_text: torch.Tensor) -> str:
        try:
            logger.debug(f"Translating text to {self.tgt_lang}")
            generated_tokens = self.model.generate(**tokenized_text,
                                                    **self.inference_kwargs)
            
            return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

            

        except Exception as e:
            logger.error(f"Error translating to {self.tgt_lang}\nError:\n\n{e}")
            return "[ Translation Error ]" 
        
    def translate_to_output(self, tokenized_text: torch.Tensor) -> None:
        translation = self.translate_tokenized_text(tokenized_text)

        self.output_stream.write(translation)

        

    def stop(self):
        self.output_stream.stop()




# TODO: keybord interupt handling. write stoping but wait until all threads are stopped.
   # TODO: make a central queue for all model executions 
# TODO: pipeline start function


        


class TranslationPipeline():

    def __init__(self,src_lang,target_languages: List[str],output_folder: Optional[Path | str ] = None):

        signal.signal(signal.SIGINT, lambda s, f: self.stop())

        # Load model
        
        logger.info(f"Loading model '{TRANSLATION_MODEL}'")
        self.model = M2M100ForConditionalGeneration.from_pretrained(TRANSLATION_MODEL).to(TORCH_DEVICE)

        # Self tokenizer no target-lang
        self.src_lang = src_lang
        self.tokenizer = M2M100Tokenizer.from_pretrained(TRANSLATION_MODEL, src_lang=self.src_lang)

        # Create Output folder if specified
        if output_folder is not None:
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)

            self.original_output_stream = FileOutputStream(output_folder / f"original_{src_lang}.md",src_lang)
        
        else:
            output_file = None
            self.original_output_stream = ConsoleOutputStream(src_lang,console_color=36)

        self.translators = []
        for lang in target_languages:
 

            if output_folder is not None:
                output_file = output_folder / f"translation_{lang}.md"

                

            self.translators.append(OnlineTranslator(self.model,
                                                    src_lang=src_lang,
                                                 tgt_lang=lang,
                                                 output_file=output_file
                                                 ))

        # Set up translation queue
        self.translation_queue = queue.Queue()


        # set up keybord interrupt handling
        # signal.signal(signal.SIGINT, lambda s, f: self.stop())

        logger.debug("Initialized multi-language translation pipeline")

    def __del__(self):
        self.stop()
        


    def _translation_thread(self):
        while self.should_run:
            try:
                T, tokenized_text = self.translation_queue.get(timeout=1)
            except queue.Empty:
                continue

            T.translate_to_output(tokenized_text)



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

            self.original_output_stream.stop()
            self.should_run = False
            self.translation_thread.join()

            for T in self.translators:
                T.stop()

            

            logger.info("Translation pipeline stopped")
            


    def put_text(self,text:str):

        self.original_output_stream.write(text)

        tokenized_text = self.tokenizer(text, return_tensors="pt").to(TORCH_DEVICE)

        for T in self.translators:
            self.translation_queue.put((T,tokenized_text))

        




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