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
    def write(self, translated_text: str):
        print(f"\033[93m[{self.language}]\033[0m: {translated_text}")

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
    
    def __init__(self,model,tokenizer,src_lang, tgt_lang, output_file: Optional[ Path | str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        if output_file is None:
            self.output_stream = ConsoleOutputStream(tgt_lang)
        else:
            self.output_stream = FileOutputStream(output_file,tgt_lang)

        self.should_stop = False

        self.translation_queue = queue.Queue()
        self._forced_bos_token_id=self.tokenizer.get_lang_id(self.tgt_lang)

        # Create and start translation thread
        self.translation_thread = threading.Thread(
            target=self._translation_worker,
            daemon=True  # Make thread exit when main program exits
        )
        self.translation_thread.start()
        
        # Set up signal handler for main thread
        try:
            signal.signal(signal.SIGINT, lambda s, f: self.stop())
        except (ImportError, ValueError):
            # Handle cases where signal isn't available
            pass

    def _translation_worker(self):
        while self.should_stop is False:
            try:
                text = self.translation_queue.get()
                if text is None:  # Sentinel value to stop the thread
                    break
                    
                translation = self._translate(text)
                self.output_stream.write(translation)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Translation worker error: {e}")
            finally:
                self.translation_queue.task_done()

    def stop(self):
        """Gracefully stop the translation thread"""
        self.should_stop = True
        self.translation_thread.join()
        self.output_stream.stop()

    def put_text(self,text:str):
        self.translation_queue.put(text)

    def _translate(self,src_text: str) -> str:
        try:
            encoded_text = self.tokenizer(src_text, return_tensors="pt").to(TORCH_DEVICE)
            generated_tokens = self.model.generate(**encoded_text, forced_bos_token_id=self._forced_bos_token_id)
            return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

            

        except Exception as e:
            logger.error(f"Error translating to {self.tgt_lang}\nsource_text: {src_text}\nError:\n\n{e}")
            return "[" + src_text + "]" # Return the original text surrounded by brackets to indicate an error
        


# TODO: keybord interupt handling. write stoping but wait until all threads are stopped.
   # TODO: make a central queue for all model executions 
# TODO: pipline start function


        


class TranslationPipeline():

    def __init__(self,src_lang,target_languages: List[str],output_folder: Optional[Path | str ] = None):

    
        # Load model
        logger.info("Loading model 'facebook/m2m100_418M'")
        self.model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").to(TORCH_DEVICE)

        self.src_lang = src_lang
        self.targets=[]

        if output_folder is not None:
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)

        output_file = None
        for lang in target_languages:
            tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang=self.src_lang, tgt_lang=lang)

            if output_folder is not None:
                output_file = output_folder / f"translation_{lang}.md"

                

            self.targets.append(OnlineTranslator(self.model,tokenizer,self.src_lang,lang,output_file=output_file))

        logger.debug("Initialized multi-language translation pipeline")




    def put_text(self,text:str):
        for target in self.targets:
            target.put_text(text)

    def stop(self):
        """Stop all translation threads"""
        for translator in self.targets:
            translator.stop()

        logger.info("Translation pipeline stopped")

 

if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    logger.info("Testing translation pipeline")

    Test_sentences = [
        "Bonjour à tous.",
        "Aujourd'hui, nous allons parler de la traduction en temps réel.",
        "C'est un sujet très intéressant."
    ]

    pipeline = TranslationPipeline("fr",["en","uk","de"])

    for sentence in Test_sentences:
        pipeline.put_text(sentence)
    pipeline.stop()

    # Send to files

    temp_folder= Path("temp")

    pipeline = TranslationPipeline("fr",["en","uk","de"],output_folder=temp_folder)

    

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
    try:
        pipeline.put_text("Une phrase avant l'interruption.")
        logger.debug("Waiting before interruption")
        time.sleep(2)
        KeyboardInterrupt()
    except KeyboardInterrupt:
        pass





    logger.info("End of test")