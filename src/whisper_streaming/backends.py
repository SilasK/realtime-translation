import sys
import logging

import io
import soundfile as sf
import math


logger = logging.getLogger(__name__)

class ASRBase:
    sep = " "  # join transcribe words with this character (" " for whisper_timestamped,
    # "" for faster-whisper because it emits the spaces when neeeded)

    def __init__(
        self, lan, modelsize=None, cache_dir=None, model_dir=None, logfile=sys.stderr
    ):
        self.logfile = logfile

        self.transcribe_kargs = {}
        if lan == "auto":
            self.original_language = None
        else:
            self.original_language = lan

        self.model = self.load_model(modelsize, cache_dir, model_dir)

    def load_model(self, modelsize, cache_dir):
        raise NotImplemented("must be implemented in the child class")

    def transcribe(self, audio, init_prompt=""):
        raise NotImplemented("must be implemented in the child class")

    def use_vad(self):
        raise NotImplemented("must be implemented in the child class")


class WhisperTimestampedASR(ASRBase):
    """Uses whisper_timestamped library as the backend. Initially, we tested the code on this backend. It worked, but slower than faster-whisper.
    On the other hand, the installation for GPU could be easier.
    """

    sep = " "

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        import whisper
        import whisper_timestamped
        from whisper_timestamped import transcribe_timestamped

        self.transcribe_timestamped = transcribe_timestamped
        if model_dir is not None:
            logger.debug("ignoring model_dir, not implemented")
        return whisper.load_model(modelsize, download_root=cache_dir)

    def transcribe(self, audio, init_prompt=""):
        result = self.transcribe_timestamped(
            self.model,
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            verbose=None,
            condition_on_previous_text=True,
            **self.transcribe_kargs,
        )
        return result

    def ts_words(self, r):
        # return: transcribe result object to [(beg,end,"word1"), ...]
        o = []
        for s in r["segments"]:
            for w in s["words"]:
                t = (w["start"], w["end"], w["text"])
                o.append(t)
        return o

    def segments_end_ts(self, res):
        return [s["end"] for s in res["segments"]]

    def use_vad(self):
        self.transcribe_kargs["vad"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


class FasterWhisperASR(ASRBase):
    """Uses faster-whisper library as the backend. Works much faster, appx 4-times (in offline mode). For GPU, it requires installation with a specific CUDNN version."""

    sep = ""

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        from faster_whisper import WhisperModel

        #        logging.getLogger("faster_whisper").setLevel(logger.level)
        if model_dir is not None:
            logger.debug(
                f"Loading whisper model from model_dir {model_dir}. modelsize and cache_dir parameters are not used."
            )
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = modelsize
        else:
            raise ValueError("modelsize or model_dir parameter must be set")

        # this worked fast and reliably on NVIDIA L40
        model = WhisperModel(
            model_size_or_path,
            device="cuda",
            compute_type="float16",
            download_root=cache_dir,
        )

        # or run on GPU with INT8
        # tested: the transcripts were different, probably worse than with FP16, and it was slightly (appx 20%) slower
        # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

        # or run on CPU with INT8
        # tested: works, but slow, appx 10-times than cuda FP16
        #        model = WhisperModel(modelsize, device="cpu", compute_type="int8") #, download_root="faster-disk-cache-dir/")
        return model

    def transcribe(self, audio, init_prompt=""):

        # tested: beam_size=5 is faster and better than 1 (on one 200 second document from En ESIC, min chunk 0.01)
        segments, info = self.model.transcribe(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=True,
            **self.transcribe_kargs,
        )
        # print(info)  # info contains language detection result

        return list(segments)

    def ts_words(self, segments):
        o = []
        for segment in segments:
            for word in segment.words:
                if segment.no_speech_prob > 0.9:
                    continue
                # not stripping the spaces -- should not be merged with them!
                w = word.word
                t = (word.start, word.end, w)
                o.append(t)
        return o

    def segments_end_ts(self, res):
        return [s.end for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


class MLXWhisper(ASRBase):
    """
    Uses MPX Whisper library as the backend, optimized for Apple Silicon.
    Models available: https://huggingface.co/collections/mlx-community/whisper-663256f9964fbb1177db93dc
    Significantly faster than faster-whisper (without CUDA) on Apple M1.
    """

    sep = " "

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        """
        Loads the MLX-compatible Whisper model.

        Args:
            modelsize (str, optional): The size or name of the Whisper model to load.
                If provided, it will be translated to an MLX-compatible model path using the `translate_model_name` method.
                Example: "large-v3-turbo" -> "mlx-community/whisper-large-v3-turbo".
            cache_dir (str, optional): Path to the directory for caching models.
                **Note**: This is not supported by MLX Whisper and will be ignored.
            model_dir (str, optional): Direct path to a custom model directory.
                If specified, it overrides the `modelsize` parameter.
        """
        from mlx_whisper.transcribe import ModelHolder, transcribe
        import mlx.core as mx

        if model_dir is not None:
            logger.debug(
                f"Loading whisper model from model_dir {model_dir}. modelsize parameter is not used."
            )
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = self.translate_model_name(modelsize)
            logger.debug(
                f"Loading whisper model {modelsize}. You use mlx whisper, so {model_size_or_path} will be used."
            )

        self.model_size_or_path = model_size_or_path
        
        # In mlx_whisper.transcribe, dtype is defined as:
        # dtype = mx.float16 if decode_options.get("fp16", True) else mx.float32
        # Since we do not use decode_options in self.transcribe, we will set dtype to mx.float16
        dtype = mx.float16 
        ModelHolder.get_model(model_size_or_path, dtype)
        return transcribe

    def translate_model_name(self, model_name):
        """
        Translates a given model name to its corresponding MLX-compatible model path.

        Args:
            model_name (str): The name of the model to translate.

        Returns:
            str: The MLX-compatible model path.
        """
        # Dictionary mapping model names to MLX-compatible paths
        model_mapping = {
            "tiny.en": "mlx-community/whisper-tiny.en-mlx",
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base.en": "mlx-community/whisper-base.en-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "small.en": "mlx-community/whisper-small.en-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "medium.en": "mlx-community/whisper-medium.en-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "large-v1": "mlx-community/whisper-large-v1-mlx",
            "large-v2": "mlx-community/whisper-large-v2-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            "large": "mlx-community/whisper-large-mlx",
        }

        # Retrieve the corresponding MLX model path
        mlx_model_path = model_mapping.get(model_name)

        if mlx_model_path:
            return mlx_model_path
        else:
            raise ValueError(
                f"Model name '{model_name}' is not recognized or not supported."
            )

    def transcribe(self, audio, init_prompt=""):
        if self.transcribe_kargs:
            logger.warning("Transcribe kwargs (vad, task) are not compatible with MLX Whisper and will be ignored.")
        segments = self.model(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            word_timestamps=True,
            condition_on_previous_text=True,
            path_or_hf_repo=self.model_size_or_path,
        )
        return segments.get("segments", [])

    def ts_words(self, segments):
        """
        Extract timestamped words from transcription segments and skips words with high no-speech probability.
        """
        return [
            (word["start"], word["end"], word["word"])
            for segment in segments
            for word in segment.get("words", [])
            if segment.get("no_speech_prob", 0) <= 0.9
        ]

    def segments_end_ts(self, res):
        return [s["end"] for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


class OpenaiApiASR(ASRBase):
    """Uses OpenAI's Whisper API for audio transcription."""

    def __init__(self, lan=None, temperature=0, logfile=sys.stderr):
        self.logfile = logfile

        self.modelname = "whisper-1"
        self.original_language = (
            None if lan == "auto" else lan
        )  # ISO-639-1 language code
        self.response_format = "verbose_json"
        self.temperature = temperature

        self.load_model()

        self.use_vad_opt = False

        # reset the task in set_translate_task
        self.task = "transcribe"

    def load_model(self, *args, **kwargs):
        from openai import OpenAI

        self.client = OpenAI()

        self.transcribed_seconds = (
            0  # for logging how many seconds were processed by API, to know the cost
        )

    def ts_words(self, segments):
        no_speech_segments = []
        if self.use_vad_opt:
            for segment in segments.segments:
                # TODO: threshold can be set from outside
                if segment["no_speech_prob"] > 0.8:
                    no_speech_segments.append(
                        (segment.get("start"), segment.get("end"))
                    )

        o = []
        for word in segments.words:
            start = word.start
            end = word.end
            if any(s[0] <= start <= s[1] for s in no_speech_segments):
                # print("Skipping word", word.get("word"), "because it's in a no-speech segment")
                continue
            o.append((start, end, word.word))
        return o

    def segments_end_ts(self, res):
        return [s.end for s in res.words]

    def transcribe(self, audio_data, prompt=None, *args, **kwargs):
        # Write the audio data to a buffer
        buffer = io.BytesIO()
        buffer.name = "temp.wav"
        sf.write(buffer, audio_data, samplerate=16000, format="WAV", subtype="PCM_16")
        buffer.seek(0)  # Reset buffer's position to the beginning

        self.transcribed_seconds += math.ceil(
            len(audio_data) / 16000
        )  # it rounds up to the whole seconds

        params = {
            "model": self.modelname,
            "file": buffer,
            "response_format": self.response_format,
            "temperature": self.temperature,
            "timestamp_granularities": ["word", "segment"],
        }
        if self.task != "translate" and self.original_language:
            params["language"] = self.original_language
        if prompt:
            params["prompt"] = prompt

        if self.task == "translate":
            proc = self.client.audio.translations
        else:
            proc = self.client.audio.transcriptions

        # Process transcription/translation
        transcript = proc.create(**params)
        logger.debug(
            f"OpenAI API processed accumulated {self.transcribed_seconds} seconds"
        )

        return transcript

    def use_vad(self):
        self.use_vad_opt = True

    def set_translate_task(self):
        self.task = "translate"
