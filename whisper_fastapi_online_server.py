import io
import argparse
import asyncio
import numpy as np
import ffmpeg
from time import time, sleep
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware


from whisper_streaming_custom.whisper_online import backend_factory, online_factory, add_shared_args,warmup_asr
from timed_objects import ASRToken, TimedList

import math
import logging 
from datetime import timedelta
import traceback

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


# Configure logging for all modules
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Set root logger level
logging.getLogger().setLevel(logging.WARNING)




# Configure main module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)




##### LOAD ARGS #####

parser = argparse.ArgumentParser(description="Whisper FastAPI Online Server")
parser.add_argument(
    "--host",
    type=str,
    default="localhost",
    help="The host address to bind the server to.",
)
parser.add_argument(
    "--port", type=int, default=8000, help="The port number to bind the server to."
)
parser.add_argument(
    "--warmup-file",
    type=str,
    dest="warmup_file",
    help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .",
)

parser.add_argument(
    "--confidence-validation",
    type=bool,
    default=False,
    help="Accelerates validation of tokens using confidence scores. Transcription will be faster but punctuation might be less accurate.",
)

parser.add_argument(
    "--diarization",
    type=bool,
    default=False,
    help="Whether to enable speaker diarization.",
)

parser.add_argument(
    "--transcription",
    type=bool,
    default=True,
    help="To disable to only see live diarization results.",
)

add_shared_args(parser)
args = parser.parse_args()


# Configure specific loggers for different modules
logging.getLogger("whisper_streaming_custom").setLevel(logging.getLevelName(args.log_level))
logging.getLogger("diarization").setLevel(logging.getLevelName(args.log_level))

## Is not this used?
# MIN_CHUNK_SIZE = int(args.vac_chunk_size if args.vac else args.min_chunk_size)


SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLES_PER_SEC = SAMPLE_RATE * int(args.min_chunk_size)
BYTES_PER_SAMPLE = 2  # s16le = 2 bytes per sample
BYTES_PER_SEC = SAMPLES_PER_SEC * BYTES_PER_SAMPLE
MAX_BYTES_PER_SEC = 32000 * 5  # 5 seconds of audio at 32 kHz


class SharedState:
    def __init__(self):
        self.tokens = []
        self.buffer_transcription = ""
        self.buffer_diarization = ""
        self.full_transcription = ""
        self.end_buffer = 0
        self.end_attributed_speaker = 0
        self.lock = asyncio.Lock()
        self.beg_loop = time()
        self.sep = " "  # Default separator
        self.last_response_content = ""  # To track changes in response
        self.recorded_seconds = 0
        
    async def update_transcription(self, new_tokens, buffer, end_buffer, full_transcription, sep):
        async with self.lock:
            self.tokens.extend(new_tokens)
            self.buffer_transcription = buffer
            self.end_buffer = end_buffer
            self.full_transcription = full_transcription
            self.sep = sep
            
    async def update_diarization(self, end_attributed_speaker, buffer_diarization=""):
        async with self.lock:
            self.end_attributed_speaker = end_attributed_speaker
            if buffer_diarization:
                self.buffer_diarization = buffer_diarization
            
    async def add_dummy_token(self):
        async with self.lock:
            current_time = time() - self.beg_loop
            dummy_token = ASRToken(
                start=current_time,
                end=current_time + 1,
                text=".",
                speaker=-1,
                is_dummy=True
            )
            self.tokens.append(dummy_token)
            
    async def get_current_state(self):
        async with self.lock:
            
            if self.beg_loop is not None:
                expected_recording_duration = time() - self.beg_loop
                input_lag = max(expected_recording_duration - self.recorded_seconds, 0)
            else:
                input_lag = 0

                
            # Calculate remaining time for transcription buffer
            transcription_lag = max(0, round(self.recorded_seconds - self.end_buffer + input_lag, 2))
            
            # Calculate remaining time for diarization
            diarization_lag = max(0, round(self.recorded_seconds - self.end_attributed_speaker + input_lag, 2))
                
            return {
                "tokens": self.tokens.copy(),
                "buffer_transcription": self.buffer_transcription,
                "buffer_diarization": self.buffer_diarization,
                "end_buffer": self.end_buffer,
                "end_attributed_speaker": self.end_attributed_speaker,
                "sep": self.sep,
                "transcription_lag": transcription_lag,
                "diarization_lag": diarization_lag
            }
            
    async def reset(self,reset_content=True):
        """Reset the state.
        If reset_content is True, clears the full transcript.
        If reset_content is False, keeps the full transcript.
        """
        async with self.lock:
            self.tokens = []
            self.buffer_transcription = ""
            self.buffer_diarization = ""
            self.end_buffer = 0
            self.last_response_content = ""
            if reset_content:
                self.full_transcription = ""
                self.end_attributed_speaker = 0

    async def start_recording(self):
        await self.reset(reset_content=False)
        async with self.lock:
            self.recorded_seconds = 0
            self.beg_loop = time()
        logger.debug("Start recording")

    async def stop_recording(self):
        async with self.lock:
            self.beg_loop = None

        logger.debug("Stop recording")

    async def update_audio_duration(self,n_seconds):
        async with self.lock:
            self.recorded_seconds += n_seconds








##### LOAD APP #####

@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr, tokenizer, diarization
    if args.transcription:
        asr, tokenizer = backend_factory(args)
        warmup_asr(asr, args.warmup_file)
    else:
        asr, tokenizer = None, None

    if args.diarization:
        from diarization.diarization_online import DiartDiarization
        diarization = DiartDiarization(SAMPLE_RATE)
    else :
        diarization = None
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load demo HTML for the root endpoint
with open("web/live_transcription.html", "r", encoding="utf-8") as f:
    html = f.read()

async def start_ffmpeg_decoder():
    """
    Start an FFmpeg process in async streaming mode that reads WebM from stdin
    and outputs raw s16le PCM on stdout. Returns the process object.
    """
    process = (
        ffmpeg.input("pipe:0", format="webm")
        .output(
            "pipe:1",
            format="s16le",
            acodec="pcm_s16le",
            ac=CHANNELS,
            ar=str(SAMPLE_RATE),
        )
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    )
    return process

async def transcription_processor(shared_state, pcm_queue, transcriber):

    

    full_transcription = ""
    
    # TODO: simplify and use sep from TimedList
    try:
        sep = transcriber.asr.sep
    except AttributeError:
        try:
            sep = transcriber.online.asr.sep
        except AttributeError:
            logger.warning("No separator found for transcription. Using default separator.' '")
            sep = " "


    logger.info("Transcription processor started.")
    
    transcription_is_running = True
    while transcription_is_running:
        try:
            pcm_array = await pcm_queue.get()


            if type(pcm_array) == str and pcm_array == "stop":
                logger.info("Transcription processor received stop signal.")
                new_tokens = transcriber.finish()
                transcription_is_running = False
            else:
                logger.info(f"{len(transcriber.audio_buffer) / transcriber.SAMPLING_RATE} seconds of audio will be processed by the model.")
            
                # Process transcription
                transcriber.insert_audio_chunk(pcm_array)
                new_tokens = transcriber.process_iter()


            if new_tokens:
                full_transcription += new_tokens.get_text(sep=sep)
                
            _buffer = transcriber.get_buffer()
            buffer = _buffer.text
            end_buffer = _buffer.end if _buffer.end else (new_tokens[-1].end if new_tokens else 0)
            
            if buffer in full_transcription:
                buffer = ""
                
            await shared_state.update_transcription(
                new_tokens, buffer, end_buffer, full_transcription, sep)
            
        except Exception as e:
            logger.warning(f"Exception in transcription_processor: {e}")
            logger.warning(f"Traceback: {traceback.format_exc()}")
        finally:
            pcm_queue.task_done()

    logger.info("Transcription processor finished.")


async def diarization_processor(shared_state, pcm_queue, diarization_obj):
    buffer_diarization = ""
    
    diarization_is_running = True
    while diarization_is_running:
        try:
            pcm_array = await pcm_queue.get()


            if type(pcm_array) == str and pcm_array == "stop":
                logger.info("Diarization processor received stop signal.")
                diarization_is_running = False
                
            else:
                # Process diarization
                logger.debug(f"Diarization processor received {len(pcm_array)/SAMPLE_RATE:.2f} seconds of audio")
                await diarization_obj.diarize(pcm_array)
                
            # Get current state
            state = await shared_state.get_current_state()
            tokens = state["tokens"]
            end_attributed_speaker = state["end_attributed_speaker"]
            
            # Update speaker information
            new_end_attributed_speaker = diarization_obj.assign_speakers_to_tokens(
                end_attributed_speaker, tokens)
            
            await shared_state.update_diarization(new_end_attributed_speaker, buffer_diarization)
                
        except Exception as e:
            logger.warning(f"Exception in diarization_processor: {e}")
            logger.warning(f"Traceback: {traceback.format_exc()}")
        finally:
            pcm_queue.task_done()



    logger.info("Diarization processor finished.")

async def results_formatter(shared_state, websocket):
    while True:
        try:

            # Get the current state
            state = await shared_state.get_current_state()
            tokens = state["tokens"]
            buffer_transcription = state["buffer_transcription"]
            buffer_diarization = state["buffer_diarization"]
            end_attributed_speaker = state["end_attributed_speaker"]
            transcription_lag = state["transcription_lag"]
            diarization_lag = state["diarization_lag"]
            sep = state["sep"]


            
            # If diarization is enabled but no transcription, add dummy tokens periodically
            if (not tokens or tokens[-1].is_dummy) and not args.transcription and args.diarization:
                await shared_state.add_dummy_token()
                sleep(0.5)
                state = await shared_state.get_current_state()
                tokens = state["tokens"]
            # Process tokens to create response
            previous_speaker = -1
            lines = []
            last_end_diarized = 0
            undiarized_text = []
            
            for token in tokens:
                speaker = token.speaker
                if args.diarization:
                    if (speaker == -1 or speaker == 0) and token.end >= end_attributed_speaker:
                        undiarized_text.append(token.text)
                        continue
                    elif (speaker == -1 or speaker == 0) and token.end < end_attributed_speaker:
                        speaker = previous_speaker
                    if speaker not in [-1, 0]:
                        last_end_diarized = max(token.end, last_end_diarized)

                if speaker != previous_speaker or not lines:
                    lines.append(
                        {
                            "speaker": speaker,
                            "text": token.text,
                            "beg": format_time(token.start),
                            "end": format_time(token.end),
                            "diff": round(token.end - last_end_diarized, 2)
                        }
                    )
                    previous_speaker = speaker
                elif token.text:  # Only append if text isn't empty
                    lines[-1]["text"] += sep + token.text
                    lines[-1]["end"] = format_time(token.end)
                    lines[-1]["diff"] = round(token.end - last_end_diarized, 2)
            
            if undiarized_text:
                combined_buffer_diarization = sep.join(undiarized_text)
                if buffer_transcription:
                    combined_buffer_diarization += sep
                await shared_state.update_diarization(end_attributed_speaker, combined_buffer_diarization)
                buffer_diarization = combined_buffer_diarization
                
            if lines:
                response = {
                    "lines": lines, 
                    "buffer_transcription": buffer_transcription,
                    "buffer_diarization": buffer_diarization,
                    "transcription_lag": transcription_lag,
                    "diarization_lag": diarization_lag
                }
            else:
                response = {
                    "lines": [{
                        "speaker": 1,
                        "text": "",
                        "beg": format_time(0),
                        "end": format_time(tokens[-1].end) if tokens else format_time(0),
                        "diff": 0
                }],
                    "buffer_transcription": buffer_transcription,
                    "buffer_diarization": buffer_diarization,
                    "transcription_lag": transcription_lag,
                    "diarization_lag": diarization_lag

                }
            
            response_content = ' '.join([str(line['speaker']) + ' ' + line["text"] for line in lines]) + ' | ' + buffer_transcription + ' | ' + buffer_diarization
            
            if response_content != shared_state.last_response_content:
                if lines or buffer_transcription or buffer_diarization:
                    await websocket.send_json(response)
                    shared_state.last_response_content = response_content
            

            # Add a small delay to avoid overwhelming the client
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.warning(f"Exception in results_formatter: {e}")
            logger.warning(f"Traceback: {traceback.format_exc()}")
            await asyncio.sleep(0.5)  # Back off on error



##### ENDPOINTS #####

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection opened.")

    ffmpeg_process = None
    pcm_buffer = bytearray()
    shared_state = SharedState()
    await shared_state.start_recording()
    
    transcription_queue = asyncio.Queue() if args.transcription else None
    diarization_queue = asyncio.Queue() if args.diarization else None
    
    transcriber = None

    async def restart_ffmpeg():
        nonlocal ffmpeg_process, transcriber, pcm_buffer
        if ffmpeg_process:
            try:
                ffmpeg_process.kill()
                await asyncio.get_event_loop().run_in_executor(None, ffmpeg_process.wait)
            except Exception as e:
                logger.warning(f"Error killing FFmpeg process: {e}")
        ffmpeg_process = await start_ffmpeg_decoder()
        pcm_buffer = bytearray()
        
        if args.transcription:
            transcriber = online_factory(args, asr, tokenizer)
        
        await shared_state.reset()
        logger.info("FFmpeg process started.")

    await restart_ffmpeg()

    tasks = []    
    if args.transcription and transcriber:
        tasks.append(asyncio.create_task(
            transcription_processor(shared_state, transcription_queue, transcriber)))    
    else:
        logger.critical("Transcription processor not started.")
    if args.diarization and diarization:
        tasks.append(asyncio.create_task(
            diarization_processor(shared_state, diarization_queue, diarization)))
    formatter_task = asyncio.create_task(results_formatter(shared_state, websocket))
    tasks.append(formatter_task)

    ## Input reader
    async def ffmpeg_stdout_reader():
        nonlocal ffmpeg_process, pcm_buffer
        loop = asyncio.get_event_loop()
        beg = time()
        
        receiving_audio = True
        while receiving_audio:
            try:
                elapsed_time = math.floor((time() - beg) * 10) / 10 # Round to 0.1 sec
                ffmpeg_buffer_from_duration = max(int(32000 * elapsed_time), 4096)
                beg = time()

                # Read chunk with timeout
                try:
                    chunk = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, ffmpeg_process.stdout.read, ffmpeg_buffer_from_duration
                        ),
                        timeout=15.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("FFmpeg read timeout. Restarting...")
                    await restart_ffmpeg()
                    beg = time()
                    continue  # Skip processing and read from new process

                if chunk:
                    pcm_buffer.extend(chunk)


                else:
                    receiving_audio = False
                    logger.error("FFmpeg din't read any data. Exiting.")
                    
                    break;
                
                    
                if len(pcm_buffer) >= BYTES_PER_SEC or not receiving_audio:
                    if len(pcm_buffer) > MAX_BYTES_PER_SEC:
                        logger.warning(
                            f"""Audio buffer is too large: {len(pcm_buffer) / BYTES_PER_SEC:.2f} seconds.
                            I process only 5 seconds of audio at a time.
                            The model probably struggles to keep up. Consider using a smaller model.
                            """)
                        pcm_buffer = pcm_buffer[MAX_BYTES_PER_SEC:]

                    # Convert int16 -> float32
                    pcm_array = (
                        np.frombuffer(pcm_buffer[:MAX_BYTES_PER_SEC], dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )

                    await shared_state.update_audio_duration(len(pcm_buffer) / BYTES_PER_SEC)
                   
                    
                    if args.transcription and transcription_queue:
                        await transcription_queue.put(pcm_array.copy())
                    
                    if args.diarization and diarization_queue:
                        await diarization_queue.put(pcm_array.copy())
                    
                    if not args.transcription and not args.diarization:
                        await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.warning(f"Exception in ffmpeg_stdout_reader: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                break
            finally:
                logger.info("Exiting ffmpeg_stdout_reader...")
                




    stdout_reader_task = asyncio.create_task(ffmpeg_stdout_reader())
    tasks.append(stdout_reader_task)    
    
    ## Main loop
    try:
        main_loop_running = True
        while main_loop_running:
            audio_bytes = await websocket.receive_bytes()
            
            try:
                if len(audio_bytes) == 0:
                    # Empty audio data means stop signal
                    logger.info("Stop signal received.")
                    main_loop_running = False
                    break
                    
                # Normal audio processing
                ffmpeg_process.stdin.write(audio_bytes)
                ffmpeg_process.stdin.flush()
            except (BrokenPipeError, AttributeError) as e:
                logger.warning(f"Error writing to FFmpeg: {e}. Restarting...")
                await restart_ffmpeg()
                ffmpeg_process.stdin.write(audio_bytes)
                ffmpeg_process.stdin.flush()

    except WebSocketDisconnect:
        logger.warning("WebSocket disconnected.")
    finally:
        

        # 1. Close FFmpeg's input
        if ffmpeg_process and ffmpeg_process.stdin:
            ffmpeg_process.stdin.close()


        if not main_loop_running:
            # controlled stop
            await shared_state.stop_recording()

            # Wait for FFmpeg to finish processing remaining audio
            await asyncio.get_event_loop().run_in_executor(None, ffmpeg_process.wait)


            # Send stop signal to processors
            if transcription_queue:
                await transcription_queue.put("stop")
                logger.info("Waiting for transcription to finish... this can take a while")

                for task in tasks:
                    if "transcription_processor" in str(task):
                        await asyncio.gather(task,return_exceptions=True)
                        tasks.remove(task)
            if diarization_queue:
                await diarization_queue.put("stop")
                logger.info("Waiting for diarization to finish... this can take a while")

                for task in tasks:
                    if "diarization_processor" in str(task):
                        await asyncio.gather(task,return_exceptions=True)
                        tasks.remove(task)


            # Send acknowledgment to client that we received stop signal
            # await websocket.send_json({"type": "stop_acknowledged"})
            logger.debug("Finished transcription/diarization. Send ready to stop signal to client.")
            await websocket.send_json({"type": "ready_to_stop"})
            # disconnect  or not client might want to restart
            # await websocket.close()
            
        

        ## Cleanup
        logger.debug("Cancelling tasks...")
        for task in tasks:
            task.cancel()
        
        await asyncio.gather(*tasks, return_exceptions=True) 
        
        # Close diarization if enabled
        if args.diarization and diarization:
            diarization.close()
                

        logger.info("WebSocket endpoint cleaned up.")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "whisper_fastapi_online_server:app", host=args.host, port=args.port, reload=True,
        log_level="info"
    )