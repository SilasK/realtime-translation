import socket
import sounddevice as sd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def stream_audio_to_server(host='localhost', port=43008):
    # Audio parameters matching arecord
    samplerate = 16000
    channels = 1
    dtype = np.int16
    
    # Setup socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    
    def audio_callback(indata, frames, time, status):
        if status:
            logger.warning(f"Audio status: {status}")
        # Convert to raw bytes and send
        sock.sendall(indata.tobytes())
    
    try:
        # Start audio stream
        with sd.InputStream(
            samplerate=samplerate,
            channels=channels,
            dtype=dtype,
            callback=audio_callback
        ):
            logger.info("Streaming audio to server... Press Ctrl+C to stop")
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        logger.info("Stopping audio stream")
    finally:
        sock.close()

if __name__ == "__main__":
    stream_audio_to_server()