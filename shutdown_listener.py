import sys
import os
import json
import signal
import time
import subprocess
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

from vosk import Model, KaldiRecognizer

SHUTDOWN_PHRASES = [
    'winter fresh stop',
    'hey winter fresh stop',
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models/vosk-model-small-en-us-0.15")

SR = int(os.getenv("WAKE_SR", "16000"))
BLOCK = int(os.getenv("WAKE_BLOCK", "4000"))

LINUX_CHANNELS = int(os.getenv("WAKE_CHANNELS", "2"))
LINUX_DEVICE = os.getenv("WAKE_ARECORD_DEVICE", "plughw:2,0")

IS_LINUX = sys.platform.startswith("linux")

# Load model (shared with wake.py)
model = Model(MODEL_PATH)

# Grammar with just shutdown phrases + filler words to reduce false positives
import filler_words

ALL_PHRASES = SHUTDOWN_PHRASES + filler_words.FILLER_PHRASES
COMBINED_GRAMMAR = json.dumps(ALL_PHRASES)

rec = KaldiRecognizer(model, SR, COMBINED_GRAMMAR)


def downmix_to_mono(raw_bytes: bytes, channels: int) -> bytes:
    pcm = np.frombuffer(raw_bytes, dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels)[:, 0].astype(np.int16)
    return pcm.tobytes()


def handle_result(result: dict) -> bool:
    """Handle recognition result. Returns True if shutdown detected."""
    text = (result.get("text", "") or "").lower().strip()
    if not text:
        return False

    # Check for shutdown phrase - EXACT match only
    if text in SHUTDOWN_PHRASES:
        print("SHUTDOWN", flush=True)
        return True

    return False


def run_linux_arecord():
    cmd = [
        "arecord",
        "-q",
        "-D", LINUX_DEVICE,
        "-f", "S16_LE",
        "-c", str(LINUX_CHANNELS),
        "-r", str(SR),
        "-t", "raw",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        start_new_session=True,
    )
    assert proc.stdout is not None

    bytes_per_frame = 2 * LINUX_CHANNELS
    chunk_bytes = BLOCK * bytes_per_frame

    def cleanup():
        try:
            proc.terminate()
        except Exception:
            pass
        deadline = time.time() + 0.5
        while time.time() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.05)
        try:
            proc.kill()
        except Exception:
            pass

    def on_signal(signum, frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    try:
        while True:
            raw = proc.stdout.read(chunk_bytes)

            if raw == b"":
                cleanup()
                sys.exit(1)

            mono = downmix_to_mono(raw, LINUX_CHANNELS)

            if rec.AcceptWaveform(mono):
                result = json.loads(rec.Result())
                if handle_result(result):
                    cleanup()
                    sys.exit(0)
    finally:
        cleanup()


def run_non_linux_sounddevice():
    import queue

    q = queue.Queue()

    def cb(indata, frames, time_info, status):
        if status:
            print(f"{status}", file=sys.stderr, flush=True)
        q.put(bytes(indata))

    with sd.RawInputStream(
        channels=1, samplerate=SR, blocksize=BLOCK, dtype="int16", callback=cb
    ):
        while True:
            data = q.get()

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if handle_result(result):
                    sys.exit(0)


if IS_LINUX:
    run_linux_arecord()
else:
    run_non_linux_sounddevice()