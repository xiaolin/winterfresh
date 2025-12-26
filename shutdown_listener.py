import sys
import os
import json
import signal
import queue

import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

SHUTDOWN_PHRASES = [
    'winter fresh stop',
    'hey winter fresh stop',
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models/vosk-model-small-en-us-0.15")

SR = int(os.getenv("WAKE_SR", "16000"))
BLOCK = int(os.getenv("WAKE_BLOCK", "4000"))

# Load model
model = Model(MODEL_PATH)

import filler_words

ALL_PHRASES = SHUTDOWN_PHRASES + filler_words.FILLER_PHRASES
COMBINED_GRAMMAR = json.dumps(ALL_PHRASES)

rec = KaldiRecognizer(model, SR, COMBINED_GRAMMAR)


def handle_result(result: dict) -> bool:
    text = (result.get("text", "") or "").lower().strip()
    if not text:
        return False

    if text in SHUTDOWN_PHRASES:
        print("SHUTDOWN", flush=True)
        return True

    return False


def on_signal(signum, frame):
    sys.exit(0)


signal.signal(signal.SIGTERM, on_signal)
signal.signal(signal.SIGINT, on_signal)

q = queue.Queue()


def cb(indata, frames, time_info, status):
    if status:
        print(f"{status}", file=sys.stderr, flush=True)
    q.put(bytes(indata))


# Use default device - PulseAudio handles sharing
with sd.RawInputStream(
    channels=1, samplerate=SR, blocksize=BLOCK, dtype="int16", callback=cb
):
    while True:
        data = q.get()

        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if handle_result(result):
                sys.exit(0)