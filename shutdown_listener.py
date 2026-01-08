import sys
import os
import json
import signal
import queue
import subprocess
import time

import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

import filler_words

ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Winter fresh")

SHUTDOWN_PHRASES = [
    f'{ASSISTANT_NAME.lower()} stop',
    f'hey {ASSISTANT_NAME.lower()} stop',
    'stop',
    'be quiet',
    'quiet',
    'enough',
    'thats enough',
    'i got it',
    'got it',
    'never mind',
    'nevermind',
    'go away',
    'go to sleep',
    'goodbye',
    'bye',
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models/vosk-model-small-en-us-0.15")

SR = int(os.getenv("WAKE_SR", "16000"))
BLOCK = int(os.getenv("WAKE_BLOCK", "4000"))

LINUX_CHANNELS = int(os.getenv("WAKE_CHANNELS", "2"))
LINUX_DEVICE = os.getenv("WAKE_ARECORD_DEVICE", "mic_share")

IS_LINUX = sys.platform.startswith("linux")

# Load model
model = Model(MODEL_PATH)

ALL_PHRASES = SHUTDOWN_PHRASES + filler_words.FILLER_PHRASES
COMBINED_GRAMMAR = json.dumps(ALL_PHRASES)

rec = KaldiRecognizer(model, SR, COMBINED_GRAMMAR)

print(f"✅ Shutdown listener ready ({len(ALL_PHRASES)} phrases)", flush=True)


def audio_level_bar(data, width=30):
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    level = np.abs(audio).mean()
    normalized = min(1.0, level * 10)
    filled = int(normalized * width)
    bar = '█' * filled + '░' * (width - filled)
    return bar


def downmix_to_mono(raw_bytes: bytes, channels: int) -> bytes:
    pcm = np.frombuffer(raw_bytes, dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels)[:, 0].astype(np.int16)
    return pcm.tobytes()


def handle_result(result: dict) -> bool:
    text = (result.get("text", "") or "").lower().strip()
    if not text:
        return False

    if text in SHUTDOWN_PHRASES:
        print(f"\r🛑 SHUTDOWN: {text}                    ", flush=True)
        print("SHUTDOWN", flush=True)
        return True

    # Print any other recognized text for debugging
    print(f"\r📝 Heard: {text}                    ", flush=True)
    return False


def run_linux_arecord():
    print(f"👂 Listening for shutdown (device={LINUX_DEVICE}, ch={LINUX_CHANNELS}, sr={SR})", flush=True)
    print("-" * 50, flush=True)

    cmd = [
        "arecord", "-q",
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
                rc = proc.poll()
                print(f"AUDIO_ERROR: arecord exited (code={rc})", file=sys.stderr, flush=True)
                sys.exit(1)

            mono = downmix_to_mono(raw, LINUX_CHANNELS)
            # enable audio level bar for debugging word capture
            # bar = audio_level_bar(mono) # Disabled to reduce output noise

            if rec.AcceptWaveform(mono):
                result = json.loads(rec.Result())
                if handle_result(result):
                    cleanup()
                    sys.exit(0)
            else:
                partial = json.loads(rec.PartialResult())
                partial_text = (partial.get("partial", "") or "")[:30]
                # print(f"\r{bar} | {partial_text:30s}", end="", flush=True)
    finally:
        cleanup()


def run_non_linux_sounddevice():
    q = queue.Queue()

    def cb(indata, frames, time_info, status):
        if status:
            print(f"{status}", file=sys.stderr, flush=True)
        q.put(bytes(indata))

    with sd.RawInputStream(channels=1, samplerate=SR, blocksize=BLOCK, dtype="int16", callback=cb):
        print("👂 Listening for shutdown (sounddevice)...", flush=True)
        print("-" * 50, flush=True)

        while True:
            data = q.get()
            # bar = audio_level_bar(data)

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if handle_result(result):
                    sys.exit(0)
            else:
                partial = json.loads(rec.PartialResult())
                partial_text = (partial.get("partial", "") or "")[:30]
                # print(f"\r{bar} | {partial_text:30s}", end="", flush=True)


if IS_LINUX:
    run_linux_arecord()
else:
    run_non_linux_sounddevice()