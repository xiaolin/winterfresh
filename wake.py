import sys, queue, json, os
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

WAKE_WORDS = [
  'winterfresh',
  'winter fresh',
  'when to fresh',
  'whent to fresh',
  'when a fresh',
  'when the fresh',
  'winner fresh',
  'winter fest',
]

# Get the directory where this script lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models/vosk-model-small-en-us-0.15")

print("Loading Vosk model...", flush=True)
model = Model(MODEL_PATH)

SR = 16000
BLOCK = 4000

# Constrain decoding to wake phrases (big accuracy boost for wake-word use cases)
WAKE_GRAMMAR = json.dumps(WAKE_WORDS)
rec = KaldiRecognizer(model, SR, WAKE_GRAMMAR)

print("✅ Model loaded", flush=True)

q = queue.Queue()

def cb(indata, frames, time, status):
  if status:
    print(f"⚠️  {status}", file=sys.stderr, flush=True)
  q.put(bytes(indata))

def audio_level_bar(data, width=30):
  audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
  level = np.abs(audio).mean()
  normalized = min(1.0, level * 10)
  filled = int(normalized * width)
  bar = '█' * filled + '░' * (width - filled)
  return bar

with sd.RawInputStream(channels=1, samplerate=SR, blocksize=BLOCK, dtype="int16", callback=cb):
  print("🎤 Listening for 'winterfresh'...", flush=True)
  print("-" * 50, flush=True)

  while True:
    data = q.get()
    bar = audio_level_bar(data)

    if rec.AcceptWaveform(data):
      result = json.loads(rec.Result())
      text = result.get("text", "").lower()
      if text:
        print(f"\r{bar} | Heard: {text}                    ", flush=True)
        if any(w in text for w in WAKE_WORDS):
          print("WAKE", flush=True)  # Keep this simple for Node to detect
          sys.exit(0)  # Exit cleanly after wake word
    else:
      partial = json.loads(rec.PartialResult())
      partial_text = partial.get("partial", "")
      print(f"\r{bar} | {partial_text[:30]:30s}", end="", flush=True)