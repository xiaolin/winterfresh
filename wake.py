import sys, queue, json, os, subprocess
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

WAKE_WORDS = [
  'winter fresh',
  'when to fresh',
  'when a fresh',
  'when the fresh',
  'winner fresh',
  'winter fest',
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models/vosk-model-small-en-us-0.15")

SR = int(os.getenv("WAKE_SR", "16000"))
BLOCK = int(os.getenv("WAKE_BLOCK", "4000"))

# EMEET hw:2,0 reports CHANNELS: 2 on your Pi; capture 2ch and downmix to mono.
LINUX_CHANNELS = int(os.getenv("WAKE_CHANNELS", "2"))
LINUX_DEVICE = os.getenv("WAKE_ARECORD_DEVICE", "plughw:2,0")

IS_LINUX = sys.platform.startswith("linux")

print("Loading Vosk model...", flush=True)
model = Model(MODEL_PATH)

WAKE_GRAMMAR = json.dumps(WAKE_WORDS)
rec = KaldiRecognizer(model, SR, WAKE_GRAMMAR)

print("Model loaded", flush=True)

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
    pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
  return pcm.tobytes()

def run_linux_arecord():
  print(f"Listening for wake word (Vosk + arecord: {LINUX_DEVICE}, ch={LINUX_CHANNELS}, sr={SR})", flush=True)
  print("-" * 50, flush=True)

  cmd = [
    "arecord",
    "-q",
    "-D", LINUX_DEVICE,
    "-f", "S16_LE",
    "-c", str(LINUX_CHANNELS),
    "-r", str(SR),
    "-t", "raw",
  ]

  proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
  assert proc.stdout is not None

  bytes_per_frame = 2 * LINUX_CHANNELS
  chunk_bytes = BLOCK * bytes_per_frame

  while True:
    raw = proc.stdout.read(chunk_bytes)
    if raw == b"":
      rc = proc.poll()
      err = b""
      try:
        if proc.stderr is not None:
          err = proc.stderr.read() or b""
      except Exception:
        pass
      msg = err.decode("utf-8", errors="replace").strip()
      print(f"AUDIO_ERROR: arecord exited (code={rc}). {msg}", file=sys.stderr, flush=True)
      sys.exit(1)

    mono = downmix_to_mono(raw, LINUX_CHANNELS)
    bar = audio_level_bar(mono)

    if rec.AcceptWaveform(mono):
      result = json.loads(rec.Result())
      text = (result.get("text", "") or "").lower()
      if text:
        print(f"\r{bar} | Heard: {text}                    ", flush=True)
        if any(w in text for w in WAKE_WORDS):
          print("WAKE", flush=True)
          try:
            proc.terminate()
          except Exception:
            pass
          sys.exit(0)
    else:
      partial = json.loads(rec.PartialResult())
      partial_text = partial.get("partial", "") or ""
      print(f"\r{bar} | {partial_text[:30]:30s}", end="", flush=True)

def run_non_linux_sounddevice():
  q = queue.Queue()

  def cb(indata, frames, time, status):
    if status:
      print(f"{status}", file=sys.stderr, flush=True)
    q.put(bytes(indata))

  with sd.RawInputStream(channels=1, samplerate=SR, blocksize=BLOCK, dtype="int16", callback=cb):
    print("Listening for wake word (Vosk + sounddevice)...", flush=True)
    print("-" * 50, flush=True)

    while True:
      data = q.get()
      bar = audio_level_bar(data)

      if rec.AcceptWaveform(data):
        result = json.loads(rec.Result())
        text = (result.get("text", "") or "").lower()
        if text:
          print(f"\r{bar} | Heard: {text}                    ", flush=True)
          if any(w in text for w in WAKE_WORDS):
            print("WAKE", flush=True)
            sys.exit(0)
      else:
        partial = json.loads(rec.PartialResult())
        partial_text = partial.get("partial", "") or ""
        print(f"\r{bar} | {partial_text[:30]:30s}", end="", flush=True)

if IS_LINUX:
  run_linux_arecord()
else:
  run_non_linux_sounddevice()