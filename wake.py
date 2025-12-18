import sys, queue, json, os, subprocess
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

WAKE_WORDS = [
  'hey winter fresh',
  'hey when to fresh',
  'hey when a fresh',
  'hey when the fresh',
  'hey winner fresh',
  'hey winter fest',
]

# Max words allowed in final result to trigger wake (prevents long sentences from waking)
MAX_WAKE_WORDS = int(os.getenv("MAX_WAKE_WORDS", "4"))

# Min confidence (0.0-1.0) for wake word detection
MIN_CONFIDENCE = float(os.getenv("MIN_WAKE_CONFIDENCE", "0.5"))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models/vosk-model-small-en-us-0.15")

SR = int(os.getenv("WAKE_SR", "16000"))
BLOCK = int(os.getenv("WAKE_BLOCK", "4000"))

LINUX_CHANNELS = int(os.getenv("WAKE_CHANNELS", "2"))
LINUX_DEVICE = os.getenv("WAKE_ARECORD_DEVICE", "plughw:2,0")

IS_LINUX = sys.platform.startswith("linux")

print("Loading Vosk model...", flush=True)
model = Model(MODEL_PATH)

# Don't use grammar - we want open-vocabulary so we can filter by length/confidence
rec = KaldiRecognizer(model, SR)
rec.SetWords(True)  # Enable word-level confidence

print(f"✅ Model loaded (max_words={MAX_WAKE_WORDS}, min_conf={MIN_CONFIDENCE})", flush=True)

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

def should_wake(result: dict) -> bool:
  """Check if result is a valid wake phrase (not a long sentence or low confidence)."""
  text = (result.get("text", "") or "").lower().strip()
  if not text:
    return False

  # Check if any wake word is in the text
  wake_found = any(w in text for w in WAKE_WORDS)
  if not wake_found:
    return False

  # Reject if sentence is too long (likely part of a conversation)
  word_count = len(text.split())
  if word_count > MAX_WAKE_WORDS:
    print(f"\r⚠️  Rejected (too long: {word_count} words): {text[:50]}", flush=True)
    return False

  # Check word-level confidence if available
  word_results = result.get("result", [])
  if word_results:
    confidences = [w.get("conf", 1.0) for w in word_results]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    
    if avg_conf < MIN_CONFIDENCE:
      print(f"\r⚠️  Rejected (low confidence {avg_conf:.2f}): {text[:50]}", flush=True)
      return False

  return True

def run_linux_arecord():
  print(f"🎤 Listening for wake word (device={LINUX_DEVICE}, ch={LINUX_CHANNELS}, sr={SR})", flush=True)
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
        if should_wake(result):
          print(f"\r{bar} | ✅ WAKE: {text}                    ", flush=True)
          print("WAKE", flush=True)
          try:
            proc.terminate()
          except Exception:
            pass
          sys.exit(0)
        else:
          # Show rejected phrases briefly
          print(f"\r{bar} | Heard: {text[:40]:40s}", flush=True)
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
    print("🎤 Listening for wake word (sounddevice)...", flush=True)
    print("-" * 50, flush=True)

    while True:
      data = q.get()
      bar = audio_level_bar(data)

      if rec.AcceptWaveform(data):
        result = json.loads(rec.Result())
        text = (result.get("text", "") or "").lower()
        
        if text:
          if should_wake(result):
            print(f"\r{bar} | ✅ WAKE: {text}                    ", flush=True)
            print("WAKE", flush=True)
            sys.exit(0)
          else:
            print(f"\r{bar} | Heard: {text[:40]:40s}", flush=True)
      else:
        partial = json.loads(rec.PartialResult())
        partial_text = partial.get("partial", "") or ""
        print(f"\r{bar} | {partial_text[:30]:30s}", end="", flush=True)

if IS_LINUX:
  run_linux_arecord()
else:
  run_non_linux_sounddevice()