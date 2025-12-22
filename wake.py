import sys, queue, json, os, subprocess
import signal
import time
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# Import volume control
import volume
import filler_words

WAKE_WORDS = [
  'winter fresh',
  'hey winter fresh',
]

MAX_WAKE_WORDS = int(os.getenv("MAX_WAKE_WORDS", "4"))
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

# Combined grammar: wake words + volume commands + filler sinks
ALL_PHRASES = WAKE_WORDS + volume.VOLUME_WORDS + filler_words.FILLER_PHRASES
COMBINED_GRAMMAR = json.dumps(ALL_PHRASES)

rec = KaldiRecognizer(model, SR, COMBINED_GRAMMAR)

print(f"✅ Model loaded (wake+volume grammar, {len(ALL_PHRASES)} phrases)", flush=True)

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
    pcm = pcm.reshape(-1, channels)[:, 0].astype(np.int16)  # pick ch0, don't average
  return pcm.tobytes()

def handle_result(result: dict) -> bool:
  """Handle recognition result. Returns True if should exit (wake detected)."""
  text = (result.get("text", "") or "").lower().strip()
  if not text:
    return False

  # Volume command: EXACT match only
  if text in volume.VOLUME_WORDS:
    level = volume.parse_volume_level(text)
    if level is not None:
      print(f"\r🔊 Volume command: {text}                    ", flush=True)
      volume.set_volume(level)
    return False

  # Check for wake word - EXACT match only (not substring)
  if text in WAKE_WORDS:
    print(f"\r✅ WAKE: {text}                    ", flush=True)
    print("WAKE", flush=True)
    return True

  # If we get here, it's not in our grammar (shouldn't happen with grammar constraint)
  return False

def run_linux_arecord():
  print(
    f"🎤 Listening for wake word + volume (device={LINUX_DEVICE}, ch={LINUX_CHANNELS}, sr={SR})",
    flush=True,
  )
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

  proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    bufsize=0,
    start_new_session=True,  # make arecord its own session/process-group
  )
  assert proc.stdout is not None

  bytes_per_frame = 2 * LINUX_CHANNELS
  chunk_bytes = BLOCK * bytes_per_frame

  def _drain_stderr(p: subprocess.Popen) -> str:
    try:
      if p.stderr is None:
        return ""
      data = p.stderr.read() or b""
      return data.decode("utf-8", errors="replace").strip()
    except Exception:
      return ""

  def cleanup():
    # Stop arecord cleanly, then hard-kill if it doesn't exit quickly.
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

      # arecord ended or pipe broke
      if raw == b"":
        rc = proc.poll()
        msg = _drain_stderr(proc)
        print(
          f"AUDIO_ERROR: arecord exited (code={rc}). {msg}",
          file=sys.stderr,
          flush=True,
        )
        sys.exit(1)

      mono = downmix_to_mono(raw, LINUX_CHANNELS)
      bar = audio_level_bar(mono)

      if rec.AcceptWaveform(mono):
        result = json.loads(rec.Result())
        if handle_result(result):
          cleanup()
          sys.exit(0)
      else:
        partial = json.loads(rec.PartialResult())
        partial_text = (partial.get("partial", "") or "")[:30]
        print(f"\r{bar} | {partial_text:30s}", end="", flush=True)
  finally:
    cleanup()

def run_non_linux_sounddevice():
  q = queue.Queue()

  def cb(indata, frames, time, status):
    if status:
      print(f"{status}", file=sys.stderr, flush=True)
    q.put(bytes(indata))

  with sd.RawInputStream(channels=1, samplerate=SR, blocksize=BLOCK, dtype="int16", callback=cb):
    print("🎤 Listening for wake word + volume (sounddevice)...", flush=True)
    print("-" * 50, flush=True)

    while True:
      data = q.get()
      bar = audio_level_bar(data)

      if rec.AcceptWaveform(data):
        result = json.loads(rec.Result())
        
        if handle_result(result):
          sys.exit(0)
      else:
        partial = json.loads(rec.PartialResult())
        partial_text = partial.get("partial", "") or ""
        print(f"\r{bar} | {partial_text[:30]:30s}", end="", flush=True)

if IS_LINUX:
  run_linux_arecord()
else:
  run_non_linux_sounddevice()