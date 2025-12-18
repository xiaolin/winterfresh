import sys, json, os, subprocess
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

VOLUME_WORDS = [
  'winter fresh volume one',
  'winter fresh volume two',
  'winter fresh volume three',
  'winter fresh volume four',
  'winter fresh volume five',
  'winter fresh volume six',
  'winter fresh volume seven',
  'winter fresh volume eight',
  'winter fresh volume nine',
  'winter fresh volume ten',
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models/vosk-model-small-en-us-0.15")

# Path to confirmation chime
CHIME_PATH = os.path.join(SCRIPT_DIR, "sounds", "volume-confirm.wav")

SR = int(os.getenv("WAKE_SR", "16000"))
BLOCK = int(os.getenv("WAKE_BLOCK", "4000"))

LINUX_CHANNELS = int(os.getenv("WAKE_CHANNELS", "2"))
LINUX_DEVICE = os.getenv("WAKE_ARECORD_DEVICE", "plughw:2,0")

# ALSA card for volume control (usually card 2 for EMEET)
ALSA_CARD = os.getenv("ALSA_CARD", "2")
ALSA_PLAY_DEVICE = os.getenv("ALSA_PLAY_DEVICE", "plughw:2,0")

IS_LINUX = sys.platform.startswith("linux")

print("Loading Vosk model for volume control...", flush=True)
model = Model(MODEL_PATH)

VOLUME_GRAMMAR = json.dumps(VOLUME_WORDS)
rec = KaldiRecognizer(model, SR, VOLUME_GRAMMAR)

print("✅ Volume control ready", flush=True)

def play_chime(volume_level: int):
  """Play confirmation chime at a volume proportional to the level (1-10)."""
  if not os.path.exists(CHIME_PATH):
    print(f"⚠️  Chime not found: {CHIME_PATH}", flush=True)
    return
  
  # Scale volume: level 1 = 10% amplitude, level 10 = 100% amplitude
  amplitude = volume_level / 10.0
  
  try:
    if IS_LINUX:
      # Use sox to scale amplitude, then pipe to aplay
      subprocess.run(
        f"sox {CHIME_PATH} -t wav - vol {amplitude} | aplay -q -D {ALSA_PLAY_DEVICE}",
        shell=True,
        check=True,
        timeout=2,
      )
    else:
      # macOS: use sox to scale and play
      subprocess.run(
        f"sox {CHIME_PATH} -t wav - vol {amplitude} | play -q -",
        shell=True,
        check=True,
        timeout=2,
      )
  except subprocess.TimeoutExpired:
    print("⚠️  Chime playback timed out", flush=True)
  except subprocess.CalledProcessError as e:
    print(f"⚠️  Chime playback failed: {e}", flush=True)

def set_volume(level: int):
  """Set system volume (0-10 scale -> 0-100% in ALSA)."""
  if not IS_LINUX:
    print(f"Volume control not implemented for {sys.platform}", flush=True)
    play_chime(level)
    return
  
  percent = level * 10  # 1->10%, 2->20%, ..., 10->100%
  
  try:
    # Set PCM volume (playback) on the EMEET card
    subprocess.run(
      ["amixer", "-c", ALSA_CARD, "sset", "PCM", f"{percent}%"],
      check=True,
      capture_output=True,
    )
    print(f"🔊 Volume set to {level}/10 ({percent}%)", flush=True)
    
    # Play confirmation chime scaled to the level
    play_chime(level)
    
  except subprocess.CalledProcessError as e:
    print(f"❌ Failed to set volume: {e.stderr.decode()}", file=sys.stderr, flush=True)

def parse_volume_level(text: str) -> int | None:
  """Extract volume level (1-10) from recognized text."""
  text = text.lower()
  
  number_map = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
  }
  
  for word, num in number_map.items():
    if word in text:
      return num
  
  return None

def downmix_to_mono(raw_bytes: bytes, channels: int) -> bytes:
  pcm = np.frombuffer(raw_bytes, dtype=np.int16)
  if channels > 1:
    pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
  return pcm.tobytes()

def run_linux_arecord():
  print(f"🎧 Listening for volume commands (device={LINUX_DEVICE}, ch={LINUX_CHANNELS}, sr={SR})", flush=True)
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

    if rec.AcceptWaveform(mono):
      result = json.loads(rec.Result())
      text = (result.get("text", "") or "").lower()
      if text:
        level = parse_volume_level(text)
        if level is not None:
          print(f"Heard: {text}", flush=True)
          set_volume(level)

def run_non_linux_sounddevice():
  import queue
  q = queue.Queue()

  def cb(indata, frames, time, status):
    if status:
      print(f"{status}", file=sys.stderr, flush=True)
    q.put(bytes(indata))

  with sd.RawInputStream(channels=1, samplerate=SR, blocksize=BLOCK, dtype="int16", callback=cb):
    print("🎧 Listening for volume commands (sounddevice)...", flush=True)
    print("-" * 50, flush=True)

    while True:
      data = q.get()

      if rec.AcceptWaveform(data):
        result = json.loads(rec.Result())
        text = (result.get("text", "") or "").lower()
        if text:
          level = parse_volume_level(text)
          if level is not None:
            print(f"Heard: {text}", flush=True)
            set_volume(level)

if IS_LINUX:
  run_linux_arecord()
else:
  run_non_linux_sounddevice()