import sys, queue, json, os, subprocess, tempfile, wave
import signal
import time
from collections import deque
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# Import volume control
import volume
import filler_words

ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Winter fresh")

WAKE_WORDS = [
  f'{ASSISTANT_NAME.lower()}',
  f'hey {ASSISTANT_NAME.lower()}',
]

MAX_WAKE_WORDS = int(os.getenv("MAX_WAKE_WORDS", "4"))
MIN_CONFIDENCE = float(os.getenv("MIN_WAKE_CONFIDENCE", "0.5"))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models/vosk-model-small-en-us-0.15")

SR = int(os.getenv("WAKE_SR", "16000"))
BLOCK = int(os.getenv("WAKE_BLOCK", "4000"))

LINUX_CHANNELS = int(os.getenv("WAKE_CHANNELS", "2"))
LINUX_DEVICE = os.getenv("WAKE_ARECORD_DEVICE", "mic_share")

IS_LINUX = sys.platform.startswith("linux")

# --with-command (app.ts): also capture "winterfresh <request>" so it can be answered
# without the "What's up?" round trip. Without it, exit on the bare wake word as before.
WITH_COMMAND = "--with-command" in sys.argv
# After a bare wake word, how long the user has to keep talking ("winterfresh ... what time is it").
FOLLOWUP_SEC = float(os.getenv("WAKE_FOLLOWUP_MS", "600")) / 1000
# Loudness that counts as the user continuing (same scale as app.ts VAD_RMS_THRESHOLD).
FOLLOWUP_RMS = float(os.getenv("WAKE_FOLLOWUP_RMS", "500"))
PREROLL_SEC = 0.25  # audio kept before the wake word
RING_SEC = 25  # longer than Vosk's 20s max utterance, so the wake utterance is always buffered
COMMAND_WAV = os.path.join(tempfile.gettempdir(), "winterfresh-command.wav")

print("Loading Vosk model...", flush=True)
model = Model(MODEL_PATH)

# Combined grammar: wake words + volume commands + filler sinks
ALL_PHRASES = WAKE_WORDS + volume.VOLUME_WORDS + filler_words.FILLER_PHRASES
COMBINED_GRAMMAR = json.dumps(ALL_PHRASES)

rec = KaldiRecognizer(model, SR, COMBINED_GRAMMAR)
rec.SetWords(True)  # word start times locate the wake word in the audio buffer

print(f"✅ Model loaded (wake+volume grammar, {len(ALL_PHRASES)} phrases)", flush=True)

def audio_level_bar(data, width=30):
  audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
  level = np.abs(audio).mean()
  normalized = min(1.0, level * 10)
  filled = int(normalized * width)
  bar = '█' * filled + '░' * (width - filled)
  return bar

def rms(data: bytes) -> float:
  audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
  return float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0

def downmix_to_mono(raw_bytes: bytes, channels: int) -> bytes:
  pcm = np.frombuffer(raw_bytes, dtype=np.int16)
  if channels > 1:
    pcm = pcm.reshape(-1, channels)[:, 0].astype(np.int16)  # pick ch0, don't average
  return pcm.tobytes()

class Listener:
  """Feeds mono audio to Vosk and prints what it heard.

  Lines for app.ts (the last three only with --with-command):
    WAKE            bare wake word (exact match, as before)
    COMMAND <wav>   wake word + request; the wav starts at the wake word. Without a
                    preceding WAKE it was only a prefix match, so app.ts confirms it via Whisper.
    NO_COMMAND      nothing followed the bare wake word
  """

  def __init__(self):
    self.samples = 0  # samples fed so far; Vosk word times use this clock
    self.ring = deque()  # (start_sample, pcm) of recent audio
    self.wake_start = 0.0  # stream time (s) the wake word began
    self.followup_deadline = None  # set while waiting after a bare wake word
    self.followup_heard = False

  def now(self) -> float:
    return self.samples / SR

  def feed(self, mono: bytes) -> bool:
    """Process one chunk. Returns True when done (caller should exit)."""
    self.ring.append((self.samples, mono))
    self.samples += len(mono) // 2
    while self.ring[0][0] < self.samples - RING_SEC * SR:
      self.ring.popleft()

    if rec.AcceptWaveform(mono):
      return self.on_final(json.loads(rec.Result()))

    if self.followup_deadline is not None and not self.followup_heard:
      if rms(mono) > FOLLOWUP_RMS:
        self.followup_heard = True  # Vosk's endpoint marks the end of the request
      elif self.now() >= self.followup_deadline:
        print("\nNO_COMMAND", flush=True)
        return True

    partial = json.loads(rec.PartialResult())
    partial_text = (partial.get("partial", "") or "")[:30]
    print(f"\r{audio_level_bar(mono)} | {partial_text:30s}", end="", flush=True)
    return False

  def on_final(self, result: dict) -> bool:
    text = (result.get("text", "") or "").lower().strip()

    if self.followup_deadline is not None:
      # Words Vosk decoded below FOLLOWUP_RMS (e.g. a quiet TV) don't count as the user
      if text and self.followup_heard:
        return self.emit_command()
      if self.followup_heard or self.now() >= self.followup_deadline:
        print("\nNO_COMMAND", flush=True)
        return True
      return False

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
      if not WITH_COMMAND:
        return True
      self.wake_start = self.first_word_start(result)
      self.followup_deadline = self.now() + FOLLOWUP_SEC
      return False

    # Wake word + request in one breath. Looser than the exact match, so app.ts
    # only acts on it if Whisper also hears the name at the start.
    if WITH_COMMAND and any(text.startswith(w + " ") for w in WAKE_WORDS):
      print(f"\r🗣️  Wake + request: {text}", flush=True)
      self.wake_start = self.first_word_start(result)
      return self.emit_command()

    # If we get here, it's not in our grammar (shouldn't happen with grammar constraint)
    return False

  def first_word_start(self, result: dict) -> float:
    words = result.get("result") or []
    return words[0]["start"] if words else 0.0

  def emit_command(self) -> bool:
    """Write the audio from just before the wake word up to now for app.ts to transcribe."""
    first = self.ring[0][0]
    start = max(first, int((self.wake_start - PREROLL_SEC) * SR))
    pcm = b"".join(chunk for _, chunk in self.ring)[(start - first) * 2:]
    with wave.open(COMMAND_WAV, "wb") as w:
      w.setnchannels(1)
      w.setsampwidth(2)
      w.setframerate(SR)
      w.writeframes(pcm)
    print(f"\r🎙️  Captured {len(pcm) / 2 / SR:.1f}s", flush=True)
    print(f"COMMAND {COMMAND_WAV}", flush=True)
    return True

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

  listener = Listener()
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

      if listener.feed(downmix_to_mono(raw, LINUX_CHANNELS)):
        cleanup()
        sys.exit(0)
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

    listener = Listener()
    while True:
      if listener.feed(q.get()):
        sys.exit(0)

if __name__ == "__main__":
  if IS_LINUX:
    run_linux_arecord()
  else:
    run_non_linux_sounddevice()
