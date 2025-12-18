import os, subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHIME_PATH = os.path.join(SCRIPT_DIR, "sounds", "volume-confirm.wav")

ALSA_CARD = os.getenv("ALSA_CARD", "2")
ALSA_PLAY_DEVICE = os.getenv("ALSA_PLAY_DEVICE", "plughw:2,0")

IS_LINUX = os.sys.platform.startswith("linux")

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

def play_chime(volume_level: int):
  """Play confirmation chime at a volume proportional to the level (1-10)."""
  if not os.path.exists(CHIME_PATH):
    print(f"⚠️  Chime not found: {CHIME_PATH}", flush=True)
    return
  
  amplitude = volume_level / 10.0
  
  try:
    if IS_LINUX:
      subprocess.run(
        f"sox {CHIME_PATH} -t wav - vol {amplitude} | aplay -q -D {ALSA_PLAY_DEVICE}",
        shell=True,
        check=True,
        timeout=2,
      )
    else:
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
    print(f"Volume control not implemented for {os.sys.platform}", flush=True)
    play_chime(level)
    return
  
  percent = level * 10
  
  try:
    subprocess.run(
      ["amixer", "-c", ALSA_CARD, "sset", "PCM", f"{percent}%"],
      check=True,
      capture_output=True,
    )
    print(f"🔊 Volume set to {level}/10 ({percent}%)", flush=True)
    play_chime(level)
  except subprocess.CalledProcessError as e:
    print(f"❌ Failed to set volume: {e.stderr.decode()}", file=os.sys.stderr, flush=True)

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