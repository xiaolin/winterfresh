import os, subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHIME_PATH = os.path.join(SCRIPT_DIR, "sounds", "volume-confirm.wav")

ALSA_CARD = os.getenv("ALSA_CARD", "2")
ALSA_PLAY_DEVICE = os.getenv("ALSA_PLAY_DEVICE", "plughw:2,0")
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Winter fresh")

IS_LINUX = os.sys.platform.startswith("linux")

VOLUME_WORDS = [
  f'{ASSISTANT_NAME.lower()} volume zero',
  f'{ASSISTANT_NAME.lower()} volume one',
  f'{ASSISTANT_NAME.lower()} volume two',
  f'{ASSISTANT_NAME.lower()} volume three',
  f'{ASSISTANT_NAME.lower()} volume four',
  f'{ASSISTANT_NAME.lower()} volume five',
  f'{ASSISTANT_NAME.lower()} volume six',
  f'{ASSISTANT_NAME.lower()} volume seven',
  f'{ASSISTANT_NAME.lower()} volume eight',
  f'{ASSISTANT_NAME.lower()} volume nine',
  f'{ASSISTANT_NAME.lower()} volume ten',
]

def play_chime(volume_level: int):
  """Play confirmation chime at a volume proportional to the level (0-10)."""
  if not os.path.exists(CHIME_PATH):
    print(f"⚠️  Chime not found: {CHIME_PATH}", flush=True)
    return
  
  amplitude = volume_level / 10.0
  if (volume_level < 1):
    amplitude = 0.1  # minimal audible for volume 0
  
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
  """Extract volume level (0-10) from recognized text."""
  text = (text or "").lower()

  number_map = {
    'zero': 0,
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
  }

  # Handle digit forms too ("volume 0", "volume 10")
  for n in range(0, 11):
    if f" {n}" in f" {text}":
      return n

  for word, num in number_map.items():
    if word in text:
      return num

  return None