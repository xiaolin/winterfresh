# Winterfresh 🌨️

A fast, concise voice assistant for Raspberry Pi with wake word detection, conversation memory, and barge-in support.

## Features

- 🎤 Local wake word detection ("winterfresh") - no API cost for listening
- 💬 Conversational memory (5-minute timeout)
- 🔇 Barge-in support (interrupt while speaking)
- 🎵 Audio feedback with elegant chimes
- 🔄 Auto-restart on inactivity
- 🎙️ Hardware echo cancellation (with USB speakerphone)

## Hardware Requirements

- Raspberry Pi 5 (8GB recommended) or Raspberry Pi 4 (4GB+)
- Raspberry Pi 5 27W USB-C Power Supply (recommended for Pi 5 with USB peripherals)
- SanDisk 64GB Extreme microSDXC UHS-I Memory Card (fast boot and read/write)
- USB Speakerphone with echo cancellation (recommended):
  - EMEET Conference Speakerphone M0 Plus

## Prerequisites

- Node.js v20+
- Python 3.11+
- OpenAI API key
- Sox audio tools

## Installation

### 1. System Setup (Raspberry Pi)

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Node.js v20
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Python 3.11+ and pip
sudo apt-get install -y python3 python3-pip python3-venv

# Install sox for audio recording/playback
sudo apt-get install -y sox libsox-fmt-all

# Install portaudio (required for sounddevice)
sudo apt-get install -y portaudio19-dev

# Install git
sudo apt-get install -y git

# Verify versions
node --version    # Should be v20+
python3 --version # Should be 3.11+
```

### 2. Clone & Setup Project

```bash
# Clone repository
git clone https://github.com/xiaolin/winterfresh.git ~/winterfresh
cd ~/winterfresh

# Install Node.js dependencies
npm install

# Build TypeScript
npm run build
```

### 3. Python Virtual Environment Setup

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Verify installation
python -c "import numpy, sounddevice, vosk; print('✅ All Python packages installed')"

# Deactivate (optional, the app will use .venv/bin/python directly)
deactivate
```

### 4. Download Vosk Model (for local wake word detection)

```bash
# Create models directory
mkdir -p models
cd models

# Download small English model (~50MB)
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
rm vosk-model-small-en-us-0.15.zip

cd ..
```

### 5. Environment Configuration

```bash
# Create environment file
nano .env
```

Environment variables in `.env`:

| Variable                | Default                  | Description                            |
| ----------------------- | ------------------------ | -------------------------------------- |
| `OPENAI_API_KEY`        | (required)               | Your OpenAI API key                    |
| `CHAT_MODEL`            | `gpt-4o-mini`            | OpenAI chat model                      |
| `TRANSCRIBE_MODEL`      | `gpt-4o-mini-transcribe` | OpenAI transcription model             |
| `TTS_MODEL`             | `gpt-4o-mini-tts`        | OpenAI text-to-speech model            |
| `WINTERFRESH_MAX_TURNS` | `20`                     | Max conversation turns before trimming |
| `SAMPLE_RATE`           | `24000`                  | Audio sample rate for recording        |

Example `.env` file:

```bash
OPENAI_API_KEY=your_openai_api_key_here
CHAT_MODEL=gpt-4o-mini
TRANSCRIBE_MODEL=gpt-4o-mini-transcribe
TTS_MODEL=gpt-4o-mini-tts
WINTERFRESH_MAX_TURNS=20
```

Add your OpenAI API key:

```bash
OPENAI_API_KEY=your_openai_api_key_here
WINTERFRESH_MAX_TURNS=20
```

### 7. Audio Device Setup (Raspberry Pi)

```bash
# Plug in your USB speakerphone

# Check if detected
arecord -l  # List capture devices
aplay -l    # List playback devices

# Test recording (speak into mic, then Ctrl+C)
rec -c 1 -r 16000 /tmp/test.wav trim 0 3

# Test playback
play /tmp/test.wav

# Test Python audio
source .venv/bin/activate
python -c "import sounddevice as sd; print(sd.query_devices())"
deactivate
```

If your USB device is not the default, configure it:

```bash
# Find your device card number from arecord -l (e.g., card 1)
echo "defaults.pcm.card 1" >> ~/.asoundrc
echo "defaults.ctl.card 1" >> ~/.asoundrc
```

### 8. Test the Setup

```bash
# Test wake word detection first
source .venv/bin/activate
python wake.py
# Say "winterfresh" - should print "WAKE" and exit
deactivate

# Test full app
npm run dev
# Or: npx tsx src/app.ts
```

### 9. Install PM2 for Production

```bash
# Install PM2 globally
sudo npm install -g pm2

# Build TypeScript
npm run build

# Start winterfresh using ecosystem config
pm2 start ecosystem.config.cjs

# Or start directly
pm2 start dist/app.js --name winterfresh

# Save PM2 configuration
pm2 save

# Setup auto-start on boot
pm2 startup
# Follow the command it outputs (copy/paste and run it)

# Verify it's running
pm2 status
pm2 logs winterfresh --lines 50
```

## Usage

### Voice Commands

1. **Wake the assistant**: Say "winterfresh", "winter fresh", or "hey winterfresh"
2. **Speak your request**: After hearing the wake chime
3. **Interrupt anytime**: Start speaking to interrupt long responses
4. **Wait for timeout**: 7 seconds of inactivity returns to wake word mode

### Audio Feedback

- 🎵 **Wake chime** - Wake word detected, ready to listen
- 🎵 **Processing chime** - Processing your request, stops when assistant speaks

### PM2 Commands

```bash
# View live logs
pm2 logs winterfresh

# Monitor resource usage
pm2 monit

# Restart after code changes
pm2 restart winterfresh

# Stop the assistant
pm2 stop winterfresh

# Start the assistant
pm2 start winterfresh
```

## Development

### Local Testing (Mac)

**Note:** For Mac testing, use headphones to avoid echo (built-in speakers will be heard by built-in mic).

```bash
# Install system dependencies (Mac)
brew install sox portaudio

# Setup Python venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
deactivate

# Download Vosk model
mkdir -p models && cd models
curl -LO https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
cd ..

# Install Node dependencies
npm install

# Run
npm run dev
```

### Update Script

Create `update.sh` for easy updates:

```bash
#!/bin/bash
cd ~/winterfresh
git pull
npm install
npm run build
source .venv/bin/activate
pip install -r requirements.txt 2>/dev/null || pip install numpy sounddevice vosk
deactivate
pm2 restart winterfresh
echo "✅ Winterfresh updated!"
```

Make it executable:

```bash
chmod +x update.sh
```

## Troubleshooting

### Wake word not detecting

```bash
# Test Python script directly
source .venv/bin/activate
python wake.py
# Speak and watch the audio level bar - it should move
# Say "winterfresh" clearly
deactivate

# If no audio level, check your mic:
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### "No module named numpy" error

```bash
# Make sure packages are in the venv, not global Python
.venv/bin/pip install numpy sounddevice vosk

# Verify
.venv/bin/python -c "import numpy, sounddevice, vosk; print('OK')"
```

### Audio device not found

```bash
# List available devices
arecord -l
aplay -l

# Check Python sees the device
source .venv/bin/activate
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### Permission issues (Raspberry Pi)

```bash
# Add user to audio group
sudo usermod -a -G audio $USER

# Logout and login again
```

### Mac microphone permission

Go to **System Settings → Privacy & Security → Microphone** and enable for:

- Terminal (or iTerm)
- Visual Studio Code (if running from VS Code)

Quit and reopen the app after enabling.

## Performance

- Wake word detection: Local (Vosk) - no API cost
- Speech-to-text: OpenAI Whisper API
- Chat: OpenAI GPT-4o-mini
- Text-to-speech: OpenAI TTS

**Raspberry Pi 5 (8GB):**

- Memory usage: ~150-200MB
- CPU usage: ~5-10% during wake word listening
- Wake word latency: ~0.3-0.5 second
- Response latency: ~1-2 seconds (API dependent)

**Raspberry Pi 4 (4GB):**

- Memory usage: ~150-200MB
- CPU usage: ~10-20% during wake word listening
- Wake word latency: ~0.5-1 second
- Response latency: ~1-3 seconds (API dependent)

### Estimated Monthly Cost by Usage

| Usage Level | Exchanges/month | GPT-5      | GPT-4o-mini |
| ----------- | --------------- | ---------- | ----------- |
| Light       | ~50             | $0.03-0.05 | $0.01-0.02  |
| Moderate    | ~200            | $0.10-0.20 | $0.03-0.05  |
| Heavy       | ~500            | $0.25-0.50 | $0.08-0.15  |

**Additional costs per exchange:**

- Transcription (gpt-4o-transcribe): ~$0.003-0.006 (5-10 sec audio)
- TTS (gpt-4o-mini-tts): ~$0.001-0.003 (50-200 chars response)

**Total realistic cost:** Light usage with GPT-5 = **~$0.05-0.10/month** 🎉

## License

MIT
