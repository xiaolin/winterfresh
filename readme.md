# Winterfresh 🌨️

A fast, concise voice assistant for Raspberry Pi with wake word detection, conversation memory, and barge-in support.

## Features

- 🎤 Wake word detection ("winterfresh", "winter fresh", "hey winterfresh")
- 💬 Conversational memory (5-minute timeout)
- 🔇 Barge-in support (interrupt while speaking)
- 🎵 Pleasant audio chimes for feedback
- 🔄 Auto-restart on inactivity
- 🎙️ Hardware echo cancellation (with USB speakerphone)

## Hardware Requirements

- Raspberry Pi 5 8GB
- USB Speakerphone with echo cancellation (recommended):
  - eMeet M2 (~$80) - Budget option

## Prerequisites

- Node.js v20+
- OpenAI API key
- Sox audio tools

## Installation

### 1. System Setup

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install sox for audio
sudo apt-get install -y sox libsox-fmt-all

# Install git
sudo apt-get install -y git

# Verify versions
node --version  # Should be v20+
npm --version
```

### 2. Clone & Setup Project

```bash
# Clone repository
git clone <your-repo-url> ~/winterfresh
cd ~/winterfresh

# Install dependencies
npm install

# Build TypeScript
npm run build

# Create environment file
cp .env.example .env
nano .env
```

Add your OpenAI API key to `.env`:

```bash
OPENAI_API_KEY=your_key_here
WINTERFRESH_MAX_TURNS=12
```

### 3. Audio Device Setup

```bash
# Plug in your USB speakerphone (Jabra recommended)

# Check if detected
arecord -l  # List capture devices
aplay -l    # List playback devices

# Should see your device (e.g., "Jabra SPEAK 510")

# Test recording
rec -c 1 -r 24000 test.wav trim 0 3

# Test playback
play test.wav

# If device is not default, set it manually
# Find card number from arecord -l (e.g., card 1)
echo "defaults.pcm.card 1" >> ~/.asoundrc
echo "defaults.ctl.card 1" >> ~/.asoundrc
```

### 4. Install PM2 Process Manager

```bash
# Install PM2 globally
sudo npm install -g pm2

# Start winterfresh
pm2 start dist/loop.js --name winterfresh

# Save PM2 configuration
pm2 save

# Setup auto-start on boot
pm2 startup
# Follow the command it outputs (example):
# sudo env PATH=$PATH:/usr/bin pm2 startup systemd -u pi --hp /home/pi

# Verify it's running
pm2 status
pm2 logs winterfresh --lines 50
```

## Usage

### PM2 Commands

```bash
# View live logs
pm2 logs winterfresh

# View logs with filter
pm2 logs winterfresh --lines 100

# Monitor resource usage
pm2 monit

# Restart after code changes
pm2 restart winterfresh

# Stop the assistant
pm2 stop winterfresh

# Start the assistant
pm2 start winterfresh

# Remove from PM2
pm2 delete winterfresh
```

### Voice Commands

1. **Wake the assistant**: Say "winterfresh", "winter fresh", or "hey winterfresh"
2. **Speak your request**: After hearing the chime
3. **Interrupt anytime**: Start speaking to interrupt long responses
4. **Wait for timeout**: 7 seconds of inactivity returns to wake word mode

### Audio Feedback

- 🎵 **Rising chime** - Wake word detected
- 🎵 **Single tone** - Ready to listen (during conversation)
- 🎵 **Looping chime** - Processing your request
- 🎵 **Falling chime** - Going to sleep

## Development

### Local Testing (Mac/Linux)

```bash
# Install dependencies
npm install

# Build
npm run build

# Run locally
node dist/loop.js
```

### Update Script

Create `update.sh` for easy updates:

```bash
#!/bin/bash
cd ~/winterfresh
git pull
npm install
npm run build
pm2 restart winterfresh
echo "✅ Winterfresh updated!"
```

Make it executable:

```bash
chmod +x update.sh
```

Run updates:

```bash
./update.sh
```

## Troubleshooting

### Audio Device Not Found

```bash
# List available devices
arecord -l
aplay -l

# Test recording manually
rec -c 1 -r 24000 /tmp/test.wav silence 1 0.10 2% 1 1.0 2%

# Test playback
play sounds/wake.wav
```

### Permission Issues

```bash
# Add user to audio group
sudo usermod -a -G audio pi

# Logout and login again for changes to take effect
```

### Check Logs

```bash
# PM2 logs
pm2 logs winterfresh --lines 100

# Check for errors
pm2 logs winterfresh --err

# System logs (if using systemd)
sudo journalctl -u winterfresh -f
```

### API Key Issues

```bash
# Verify API key is set
cat .env | grep OPENAI_API_KEY

# Test API connection
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Clean Temp Files

```bash
# Remove old recordings
rm /tmp/winterfresh-*.wav

# Check disk space
df -h
```

## Configuration

Environment variables in `.env`:

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
WINTERFRESH_MAX_TURNS=20          # Max conversation turns (default: 20, ~5-6 min conversations)
```

**Conversation length guide:**

- `MAX_TURNS=12` → ~3 minutes
- `MAX_TURNS=20` → ~5-6 minutes (recommended)
- `MAX_TURNS=30` → ~7-8 minutes
- `MAX_TURNS=50` → ~12+ minutes

## System Service (Alternative to PM2)

If you prefer systemd over PM2:

Create `/etc/systemd/system/winterfresh.service`:

```ini
[Unit]
Description=Winterfresh Voice Assistant
After=network.target sound.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/winterfresh
Environment="NODE_ENV=production"
EnvironmentFile=/home/pi/winterfresh/.env
ExecStart=/usr/bin/node /home/pi/winterfresh/dist/loop.js
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable winterfresh
sudo systemctl start winterfresh
sudo systemctl status winterfresh

# View logs
sudo journalctl -u winterfresh -f
```

## Maintenance

### Daily Restart (Optional)

Add to crontab:

```bash
crontab -e
```

Add line:

```
0 4 * * * pm2 restart winterfresh
```

### Weekly System Updates

```bash
sudo apt-get update && sudo apt-get upgrade -y
```

## Architecture

```
winterfresh/
├── src/
│   ├── loop.ts          # Main application loop
│   └── tones.ts         # Audio chime functions
├── sounds/              # Generated audio files
│   ├── wake.wav
│   ├── listen.wav
│   ├── processing.wav
│   └── sleep.wav
├── dist/                # Compiled JavaScript
└── .env                 # Environment variables
```

## Performance

- Wake word detection: ~1.5-2 seconds
- Response latency: ~1-3 seconds (depends on OpenAI API)
- Memory usage: ~50-100MB
- CPU usage: ~5-15% on Raspberry Pi 4

## License

MIT

## Credits

Built with:

- OpenAI API (GPT-4o-mini, Whisper)
- Sox (audio processing)
- PM2 (process management)
