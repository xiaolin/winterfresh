import { spawn } from 'child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Use aplay on Linux/Raspberry Pi for even faster playback
const isLinux = process.platform === 'linux';

let processingChimeProcess: ReturnType<typeof spawn> | null = null;

async function playSound(filename: string) {
  return new Promise<void>((resolve) => {
    const soundPath = path.join(__dirname, '../sounds', filename);
    const player = isLinux
      ? spawn('aplay', ['-q', soundPath])
      : spawn('play', ['-q', soundPath]);

    player.on('close', () => resolve());
    player.on('error', () => resolve());
  });
}

// Pleasant chime patterns
export async function chimeWakeDetected() {
  await playSound('wake.wav');
}

// Start looping processing chime
export async function chimeProcessingStart() {
  // Kill any existing processing chime
  if (processingChimeProcess) {
    processingChimeProcess.kill('SIGKILL');
    processingChimeProcess = null;
  }

  const soundPath = path.join(__dirname, '../sounds', 'processing.wav');

  // Play processing chime in a loop
  const player = isLinux
    ? spawn('aplay', ['-q', soundPath, '--loop', '999'])
    : spawn('play', ['-q', soundPath, 'repeat', '999']);

  processingChimeProcess = player;

  player.on('error', () => {
    processingChimeProcess = null;
  });
}

// Stop looping processing chime
export async function chimeProcessingStop() {
  if (processingChimeProcess) {
    processingChimeProcess.kill('SIGKILL');
    processingChimeProcess = null;
  }
}
