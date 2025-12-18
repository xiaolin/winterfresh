import { spawn } from 'child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const isLinux = process.platform === 'linux';

let processingChimeProcess: ReturnType<typeof spawn> | null = null;
let processingLoopTimer: NodeJS.Timeout | null = null;
let processingLoopRunning = false;

function spawnPlayer(soundPath: string) {
  return isLinux
    ? spawn('aplay', ['-q', soundPath])
    : spawn('play', ['-q', soundPath]);
}

export async function chimeWakeDetected() {
  const soundPath = path.join(__dirname, '../sounds', 'wake.wav');
  await new Promise<void>((resolve) => {
    const p = spawnPlayer(soundPath);
    p.on('close', () => resolve());
    p.on('error', () => resolve());
  });
}

export function chimeProcessingStart() {
  chimeProcessingStop();

  const soundPath = path.join(__dirname, '../sounds', 'processing.wav');
  processingLoopRunning = true;

  const playNext = () => {
    if (!processingLoopRunning) return;

    const p = spawnPlayer(soundPath);
    processingChimeProcess = p;

    p.on('close', () => {
      processingChimeProcess = null;
      if (!processingLoopRunning) return;

      // small gap; tune if you want
      processingLoopTimer = setTimeout(playNext, 150);
    });

    p.on('error', () => {
      processingChimeProcess = null;
      if (!processingLoopRunning) return;
      processingLoopTimer = setTimeout(playNext, 500);
    });
  };

  playNext();
}

export function chimeProcessingStop() {
  processingLoopRunning = false;

  if (processingLoopTimer) {
    clearTimeout(processingLoopTimer);
    processingLoopTimer = null;
  }

  if (processingChimeProcess) {
    // SIGTERM is usually enough; SIGKILL can cut off ALSA weirdly
    processingChimeProcess.kill('SIGTERM');
    processingChimeProcess = null;
  }
}
