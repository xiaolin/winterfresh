import { spawn } from 'child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const isLinux = process.platform === 'linux';

let processingChimeProcess: ReturnType<typeof spawn> | null = null;
let processingLoopTimer: NodeJS.Timeout | null = null;
let processingLoopRunning = false;

function spawnPlayer(soundPath: string, ...effects: string[]) {
  // Use SoX `play` everywhere for consistent audio routing on Linux (Pulse/PipeWire setups).
  // Also inherit stderr so failures aren't silent.
  return spawn('play', ['-q', soundPath, ...effects], {
    stdio: ['ignore', 'ignore', 'inherit'],
  });
}

export async function chimeWakeDetected() {
  const soundPath = path.join(__dirname, '../sounds', 'wake.wav');
  await new Promise<void>((resolve) => {
    const p = spawnPlayer(soundPath);
    p.on('close', () => resolve());
    p.on('error', () => resolve());
  });
}

// Soft "still thinking" beep. Quick replies stay silent: it only starts after
// PROCESSING_BEEP_DELAY_MS, then repeats until the voice starts (chimeProcessingStop).
const PROCESSING_DELAY_MS = Number(
  process.env.PROCESSING_BEEP_DELAY_MS ?? '1500',
);
const PROCESSING_GAP_MS = 900; // silence between beeps (the beep itself is 0.3s)
const PROCESSING_VOLUME = '0.6'; // sox gain on processing.wav; keep it under the voice

export function chimeProcessingStart() {
  if (processingLoopRunning) return; // already waiting/beeping for this reply
  processingLoopRunning = true;

  const soundPath = path.join(__dirname, '../sounds', 'processing.wav');
  const playNext = () => {
    if (!processingLoopRunning) return;
    const p = spawnPlayer(soundPath, 'vol', PROCESSING_VOLUME);
    processingChimeProcess = p;
    const scheduleNext = () => {
      if (processingChimeProcess !== p) return; // stopped, or 'close' after 'error'
      processingChimeProcess = null;
      if (!processingLoopRunning) return;
      processingLoopTimer = setTimeout(playNext, PROCESSING_GAP_MS);
    };
    p.on('close', scheduleNext);
    p.on('error', scheduleNext);
  };

  processingLoopTimer = setTimeout(playNext, PROCESSING_DELAY_MS);
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
