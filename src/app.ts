import 'dotenv/config';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { readFileSync, rmSync } from 'node:fs';
import path from 'node:path';
import OpenAI, { toFile } from 'openai';
import Groq from 'groq-sdk';
import { CartesiaClient } from '@cartesia/cartesia-js';

import {
  chimeWakeDetected,
  chimeProcessingStart,
  chimeProcessingStop,
} from './tones.js';

import {
  runPreflightChecks,
  cleanupZombieProcesses,
  recordSuccess,
  recordError,
  resetErrorCounter,
} from './cleanup.js';

import {
  getCachedAudio,
  cacheAudio,
  CACHED_PHRASES,
  ERROR_PHRASE,
} from './tts-cache.js';

const IS_LINUX = process.platform === 'linux';
// SDK defaults (OpenAI: 10min x 3 attempts) make a hung request look like a crash.
const API_TIMEOUT_MS = Number(process.env.API_TIMEOUT_MS ?? '15000');
const API_MAX_RETRIES = 1;
// Healthy TTS sends its first byte in <1s, so a long wait means a stalled request;
// give up fast and let the retry go out. (Covers time-to-headers only, not playback.)
const TTS_TIMEOUT_MS = Number(process.env.TTS_TIMEOUT_MS ?? '5000');
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  timeout: API_TIMEOUT_MS,
  maxRetries: API_MAX_RETRIES,
});
const ASSISTANT_NAME = process.env.ASSISTANT_NAME ?? 'Winter fresh';
const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY,
  timeout: API_TIMEOUT_MS,
  maxRetries: API_MAX_RETRIES,
});
const cartesia = new CartesiaClient({ apiKey: process.env.CARTESIA_API_KEY });
// faster TTS but has accent when speaking other languages
const USE_CARTESIA_TTS = process.env.USE_CARTESIA_TTS === 'true';
const CARTESIA_VOICE_ID =
  process.env.CARTESIA_VOICE_ID ?? '829ccd10-f8b3-43cd-b8a0-4aeaa81f3b30';
const SAMPLE_RATE = process.env.SAMPLE_RATE ?? '16000';
const LINUX_ARECORD_DEVICE = process.env.ARECORD_DEVICE ?? 'mic_share';
const LINUX_ARECORD_RATE = process.env.ARECORD_RATE ?? '16000';
const LINUX_ARECORD_CHANNELS = process.env.ARECORD_CHANNELS ?? '2';
const CHAT_MODEL = process.env.CHAT_MODEL ?? 'gpt-5.4-mini';
const USE_GROQ_CHAT = process.env.USE_GROQ_CHAT === 'true';
const GROQ_CHAT_MODEL = process.env.GROQ_CHAT_MODEL ?? 'groq/compound-mini';
const TRANSCRIBE_MODEL =
  process.env.TRANSCRIBE_MODEL ?? 'gpt-4o-mini-transcribe';
const TTS_MODEL = process.env.TTS_MODEL ?? 'tts-1';
const TTS_VOICE_ID = process.env.TTS_VOICE_ID ?? 'alloy';
const DEFAULT_RULES = [
  'This is a voice conversation.',
  'User input is an automatic speech-to-text transcript and may contain errors.',
  'Reply in plain text',
  'Do not use markdown.',
  'Do not use bulleted or numbered lists.',
  'Prefer one sentence when possible.',
].join('\n');
const ASSISTANT_RULES = process.env.ASSISTANT_RULES ?? DEFAULT_RULES;

type Msg = { role: 'system' | 'user' | 'assistant'; content: string };
function parsePositiveInteger(
  value: string | undefined,
  fallback: number,
): number {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

const system: Msg = {
  role: 'system',
  content: [
    `You are ${ASSISTANT_NAME}, a helpful assistant that prioritizes answering in one sentence.`,
    ASSISTANT_RULES,
  ].join('\n'),
};

const MAX_TURNS = parsePositiveInteger(process.env.WINTERFRESH_MAX_TURNS, 20);
const IDLE_TIMEOUT_MS = 7000; // 7 seconds
const HISTORY_TIMEOUT_MS = 5 * 60 * 1000; // 5 minutes

let currentTtsProcess: ReturnType<typeof spawn> | null = null;
let currentRecProcess: ReturnType<typeof spawn> | null = null;
let shutdownListenerProcess: ReturnType<typeof spawn> | null = null;
let isShutdownListenerStarting = false;
let wakeListenerProcess: ReturnType<typeof spawn> | null = null;
let isWakeListenerStarting = false;
let isAppRunning = false;
// Bumped by every start(). Loops and in-flight turns capture it so a slow API
// call from an old session can't resume after a restart flips isAppRunning back on.
let runId = 0;
let isGoingToSleep = false;
// Bumped by every killCurrentTTS() so a TTS fetch that was cancelled mid-request
// doesn't start playing once its audio arrives.
let ttsGen = 0;
let restartTimeout: NodeJS.Timeout | null = null;
let historyTimeout: NodeJS.Timeout | null = null;

// Persistent conversation history
let conversationHistory: Msg[] = [system];
let lastInteractionTime = 0;

// Track current operations
type Operation = 'TTSSpeaking' | 'ActiveAsking';
const currentOperations = new Set<Operation>();

// Goal: make quiet speech reliably transcribable and avoid early cutoff.
// Volume level (in %) below which audio is considered "silence"
// Higher = more sensitive (quieter sounds count as silence)
// '2.0' means audio below 2% of max volume is silence
const SILENCE_THRESHOLD = '2.0';
// How long silence must persist to stop recording
// '1.0' means recording stops after 1.0 seconds of continuous silence
const SILENCE_DURATION_SEC = '1.0'; // leave at 1 otherwise too eager to cut off
const INPUT_VOLUME = 2; // Linux only (linear factor)
const MAC_GAIN_DB = 6; // ~20*log10(2) = +6.02 dB

// shutdown listener functions
function startShutdownListener(): void {
  // Prevent duplicate listeners
  if (shutdownListenerProcess || isShutdownListenerStarting) {
    console.log('🎧 Shutdown listener already running or starting, skipping');
    return;
  }
  isShutdownListenerStarting = true;
  const pythonPath = path.join(process.cwd(), '.venv', 'bin', 'python');
  const listenerPath = path.join(process.cwd(), 'shutdown_listener.py');

  shutdownListenerProcess = spawn(pythonPath, ['-u', listenerPath], {
    stdio: ['inherit', 'pipe', 'pipe'],
  });

  isShutdownListenerStarting = false; // Reset after spawn

  shutdownListenerProcess.stdout?.on('data', async (data: Buffer) => {
    const text = data.toString().trim();

    // Print all output from shutdown listener for debugging
    if (text && text !== 'SHUTDOWN') {
      process.stdout.write(`[shutdown] ${text}\n`);
    }

    if (text.includes('SHUTDOWN')) {
      console.log('\n🛑 Shutdown phrase detected by background listener');
      killCurrentTTS();
      await new Promise((resolve) => setTimeout(resolve, 200));
      await backToSleep();
    }
  });

  shutdownListenerProcess.stderr?.on('data', (data: Buffer) => {
    const msg = data.toString().trim();
    if (msg) {
      console.error(`[shutdown:err] ${msg}`);
    }
  });

  shutdownListenerProcess.on('error', (err) => {
    console.error('[shutdown] spawn error:', err);
    shutdownListenerProcess = null; // Clear on spawn error
  });

  shutdownListenerProcess.on('close', (code) => {
    console.log(`[shutdown] exited with code ${code}`);
    shutdownListenerProcess = null;

    // Auto-restart if it crashed unexpectedly while app is still running
    if (isAppRunning && code !== 0) {
      console.warn(`[shutdown] unexpected exit code ${code}, restarting...`);
      // Small delay to avoid rapid restart loops
      setTimeout(() => {
        if (
          isAppRunning &&
          !shutdownListenerProcess &&
          !isShutdownListenerStarting
        ) {
          startShutdownListener();
        }
      }, 500);
    }
  });

  console.log('🎧 Shutdown listener started');
}

function stopShutdownListener(): void {
  isShutdownListenerStarting = false; // Reset flag on stop
  if (shutdownListenerProcess) {
    try {
      shutdownListenerProcess.kill('SIGTERM');
    } catch {}
    shutdownListenerProcess = null;
    console.log('🎧 Shutdown listener stopped');
  }
}
// end shutdown listener functions

function isCurrentRun(id: number) {
  return isAppRunning && id === runId;
}

function ms(n: number) {
  return `${Math.round(n)}ms`;
}

function waitForOperationsCompleteAsync(): Promise<void> {
  return new Promise((resolve) => {
    if (currentOperations.size === 0) {
      resolve();
      return;
    }

    const checkInterval = setInterval(() => {
      if (currentOperations.size === 0) {
        clearInterval(checkInterval);
        resolve();
      }
    }, 200);
  });
}

function trimHistory(messages: Msg[]) {
  const keep = 1 + MAX_TURNS * 2;
  if (messages.length > keep) {
    messages.splice(1, messages.length - keep);
  }
}

function clearHistoryTimeout() {
  if (historyTimeout) {
    clearTimeout(historyTimeout);
    historyTimeout = null;
  }
}

function clearRestartTimeout() {
  if (restartTimeout) {
    clearTimeout(restartTimeout);
    restartTimeout = null;
  }
}

function addSpeakingOperation() {
  currentOperations.add('TTSSpeaking');
  clearRestartTimeout();
}

function clearSpeakingOperation() {
  currentOperations.delete('TTSSpeaking');
}

function restartHistoryTimeout() {
  lastInteractionTime = Date.now();

  // Clear existing timeout
  clearHistoryTimeout();

  // Set new timeout to clear history after 5 minutes
  historyTimeout = setTimeout(() => {
    console.log(
      '🗑️  Clearing conversation history after 5 minutes of inactivity',
    );
    conversationHistory = [system];
    historyTimeout = null;
  }, HISTORY_TIMEOUT_MS);
}

async function backToSleep() {
  // Shutdown phrase and idle timeout can both fire; only one goodbye/restart.
  if (isGoingToSleep) return;
  isGoingToSleep = true;
  isAppRunning = false; // stop app loop and let restart handle it

  // Stop the shutdown listener first
  stopShutdownListener();

  // Stop recording so the chat loop exits now (and doesn't barge-in on the goodbye)
  killCurrentRecProcess();

  // Kill any current TTS before speaking goodbye
  killCurrentTTS();

  await speakTTS('Alright, going back to sleep.');
  // Wait for any remaining operations
  await waitForOperationsCompleteAsync();

  // Final cleanup
  clearRestartTimeout();
  clearHistoryTimeout();

  // Small buffer to ensure audio fully plays out
  await new Promise((resolve) => setTimeout(resolve, 500));
  isGoingToSleep = false; // restart() never returns (start() loops forever)
  await restart();
}

// Optional speech-friendly filters (applied before silence detection + transcription).
// Set to "0" to disable either filter.
const HIGHPASS_HZ = Number(process.env.HIGHPASS_HZ ?? '80'); // remove rumble
const LOWPASS_HZ = Number(process.env.LOWPASS_HZ ?? '7500'); // remove hiss/TV sparkle

function soxFilterArgs(): string[] {
  const args: string[] = [];
  if (Number.isFinite(HIGHPASS_HZ) && HIGHPASS_HZ > 0)
    args.push('highpass', String(HIGHPASS_HZ));
  if (Number.isFinite(LOWPASS_HZ) && LOWPASS_HZ > 0)
    args.push('lowpass', String(LOWPASS_HZ));
  return args;
}

// Voice activity detection threshold (RMS amplitude)
const VAD_RMS_THRESHOLD = 500; // Adjust based on your mic sensitivity

function calculateRMS(buffer: Buffer): number {
  // Assume 16-bit signed PCM audio
  let sumSquares = 0;
  const samples = buffer.length / 2;

  for (let i = 0; i < buffer.length; i += 2) {
    const sample = buffer.readInt16LE(i);
    sumSquares += sample * sample;
  }

  return Math.sqrt(sumSquares / samples);
}

async function recordUntilSilenceBytes(
  timeoutMs?: number,
  silenceDuration: string = SILENCE_DURATION_SEC,
): Promise<{ completedNormally: boolean; wavBytes: Buffer | null }> {
  const filterArgs = soxFilterArgs();

  let recordProcess: ReturnType<typeof spawn>;
  let monitorProcess: ReturnType<typeof spawn> | null = null;

  if (IS_LINUX) {
    // On Linux, use a single arecord and tee its output to both sox and a monitor
    recordProcess = spawn(
      'bash',
      [
        '-lc',
        [
          `set -o pipefail;`,
          `arecord -q -D ${LINUX_ARECORD_DEVICE} -f S16_LE -c ${LINUX_ARECORD_CHANNELS} -r ${LINUX_ARECORD_RATE} -t raw`,
          `| tee >(cat >&3)`, // send copy to fd 3 for monitoring voice activity
          `| sox -G -v ${INPUT_VOLUME} -t raw -r ${LINUX_ARECORD_RATE} -e signed-integer -b 16 -c ${LINUX_ARECORD_CHANNELS} - -t wav -c 1 -`,
          ...filterArgs,
          `silence 1 0.05 ${SILENCE_THRESHOLD} 1 ${silenceDuration} ${SILENCE_THRESHOLD}`,
        ].join(' '),
      ],
      {
        stdio: ['ignore', 'pipe', 'inherit', 'pipe'], // fd 3 = monitor output
        detached: true,
      },
    );
  } else {
    // macOS: two separate rec processes work fine (CoreAudio allows sharing)
    recordProcess = spawn(
      'rec',
      [
        '-G',
        '-D',
        '-c',
        '1',
        '-r',
        SAMPLE_RATE,
        '-b',
        '16',
        '-t',
        'wav',
        '-',
        'gain',
        String(MAC_GAIN_DB),

        ...(HIGHPASS_HZ > 0
          ? (['highpass', String(HIGHPASS_HZ)] as const)
          : []),
        ...(LOWPASS_HZ > 0 ? (['lowpass', String(LOWPASS_HZ)] as const) : []),

        'silence',
        '1',
        '0.05',
        SILENCE_THRESHOLD,
        '1',
        silenceDuration,
        SILENCE_THRESHOLD,
      ],
      { stdio: ['ignore', 'pipe', 'inherit'], detached: false },
    );

    // Separate monitor process for macOS
    monitorProcess = spawn(
      'rec',
      ['-q', '-c', '1', '-r', SAMPLE_RATE, '-b', '16', '-t', 'raw', '-'],
      { stdio: ['ignore', 'pipe', 'inherit'] },
    );
  }

  currentRecProcess = recordProcess;

  let killedByTimeout = false;
  let monitorBytes = 0;
  const chunks: Buffer[] = [];

  recordProcess.on('error', (err) => {
    console.error('❌ Recording process error:', err);
  });

  // Collect WAV data from main recording process (only arrives after silence detected)
  recordProcess.stdout?.on('data', (buf: Buffer) => {
    chunks.push(buf);
  });

  // Monitor raw audio bytes to detect voice activity in real-time
  // On Linux: use fd 3 (tee'd from arecord); on macOS: use separate rec process
  const monitorStream = IS_LINUX
    ? (recordProcess.stdio[3] as NodeJS.ReadableStream)
    : monitorProcess?.stdout;

  monitorStream?.on('data', (buf: Buffer) => {
    monitorBytes += buf.length;

    // Only check for voice activity once we have enough data
    if (monitorBytes > 1000) {
      const rms = calculateRMS(buf);

      // Voice detected only if audio amplitude exceeds threshold
      if (rms > VAD_RMS_THRESHOLD) {
        currentOperations.add('ActiveAsking');
        killCurrentTTS();

        if (restartTimeout) {
          clearRestartTimeout();
          clearHistoryTimeout();
        }
      }
    }
  });

  // Arm timeout only if no voice detected yet
  const monitor = setInterval(() => {
    if (timeoutMs && currentOperations.size === 0 && !restartTimeout) {
      restartTimeout = setTimeout(async () => {
        restartTimeout = null;
        killedByTimeout = true;
        killCurrentRecProcess();
        if (monitorProcess) {
          try {
            monitorProcess.kill('SIGTERM');
          } catch {}
        }
        console.log(
          `No voice detected for ${
            timeoutMs / 1000
          } seconds, going back to sleep.`,
        );
        await backToSleep();
      }, timeoutMs);
    }
  }, 200);

  try {
    await Promise.race([
      once(recordProcess, 'exit'),
      once(recordProcess, 'close'),
    ]);
  } finally {
    clearInterval(monitor);
    clearRestartTimeout();
    currentOperations.delete('ActiveAsking');
    if (monitorProcess) {
      try {
        monitorProcess.kill('SIGTERM');
      } catch {}
    }
    if (currentRecProcess === recordProcess) currentRecProcess = null;
  }

  if (killedByTimeout) {
    return { completedNormally: false, wavBytes: null };
  }

  const wavBytes = Buffer.concat(chunks);
  return {
    completedNormally: true,
    wavBytes: wavBytes.length > 0 ? wavBytes : null,
  };
}

const MIN_AUDIO_BYTES = 3500;

async function transcribeBytes(wavBytes: Buffer): Promise<string> {
  // Guard against audio that's too short
  if (wavBytes.length < MIN_AUDIO_BYTES) {
    console.log(
      `⚠️ Audio too short (${wavBytes.length} bytes), skipping transcription`,
    );
    return '';
  }

  // Groq's Whisper is significantly faster due to LPU hardware
  if (process.env.GROQ_API_KEY) {
    const resp = await groq.audio.transcriptions.create({
      model: 'whisper-large-v3-turbo',
      file: await toFile(wavBytes, `${ASSISTANT_NAME.toLowerCase()}-in.wav`),
    });
    return (resp.text ?? '').trim();
  }

  // Fallback to OpenAI
  const resp = await client.audio.transcriptions.create({
    model: TRANSCRIBE_MODEL,
    file: await toFile(wavBytes, `${ASSISTANT_NAME.toLowerCase()}-in.wav`),
  });
  return (resp.text ?? '').trim();
}

async function chat(messages: Msg[]): Promise<string> {
  // Groq is significantly faster for chat
  if (USE_GROQ_CHAT && process.env.GROQ_API_KEY) {
    const resp = await groq.chat.completions.create({
      model: GROQ_CHAT_MODEL,
      messages,
    });
    return resp.choices[0]?.message?.content?.trim() ?? '';
  }

  // Fallback to OpenAI
  const resp = await client.chat.completions.create({
    model: CHAT_MODEL,
    messages,
  });
  return resp.choices[0]?.message?.content?.trim() ?? '';
}

function killCurrentTTS() {
  ttsGen++;
  try {
    if (currentTtsProcess) {
      currentTtsProcess.kill('SIGKILL');
      currentTtsProcess = null;
    }
  } finally {
    clearSpeakingOperation();
  }
}

// The processing beep runs through the TTS fetch and stops the moment the voice starts.
async function speakTTS(text: string) {
  // Kill any previous playback (safety)
  killCurrentTTS();
  addSpeakingOperation();
  const gen = ttsGen;

  const cachedPath = await getCachedAudio(
    text,
    USE_CARTESIA_TTS ? CARTESIA_VOICE_ID : TTS_VOICE_ID,
  );
  if (gen !== ttsGen) return; // cancelled while looking up cache
  if (cachedPath) {
    console.log('⏱️ tts(cache)=hit');
    chimeProcessingStop();
    const speakProcess = spawn('play', ['-q', cachedPath], {
      stdio: ['ignore', 'inherit', 'inherit'],
    });
    currentTtsProcess = speakProcess;

    try {
      await Promise.race([
        once(speakProcess, 'exit'),
        once(speakProcess, 'close'),
      ]);
    } finally {
      finishTTS(speakProcess);
    }
    return;
  }

  // Use Cartesia for faster TTS
  if (USE_CARTESIA_TTS && process.env.CARTESIA_API_KEY) {
    await speakTTSCartesia(text, gen);
    return;
  }

  // Fallback to OpenAI TTS
  await speakTTSOpenAI(text, gen);
}

// Only tear down playback we still own; a newer speakTTS may have replaced it.
function finishTTS(proc: ReturnType<typeof spawn>) {
  if (currentTtsProcess === proc) {
    killCurrentTTS();
  } else {
    try {
      proc.kill('SIGKILL');
    } catch {}
  }
}

async function speakTTSCartesia(text: string, gen: number) {
  const tReq = performance.now();

  const response = await cartesia.tts.bytes(
    {
      modelId: 'sonic-2',
      transcript: text,
      voice: {
        mode: 'id',
        id: CARTESIA_VOICE_ID,
      },
      outputFormat: {
        container: 'wav',
        sampleRate: 44100,
        encoding: 'pcm_s16le',
      },
    },
    { timeoutInSeconds: API_TIMEOUT_MS / 1000, maxRetries: API_MAX_RETRIES },
  );

  const tResp = performance.now();
  console.log(`⏱️ tts(cartesia-api)=${ms(tResp - tReq)}`);

  // Consume the stream to get the actual bytes
  const chunks: Uint8Array[] = [];
  for await (const chunk of response) {
    chunks.push(chunk);
  }
  const audioBuffer = Buffer.concat(chunks);

  const tBuffer = performance.now();
  console.log(`⏱️ tts(cartesia-stream)=${ms(tBuffer - tResp)}`);

  if (gen !== ttsGen) return; // cancelled (barge-in / going to sleep) during fetch

  chimeProcessingStop();
  const speakProcess = spawn('play', ['-q', '-t', 'wav', '-'], {
    stdio: ['pipe', 'inherit', 'inherit'],
  });
  currentTtsProcess = speakProcess;

  if (speakProcess.stdin) {
    speakProcess.stdin.on('error', () => {});
    speakProcess.stdin.write(audioBuffer);
    speakProcess.stdin.end();
  }

  try {
    await Promise.race([
      once(speakProcess, 'exit'),
      once(speakProcess, 'close'),
      new Promise((resolve) => setTimeout(resolve, 60000)), // 60s timeout
    ]);
  } finally {
    const tDone = performance.now();
    console.log(`⏱️ tts(playback)=${ms(tDone - tBuffer)}`);
    finishTTS(speakProcess);
  }

  // Now audioBuffer is a proper Buffer, cacheAudio will work
  if (audioBuffer.length > 0 && CACHED_PHRASES.includes(text)) {
    await cacheAudio(text, CARTESIA_VOICE_ID, audioBuffer).catch(() => {});
  }
}

async function speakTTSOpenAI(text: string, gen: number) {
  const tReq = performance.now();
  const audio = await client.audio.speech.create(
    {
      model: TTS_MODEL,
      voice: TTS_VOICE_ID,
      input: text,
      response_format: 'wav',
    },
    { timeout: TTS_TIMEOUT_MS },
  );
  const tResp = performance.now();
  console.log(`⏱️ tts(api)=${ms(tResp - tReq)}`);

  if (gen !== ttsGen) {
    // cancelled (barge-in / going to sleep) during fetch
    void audio.body?.cancel().catch(() => {});
    return;
  }

  const chunks: Buffer[] = [];
  const speakProcess = spawn('play', ['-q', '-t', 'wav', '-'], {
    stdio: ['pipe', 'inherit', 'inherit'],
  });
  currentTtsProcess = speakProcess;

  if (speakProcess.stdin) {
    speakProcess.stdin.on('error', () => {});
  }

  const reader = audio.body?.getReader();
  let tFirstByte: number | null = null;
  let tFirstWrite: number | null = null;

  if (reader && speakProcess.stdin) {
    try {
      while (gen === ttsGen) {
        addSpeakingOperation();
        const { done, value } = await reader.read();
        if (done || gen !== ttsGen) break;

        if (tFirstByte === null) {
          tFirstByte = performance.now();
          console.log(`⏱️ tts(ttfb)=${ms(tFirstByte - tReq)}`);
        }

        const chunk = Buffer.from(value);
        chunks.push(chunk);

        if (!speakProcess.stdin.destroyed) {
          if (tFirstWrite === null) {
            tFirstWrite = performance.now();
            console.log(`⏱️ tts(first-write)=${ms(tFirstWrite - tReq)}`);
            chimeProcessingStop();
          }
          speakProcess.stdin.write(chunk);
        } else {
          break;
        }
      }
      if (!speakProcess.stdin.destroyed) speakProcess.stdin.end();
    } catch {
      // interrupted
    }
  }

  try {
    await Promise.race([
      once(speakProcess, 'exit'),
      once(speakProcess, 'close'),
      new Promise((resolve) => setTimeout(resolve, 60000)), // 60s timeout
    ]);
  } finally {
    const tDone = performance.now();
    if (tFirstWrite !== null) {
      console.log(`⏱️ tts(playback)=${ms(tDone - tFirstWrite)}`);
    }
    finishTTS(speakProcess);
  }

  const fullAudio = Buffer.concat(chunks);
  if (fullAudio.length > 0 && CACHED_PHRASES.includes(text)) {
    await cacheAudio(text, TTS_VOICE_ID, fullAudio).catch(() => {});
  }
}

// What the wake listener heard.
type WakeEvent = {
  // Bare wake word (the exact-match trigger); its chime is already playing.
  chime: Promise<void> | null;
  // "winterfresh <request>": audio from the wake word through the end of the request.
  commandWav: Buffer | null;
};

const BARE_WAKE: WakeEvent = { chime: null, commandWav: null };

async function startWakeWordListener(): Promise<WakeEvent> {
  if (wakeListenerProcess || isWakeListenerStarting) {
    console.log('🎤 Wake word listener already running or starting, skipping');
    return BARE_WAKE;
  }
  console.log('\n🎤 Listening for wake word (local Vosk)...');

  isWakeListenerStarting = true;
  const pythonPath = path.join(process.cwd(), '.venv', 'bin', 'python');
  const wakePath = path.join(process.cwd(), 'wake.py');

  // Use -u flag for unbuffered Python output.
  // --with-command: also capture "winterfresh <request>" (see Listener in wake.py).
  const proc = spawn(pythonPath, ['-u', wakePath, '--with-command'], {
    stdio: ['inherit', 'pipe', 'pipe'],
  });
  wakeListenerProcess = proc;

  isWakeListenerStarting = false; // Reset flag on start

  // Only clear the handle if it's still ours; a newer listener may own it.
  const release = () => {
    if (wakeListenerProcess === proc) wakeListenerProcess = null;
  };

  return new Promise((resolve, reject) => {
    let chime: Promise<void> | null = null;
    let pending = '';
    let settled = false;
    const settle = (done: () => void) => {
      if (settled) return;
      settled = true;
      proc.kill('SIGTERM');
      // Done listening even if it hasn't exited yet; a quick false alarm
      // restarts the listener before this one finishes shutting down.
      release();
      done();
    };

    proc.stdout?.on('data', (data: Buffer) => {
      const text = data.toString();
      process.stdout.write(text);

      // Protocol lines are whole lines; the level bar redraws in place with '\r'.
      pending += text;
      const lines = pending.split('\n');
      pending = lines.pop() ?? '';
      for (const raw of lines) {
        const line = raw.slice(raw.lastIndexOf('\r') + 1).trim();

        if (line === 'WAKE') {
          // Ack right away; wake.py keeps listening briefly in case the request follows.
          chime = chimeWakeDetected();
          // Kick off warmup immediately, but don't block the user flow.
          // This overlaps with chime + the user starting to talk.
          void warmUpApis();
        } else if (line === 'NO_COMMAND') {
          settle(() => resolve({ chime, commandWav: null }));
        } else if (line.startsWith('COMMAND ')) {
          if (!chime) void warmUpApis(); // overlap with transcription instead
          const wavPath = line.slice('COMMAND '.length);
          try {
            const commandWav = readFileSync(wavPath);
            rmSync(wavPath, { force: true });
            settle(() => resolve({ chime, commandWav }));
          } catch (err) {
            settle(() => reject(err));
          }
        }
      }
    });

    proc.stderr?.on('data', (data: Buffer) => {
      process.stderr.write(data);
    });

    proc.on('error', (err) => {
      isWakeListenerStarting = false;
      release();
      settle(() => reject(err));
    });

    proc.on('close', (code) => {
      isWakeListenerStarting = false;
      release();
      if (code !== 0 && code !== null && isAppRunning) {
        settle(() => reject(new Error(`Wake process exited with code ${code}`)));
      } else {
        // Resolve for clean exit, null code, or when app is stopping
        settle(() => resolve({ chime, commandWav: null }));
      }
    });
  });
}

// The name as one token: 'Winter fresh' -> 'winterfresh'
const WAKE_KEY = ASSISTANT_NAME.toLowerCase().replace(/[^a-z0-9]/g, '');

// "Hey, Winter-Fresh! Why is the sky blue?" -> "Why is the sky blue?"
// null if the transcript doesn't open with the name ("Winter's fresh snow..." doesn't).
function stripWakeWord(text: string): string | null {
  const tokens = [...text.matchAll(/[A-Za-z0-9']+/g)];
  let i = tokens[0]?.[0].toLowerCase() === 'hey' ? 1 : 0;
  let joined = '';
  let end = 0;
  for (; i < tokens.length && joined.length < WAKE_KEY.length; i++) {
    const token = tokens[i]!;
    joined += token[0].toLowerCase();
    end = token.index + token[0].length;
  }
  if (joined !== WAKE_KEY) return null;
  return text.slice(end).replace(/^[\s\p{P}]+/u, '').trim();
}

// Whisper's read of the "winterfresh <request>" audio from wake.py: the request,
// or '' if there was none. When Vosk only matched a prefix (not confirmed), Whisper
// must hear the name too; otherwise null, i.e. a false alarm.
async function transcribeWakeCommand(
  wav: Buffer,
  confirmed: boolean,
): Promise<string | null> {
  let text: string;
  try {
    const t0 = performance.now();
    // No Whisper prompt with the name: biasing it would defeat the confirmation.
    text = await transcribeBytes(wav);
    console.log(
      `⏱️ transcribeWakeCommand=${ms(performance.now() - t0)} (bytes=${wav.length})`,
    );
  } catch (err) {
    console.error('Wake command transcription error:', err);
    return confirmed ? '' : null;
  }
  console.log(`Heard: "${text}"`);
  return stripWakeWord(text) ?? (confirmed ? text : null);
}

// Chat on one user utterance and speak the reply.
async function respond(id: number, messages: Msg[], text: string) {
  console.log('You:', text);

  messages.push({ role: 'user', content: text });
  trimHistory(messages);

  const t2 = performance.now();
  // get reply from chat
  const reply = await chat(messages);
  const t3 = performance.now();
  console.log(`⏱️ chat=${ms(t3 - t2)}`);

  if (!isCurrentRun(id)) return;

  console.log(`${ASSISTANT_NAME} Reply:`, reply);

  messages.push({ role: 'assistant', content: reply });
  trimHistory(messages);

  restartHistoryTimeout();

  // speak reply from chat
  const t4 = performance.now();
  await speakTTS(reply);
  const t5 = performance.now();
  console.log(`⏱️ speakTTS(total)=${ms(t5 - t4)}`);
}

async function apologize(id: number, err: unknown) {
  console.error('Processing error:', err);
  // Say something so a failed/timed-out request doesn't sound like a crash
  if (isCurrentRun(id)) {
    await speakTTS(ERROR_PHRASE).catch(() => {});
  }
}

async function startChatSession(id: number, wake: WakeEvent) {
  const messages = conversationHistory;

  const historyAge =
    lastInteractionTime > 0 ? Date.now() - lastInteractionTime : 0;
  const isReturning = messages.length > 1 && historyAge < HISTORY_TIMEOUT_MS;

  // "winterfresh <request>" in one go: answer it instead of asking "What's up?"
  let request = '';
  if (wake.commandWav) {
    const heard = await transcribeWakeCommand(
      wake.commandWav,
      wake.chime !== null,
    );
    if (heard === null) {
      console.log('🙅 Wake word not confirmed by transcription, ignoring');
      return;
    }
    request = heard;
  }

  if (!isCurrentRun(id)) return;

  if (request) {
    void (wake.chime ?? chimeWakeDetected()); // ack while the chat request runs
    startShutdownListener();
    chimeProcessingStart();
    await respond(id, messages, request).catch((err) => apologize(id, err));
    chimeProcessingStop();
  } else {
    await (wake.chime ?? chimeWakeDetected());

    if (isReturning) {
      await speakTTS("Welcome back! What's up?");
    } else {
      await speakTTS("What's up?");
    }

    if (!isCurrentRun(id)) return;

    // Start background shutdown listener
    startShutdownListener();
  }

  let abortPending = false;

  while (isCurrentRun(id)) {
    console.log('\n--- Speak now (auto-stops on silence) ---');

    const tRec0 = performance.now();
    const { completedNormally, wavBytes } = await recordUntilSilenceBytes(
      IDLE_TIMEOUT_MS,
      SILENCE_DURATION_SEC,
    );
    const tRec1 = performance.now();
    console.log(
      `⏱️ recordUntilSilenceBytes=${ms(tRec1 - tRec0)} (bytes=${
        wavBytes?.length ?? 0
      })`,
    );

    if (!completedNormally) return;
    if (!isCurrentRun(id)) return;

    chimeProcessingStart();
    console.log('Processing voice input...');
    clearRestartTimeout();

    // Nothing captured (e.g., very short noise / process ended before output)
    if (!wavBytes || wavBytes.length === 0) {
      chimeProcessingStop();
      continue;
    }

    abortPending = true;

    // add await to block until tts is done
    await (async () => {
      abortPending = false;

      try {
        const t0 = performance.now();
        // transcribe audio that was recorded
        const text = await transcribeBytes(wavBytes);
        const t1 = performance.now();
        console.log(
          `⏱️ transcribeBytes=${ms(t1 - t0)} (bytes=${wavBytes.length})`,
        );

        if (!isCurrentRun(id) || abortPending) return;

        if (!text) {
          chimeProcessingStop();
          return;
        }

        await respond(id, messages, text);
      } catch (err) {
        await apologize(id, err);
      } finally {
        chimeProcessingStop();
      }
    })();
  }
}

async function restart() {
  console.log(`\n🔄 Restarting ${ASSISTANT_NAME}...`);
  await stop();
  // Small delay to ensure resources are freed
  await new Promise((resolve) => setTimeout(resolve, 500));
  await start();
}

function killCurrentRecProcess() {
  if (!currentRecProcess) return;

  const proc = currentRecProcess;
  currentRecProcess = null;

  if (IS_LINUX && proc.pid !== undefined) {
    // Check pid exists before using it
    // Kill entire process group (bash + arecord + sox)
    try {
      process.kill(-proc.pid, 'SIGTERM');
    } catch (err) {
      console.error('SIGTERM process group failed:', err);
    }
    setTimeout(() => {
      try {
        if (proc.pid !== undefined) {
          process.kill(-proc.pid, 'SIGKILL');
        }
      } catch {}
    }, 500);
    return;
  }

  // macOS/fallback
  try {
    proc.kill('SIGKILL');
  } catch {}
}

async function stop() {
  console.log(`\n🛑 Stopping ${ASSISTANT_NAME}...`);
  isAppRunning = false;

  // Stop shutdown listener
  stopShutdownListener();

  // Wait for current TTS to finish naturally (up to 5s) before killing
  if (currentTtsProcess) {
    const ttsProc = currentTtsProcess;
    const ttsFinished = Promise.race([
      once(ttsProc, 'exit'),
      once(ttsProc, 'close'),
      new Promise((resolve) => setTimeout(resolve, 5000)), // max wait
    ]);
    await ttsFinished;

    // Now safe to force-kill if still running
    if (currentTtsProcess) {
      currentTtsProcess.kill('SIGKILL');
      currentTtsProcess = null;
    }
  }

  // Kill recording process if running
  killCurrentRecProcess();

  // clear chimeProcessingStop incase its running for whatever reason
  chimeProcessingStop();

  // Clean up temp files and zombie processes
  cleanupZombieProcesses();

  console.log('✅ Cleanup complete');
}

async function warmUpApis(): Promise<void> {
  try {
    const t0 = performance.now();
    // Warm up chat endpoint with a minimal request
    if (USE_GROQ_CHAT && process.env.GROQ_API_KEY) {
      await groq.chat.completions.create({
        model: GROQ_CHAT_MODEL,
        messages: [{ role: 'user', content: 'hi' }],
        max_tokens: 5,
      });
    } else {
      await client.chat.completions.create({
        model: CHAT_MODEL,
        messages: [{ role: 'user', content: 'hi' }],
        max_completion_tokens: 5,
      });
    }
    const t1 = performance.now();
    console.log(`⏱️ warmup(chat)=${ms(t1 - t0)}`);
  } catch (err) {
    // warmup should never block startup
    console.warn('warmup(chat) failed:', err);
  }
}

async function start() {
  isAppRunning = true;
  const id = ++runId;
  resetErrorCounter();

  while (isCurrentRun(id)) {
    try {
      const wake = await startWakeWordListener();
      if (!isCurrentRun(id)) break;

      recordSuccess(); // Wake word detection succeeded

      await startChatSession(id, wake);
      if (!isCurrentRun(id)) break;

      recordSuccess(); // Session completed successfully
    } catch (err) {
      if (isCurrentRun(id)) {
        const shouldRestart = recordError(err);
        if (shouldRestart) {
          console.log('🔄 Attempting recovery restart...');
          await restart();
          return;
        }
      }
    }
  }
}

async function main() {
  console.log(`🌨️  ${ASSISTANT_NAME} starting...`);

  // Run pre-flight checks (not async, remove await)
  const checksPass = runPreflightChecks();
  if (!checksPass) {
    console.error('❌ Pre-flight checks failed. Exiting.');
    process.exit(1);
  }

  // Handle graceful shutdown
  process.on('SIGINT', async () => {
    console.log(`\n🛑 SIGINT received, shutting down ${ASSISTANT_NAME}...`);
    await stop();
    process.exit(0);
  });

  process.on('SIGTERM', async () => {
    console.log(`\n🛑 SIGTERM received, shutting down ${ASSISTANT_NAME}...`);
    await stop();
    process.exit(0);
  });

  await start();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
