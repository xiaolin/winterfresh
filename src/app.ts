import 'dotenv/config';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import fs from 'node:fs/promises';
import path from 'node:path';
import OpenAI, { toFile } from 'openai';

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

import { getCachedAudio, cacheAudio, CACHED_PHRASES } from './tts-cache.js';

const IS_LINUX = process.platform === 'linux';
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const SAMPLE_RATE = process.env.SAMPLE_RATE ?? '16000';
const LINUX_ARECORD_DEVICE = process.env.ARECORD_DEVICE ?? 'plughw:2,0';
const LINUX_ARECORD_RATE = process.env.ARECORD_RATE ?? '16000';
const LINUX_ARECORD_CHANNELS = process.env.ARECORD_CHANNELS ?? '1';
const CHAT_MODEL = process.env.CHAT_MODEL ?? 'gpt-4o-mini';
const TRANSCRIBE_MODEL =
  process.env.TRANSCRIBE_MODEL ?? 'gpt-4o-mini-transcribe';
const TTS_MODEL = process.env.TTS_MODEL ?? 'gpt-4o-mini-tts';
const DEFAULT_RULES = [
  'Prioritize answering in one sentence whenever possible.',
  'Be direct and honest. Never sugarcoat, never be rude.',
  'I am transcribing my speech, so you hear an audio transcription, not perfect text.',
  'You should be responding in english unless explicitly asked to use another language.',
].join('\n- ');
const ASSISTANT_RULES = process.env.ASSISTANT_RULES ?? DEFAULT_RULES;
const STOP_INTENTS = [
  'stop',
  'shut up',
  'be quiet',
  'quiet',
  'enough',
  "that's enough",
  'thats enough',
  'i got it',
  'got it',
  'never mind',
  'nevermind',
  'cancel',
  'go away',
  'go to sleep',
  'goodbye',
  'bye',
  'winter fresh stop',
  'winterfresh stop',
];

type Msg = { role: 'system' | 'user' | 'assistant'; content: string };
const system: Msg = {
  role: 'system',
  content: `
    You are Winterfresh, a helpful assistant voice assistant that prioritizes answering in one sentence
  
    - ${ASSISTANT_RULES},
  `,
};

const MAX_TURNS = Number(process.env.WINTERFRESH_MAX_TURNS ?? 20);
const IDLE_TIMEOUT_MS = 7000; // 7 seconds
const HISTORY_TIMEOUT_MS = 5 * 60 * 1000; // 5 minutes
const TTS_VOICE = 'alloy';

let currentTtsProcess: ReturnType<typeof spawn> | null = null;
let currentRecProcess: ReturnType<typeof spawn> | null = null;
let isAppRunning = false;
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

function ms(n: number) {
  return `${Math.round(n)}ms`;
}

function normalizeSpokenCommand(text: string): string {
  return (text ?? '')
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ') // drop punctuation
    .replace(/\s+/g, ' ') // collapse whitespace
    .trim();
}

function getRunningOperations(): string[] {
  return Array.from(currentOperations);
}

function waitForOperationsCompleteAsync(): Promise<void> {
  return new Promise((resolve) => {
    if (currentOperations.size === 0) {
      resolve();
      return;
    }

    const checkInterval = setInterval(async () => {
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
  isAppRunning = false; // stop app loop and let restart handle it
  await speakTTS('Alright, going back to sleep.');
  await waitForOperationsCompleteAsync();
  clearSpeakingOperation();
  clearRestartTimeout();
  clearSpeakingOperation();
  killCurrentTTS();

  // Small buffer to ensure audio fully plays out
  await new Promise((resolve) => setTimeout(resolve, 500));
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
          `| tee >(cat >&3)`, // send copy to fd 3 for monitoring
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
    if (timeoutMs && getRunningOperations().length === 0 && !restartTimeout) {
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

async function transcribeBytes(wavBytes: Buffer): Promise<string> {
  const resp = await client.audio.transcriptions.create({
    model: TRANSCRIBE_MODEL,
    file: await toFile(wavBytes, 'winterfresh-in.wav'),
  });
  return (resp.text ?? '').trim();
}

async function chat(messages: Msg[]): Promise<string> {
  const resp = await client.chat.completions.create({
    model: CHAT_MODEL,
    messages,
  });
  return resp.choices[0]?.message?.content?.trim() ?? '';
}

async function killCurrentTTS() {
  try {
    if (currentTtsProcess) {
      currentTtsProcess.kill('SIGKILL');
      currentTtsProcess = null;
    }
  } finally {
    clearSpeakingOperation();
  }
}

async function speakTTS(text: string) {
  // Stop processing chime when TTS starts
  chimeProcessingStop();
  // Kill any previous playback (safety)
  killCurrentTTS();
  addSpeakingOperation();

  const cachedPath = await getCachedAudio(text, TTS_VOICE);
  if (cachedPath) {
    console.log('⏱️ tts(cache)=hit');
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
      killCurrentTTS();
    }
    return;
  }

  const tReq = performance.now();
  const audio = await client.audio.speech.create({
    model: TTS_MODEL,
    voice: TTS_VOICE,
    input: text,
    response_format: 'wav',
  });
  const tResp = performance.now();
  console.log(`⏱️ tts(api)=${ms(tResp - tReq)}`);

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
      while (true) {
        addSpeakingOperation();
        const { done, value } = await reader.read();
        if (done) break;

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
    ]);
  } finally {
    const tDone = performance.now();
    if (tFirstWrite !== null) {
      console.log(`⏱️ tts(playback)=${ms(tDone - tFirstWrite)}`);
    }
    killCurrentTTS();
  }

  const fullAudio = Buffer.concat(chunks);
  if (fullAudio.length > 0 && CACHED_PHRASES.includes(text)) {
    await cacheAudio(text, TTS_VOICE, fullAudio).catch(() => {});
  }
}

async function waitForWakeWord(): Promise<void> {
  console.log('\n🎤 Listening for wake word (local Vosk)...');

  const pythonPath = path.join(process.cwd(), '.venv', 'bin', 'python');
  const wakePath = path.join(process.cwd(), 'wake.py');

  // Use -u flag for unbuffered Python output
  const wakeProcess = spawn(pythonPath, ['-u', wakePath], {
    stdio: ['inherit', 'pipe', 'pipe'],
  });

  return new Promise((resolve, reject) => {
    wakeProcess.stdout?.on('data', (data: Buffer) => {
      const text = data.toString();
      process.stdout.write(text);

      if (text.includes('WAKE')) {
        // Kick off warmup immediately, but don't block the user flow.
        // This overlaps with chime + the user starting to talk.
        void warmUpApis();
        wakeProcess.kill('SIGTERM');
        resolve();
      }
    });

    wakeProcess.stderr?.on('data', (data: Buffer) => {
      process.stderr.write(data);
    });

    wakeProcess.on('error', reject);
    wakeProcess.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else if (code !== null && isAppRunning) {
        reject(new Error(`Wake process exited with code ${code}`));
      }
    });
  });
}

const STOP_INTENTS_NORMALIZED = new Set(
  STOP_INTENTS.map(normalizeSpokenCommand),
);

function isStopIntent(text: string): boolean {
  const cmd = normalizeSpokenCommand(text);
  return STOP_INTENTS_NORMALIZED.has(cmd);
}

async function startChatSession() {
  const messages = conversationHistory;

  const historyAge =
    lastInteractionTime > 0 ? Date.now() - lastInteractionTime : 0;
  const isReturning = messages.length > 1 && historyAge < HISTORY_TIMEOUT_MS;

  await chimeWakeDetected();

  if (isReturning) {
    await speakTTS("Welcome back! What's up?");
  } else {
    await speakTTS("What's up?");
  }

  let abortPending = false;

  while (true && isAppRunning) {
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
    if (!isAppRunning) return;

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

        if (!isAppRunning || abortPending) return;

        // Check for stop intent (before calling chat)
        if (isStopIntent(text)) {
          console.log('🛑 Stop intent detected:', text);
          await backToSleep();
          return;
        }

        if (!text) {
          chimeProcessingStop();
          return;
        }

        console.log('You:', text);

        messages.push({ role: 'user', content: text });
        trimHistory(messages);

        const t2 = performance.now();
        // get reply from chat
        const reply = await chat(messages);
        const t3 = performance.now();
        console.log(`⏱️ chat=${ms(t3 - t2)}`);

        const replyCmd = normalizeSpokenCommand(reply);
        if (replyCmd === 'shutting down') {
          console.log('🛑 Winterfresh shutting down per sentiment request.');
          await backToSleep();
          return;
        }
        if (!isAppRunning || abortPending) return;

        console.log('Winterfresh Reply:', reply);

        messages.push({ role: 'assistant', content: reply });
        trimHistory(messages);

        restartHistoryTimeout();

        // speak reply from chat
        const t4 = performance.now();
        await speakTTS(reply);
        const t5 = performance.now();
        console.log(`⏱️ speakTTS(total)=${ms(t5 - t4)}`);
      } catch (err) {
        console.error('Processing error:', err);
      } finally {
        chimeProcessingStop();
      }
    })();
  }
}

async function restart() {
  console.log('\n🔄 Restarting Winterfresh...');
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
  console.log('\n🛑 Stopping Winterfresh...');
  isAppRunning = false;

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
    await client.chat.completions.create({
      model: CHAT_MODEL,
      messages: [{ role: 'user', content: 'hi' }],
      max_completion_tokens: 5,
    });
    const t1 = performance.now();
    console.log(`⏱️ warmup(chat)=${ms(t1 - t0)}`);
  } catch (err) {
    // warmup should never block startup
    console.warn('warmup(chat) failed:', err);
  }
}

async function start() {
  isAppRunning = true;
  resetErrorCounter();

  while (isAppRunning) {
    try {
      await waitForWakeWord();
      if (!isAppRunning) break;

      recordSuccess(); // Wake word detection succeeded

      await startChatSession();
      if (!isAppRunning) break;

      recordSuccess(); // Session completed successfully
    } catch (err) {
      if (isAppRunning) {
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
  console.log('🌨️  Winterfresh starting...');

  // Run pre-flight checks (not async, remove await)
  const checksPass = runPreflightChecks();
  if (!checksPass) {
    console.error('❌ Pre-flight checks failed. Exiting.');
    process.exit(1);
  }

  // Handle graceful shutdown
  process.on('SIGINT', async () => {
    console.log('\n🛑 SIGINT received, shutting down Winterfresh...');
    await stop();
    process.exit(0);
  });

  process.on('SIGTERM', async () => {
    console.log('\n🛑 SIGTERM received, shutting down Winterfresh...');
    await stop();
    process.exit(0);
  });

  await start();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
