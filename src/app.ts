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
const SAMPLE_RATE = process.env.SAMPLE_RATE ?? '24000';
const CHAT_MODEL = process.env.CHAT_MODEL ?? 'gpt-4o-mini';
const TRANSCRIBE_MODEL =
  process.env.TRANSCRIBE_MODEL ?? 'gpt-4o-mini-transcribe';
const TTS_MODEL = process.env.TTS_MODEL ?? 'gpt-4o-mini-tts';
const DEFAULT_RULES = [
  'Default to one or two sentences if possible, unless more detail is requested.',
  'Be direct and honest. Never sugarcoat, never be rude.',
  'Match depth to the question: If the user asks "why/how" or asks for context, teach briefly. Otherwise answer directly; add at most one short extra sentence if it improves understanding.',
  'If a question is ambiguous, ask one clarifying question and wait.',
  "If you don't know, say so clearly.",
  'When listing steps or options, use bullet points.',
  'I am transcribing my speech, so you hear an audio transcription, not perfect text.',
].join('\n- ');
const ASSISTANT_RULES = process.env.ASSISTANT_RULES ?? DEFAULT_RULES;

type Msg = { role: 'system' | 'user' | 'assistant'; content: string };
const system: Msg = {
  role: 'system',
  content: `
    You are Winterfresh, a helpful assistant voice assistant that prioritizes answering in one sentence
    
    Rules:
    - VERY IMPORTANT: If you get the sentiment based on my response like "stop", "thats enough", "I got it", "I'm done" etc...
      that I don't want more information, respond exactly with "shutting down" and stop further responses.
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
const SILENCE_DURATION_SEC = '1.5';
const MAC_GAIN_DB = 6; // ~20*log10(2) = +6.02 dB

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

async function recordUntilSilence(
  outPath: string,
  timeoutMs?: number,
  silenceDuration: string = SILENCE_DURATION_SEC,
): Promise<boolean> {
  // Return whether recording completed normally
  const recordProcess = spawn(
    'rec',
    [
      '-G', // guard against clipping
      '-D', // disable dithering (avoids "dither clipped" warnings)
      '-c',
      '1',
      '-r',
      SAMPLE_RATE,
      '-b',
      '16',
      outPath,
      'gain',
      String(MAC_GAIN_DB),
      'silence',
      '1',
      '0.05',
      SILENCE_THRESHOLD,
      '1',
      silenceDuration,
      SILENCE_THRESHOLD,
    ],
    { stdio: 'inherit', detached: IS_LINUX }, // detached for process group on Linux
  );

  currentRecProcess = recordProcess;
  let killedByTimeout = false;

  // Detect spawn errors (mic disconnected, permission issues, etc.)
  recordProcess.on('error', (err) => {
    console.error('❌ Recording process error:', err);
  });

  // Monitor file size to detect active recording
  const monitorVoiceIn = setInterval(async () => {
    try {
      const stats = await fs.stat(outPath).catch(() => null);
      if (stats && stats.size > 1000) {
        currentOperations.add('ActiveAsking');
        killCurrentTTS();
      }

      if (getRunningOperations().length > 0 && restartTimeout) {
        clearRestartTimeout();
        clearHistoryTimeout();
      }

      // Add timeout to record if specified, so we can stop active session after inactivity
      if (timeoutMs && getRunningOperations().length === 0 && !restartTimeout) {
        restartTimeout = setTimeout(async () => {
          restartTimeout = null;
          killedByTimeout = true;
          killCurrentRecProcess();
          console.log(
            `No voice detected for ${
              timeoutMs / 1000
            } seconds, going back to sleep.`,
          );
          await backToSleep();
        }, timeoutMs);
      }
    } catch (err) {
      // Ignore
    }
  }, 200);

  try {
    await Promise.race([
      once(recordProcess, 'exit'),
      once(recordProcess, 'close'),
    ]);
  } catch (err) {
    console.error('Recording error:', err);
    throw err;
  } finally {
    clearInterval(monitorVoiceIn);
    clearRestartTimeout();
    currentOperations.delete('ActiveAsking');
    if (currentRecProcess === recordProcess) {
      currentRecProcess = null;
    }
  }

  return !killedByTimeout;
}

async function transcribe(wavPath: string): Promise<string> {
  const bytes = await fs.readFile(wavPath);
  const resp = await client.audio.transcriptions.create({
    model: TRANSCRIBE_MODEL,
    file: await toFile(bytes, 'winterfresh-in.wav'),
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

  // Check cache first
  const cachedPath = await getCachedAudio(text, TTS_VOICE);
  if (cachedPath) {
    console.log(
      '🔊 Playing cached TTS:',
      text.slice(0, 30) + (text.length > 30 ? '...' : ''),
    );
    const speakProcess = spawn('play', ['-q', cachedPath], {
      stdio: ['pipe', 'inherit', 'inherit'],
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

  // Not cached - fetch from API
  const audio = await client.audio.speech.create({
    model: TTS_MODEL,
    voice: TTS_VOICE,
    input: text,
    response_format: 'mp3',
  });

  // Collect audio data for caching
  const chunks: Buffer[] = [];

  const speakProcess = spawn('play', ['-q', '-t', 'mp3', '-'], {
    stdio: ['pipe', 'inherit', 'inherit'],
  });
  currentTtsProcess = speakProcess;

  if (speakProcess.stdin) {
    speakProcess.stdin.on('error', () => {});
  }

  const reader = audio.body?.getReader();
  if (reader && speakProcess.stdin) {
    try {
      while (true) {
        addSpeakingOperation();
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = Buffer.from(value);
        chunks.push(chunk);
        if (!speakProcess.stdin.destroyed) {
          speakProcess.stdin.write(chunk);
        } else {
          break;
        }
      }
      if (!speakProcess.stdin.destroyed) {
        speakProcess.stdin.end();
      }
    } catch (err) {
      // Handle interruption gracefully
    }
  }

  try {
    await Promise.race([
      once(speakProcess, 'exit'),
      once(speakProcess, 'close'),
    ]);
  } finally {
    killCurrentTTS();
  }

  // Cache if this is a common phrase (or all short phrases)
  const fullAudio = Buffer.concat(chunks);
  if (fullAudio.length > 0 && CACHED_PHRASES.includes(text)) {
    await cacheAudio(text, TTS_VOICE, fullAudio).catch((err) => {
      console.error('Failed to cache TTS:', err);
    });
    console.log(
      '💾 Cached TTS:',
      text.slice(0, 30) + (text.length > 30 ? '...' : ''),
    );
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

function isStopIntent(text: string): boolean {
  const cmd = normalizeSpokenCommand(text);
  return STOP_INTENTS.some((phrase) => cmd.includes(phrase));
}

async function startChatSession() {
  const messages = conversationHistory;

  const historyAge =
    lastInteractionTime > 0 ? Date.now() - lastInteractionTime : 0;
  const isReturning = messages.length > 1 && historyAge < HISTORY_TIMEOUT_MS;

  await chimeWakeDetected();

  if (isReturning) {
    await speakTTS('Welcome back! How can I assist you further?');
  } else {
    await speakTTS("What's up?");
  }

  let abortPending = false;

  while (true && isAppRunning) {
    console.log('\n--- Speak now (auto-stops on silence) ---');
    const wavPath = `/tmp/winterfresh-in-${Date.now()}.wav`;

    const voiceRecCompletedNormally = await recordUntilSilence(
      wavPath,
      IDLE_TIMEOUT_MS,
      SILENCE_DURATION_SEC,
    );

    chimeProcessingStart();
    console.log('Processing voice input...');
    clearRestartTimeout();

    // If killed by timeout, exit - restart() was already called
    if (!voiceRecCompletedNormally) {
      fs.unlink(wavPath).catch(() => {});
      return;
    }

    if (!isAppRunning) return;

    abortPending = true;

    // add await to block until tts is done
    await (async () => {
      abortPending = false;

      try {
        const text = await transcribe(wavPath);

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

        const reply = await chat(messages);
        const replyCmd = normalizeSpokenCommand(reply);
        if (replyCmd.includes('shutting down')) {
          console.log('🛑 Winterfresh shutting down per user request.');
          await backToSleep();
          return;
        }
        if (!isAppRunning || abortPending) return;

        console.log('Winterfresh:', reply);

        messages.push({ role: 'assistant', content: reply });
        trimHistory(messages);

        restartHistoryTimeout();

        await speakTTS(reply);
      } catch (err) {
        console.error('Processing error:', err);
        chimeProcessingStop();
      } finally {
        fs.unlink(wavPath).catch(() => {});
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
        await new Promise((resolve) => setTimeout(resolve, 2000));
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
