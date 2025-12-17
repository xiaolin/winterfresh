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

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const SAMPLE_RATE = process.env.SAMPLE_RATE ?? '24000';
const CHAT_MODEL = process.env.CHAT_MODEL ?? 'gpt-4o-mini';
const TRANSCRIBE_MODEL =
  process.env.TRANSCRIBE_MODEL ?? 'gpt-4o-mini-transcribe';
const TTS_MODEL = process.env.TTS_MODEL ?? 'gpt-4o-mini-tts';

type Msg = { role: 'system' | 'user' | 'assistant'; content: string };

const system: Msg = {
  role: 'system',
  content: `
    You are Winterfresh, a fast, concise home voice assistant.

    Rules:
    - Default to the shortest correct answer.
    - Use plain language. No filler, no hedging.
    - Be direct and honest. Never sugarcoat, never be rude.
    - Match depth to the question:
      - If the user asks "why/how" or asks for context, teach briefly.
      - Otherwise answer directly; add at most one short extra sentence if it improves understanding.
    - Prefer practical, real-world answers over theory unless the user asks for theory/history.
    - If a question is ambiguous, ask one clarifying question and wait.
    - If you don't know, say so clearly.
    - When listing steps or options, use bullet points.
  `,
};

const MAX_TURNS = Number(process.env.WINTERFRESH_MAX_TURNS ?? 20);
const IDLE_TIMEOUT_MS = 7000; // 7 seconds
const HISTORY_TIMEOUT_MS = 5 * 60 * 1000; // 5 minutes

let currentTtsProcess: ReturnType<typeof spawn> | null = null;
let currentRecProcess: ReturnType<typeof spawn> | null = null;
let isAppRunning = false;
let restartTimeout: NodeJS.Timeout | null = null;
let historyTimeout: NodeJS.Timeout | null = null;

// Persistent conversation history
let conversationHistory: Msg[] = [system];
let lastInteractionTime = 0;

// Track current operations
type Operation = 'TTSSpeaking' | 'activeAsking';
const currentOperations = new Set<Operation>();

function getRunningOperations(): string[] {
  return Array.from(currentOperations);
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

async function recordUntilSilence(
  outPath: string,
  timeoutMs?: number,
  silenceDuration: string = '1.0', // Make this configurable
) {
  const args = [
    '-c',
    '1', // Number of channels (1 = mono, 2 = stereo)
    '-r',
    SAMPLE_RATE, // Sample rate in Hz (e.g., 48000 = 48kHz)
    '-b',
    '16', // Bit depth (16-bit audio quality)
    outPath, // Output file path where recording will be saved
    'silence', // Enable silence detection to auto-stop recording
    '1', // Stop recording after detecting 1 period of silence
    '0.10', // Minimum duration of sound before silence (0.10 = 100ms)
    '2%', // Threshold for sound detection (2% of max volume)
    '1', // Number of consecutive silence periods to detect before stopping
    silenceDuration, // Duration of silence required to stop (e.g., '1.0' = 1 second)
    '2%', // Threshold for silence detection (2% of max volume)
  ];

  const recordProcess = spawn('rec', args, { stdio: 'inherit' });
  currentRecProcess = recordProcess;

  // Detect spawn errors (mic disconnected, permission issues, etc.)
  recordProcess.on('error', (err) => {
    console.error('❌ Recording process error:', err);
    console.error('Mic may be disconnected or permission denied');
  });

  // Monitor file size to detect active recording
  const monitorVoiceIn = setInterval(async () => {
    try {
      const stats = await fs.stat(outPath).catch(() => null);
      // If file exists and has content (> 1KB WAV header), we're actively recording
      if (stats && stats.size > 1000) {
        currentOperations.add('activeAsking');
        killCurrentTTS(); // we started interupting, kill TTS if any
      }

      if (getRunningOperations().length > 0 && restartTimeout) {
        clearTimeout(restartTimeout);
        restartTimeout = null;
        // clear history timeout as well
        clearHistoryTimeout();
      }

      // Add timeout to record if specified, so we can stop active session after inactivity
      if (timeoutMs && getRunningOperations().length === 0 && !restartTimeout) {
        restartTimeout = setTimeout(async () => {
          restartTimeout = null;
          await speakTTS('Alright, going back to sleep.');
          restart();
        }, timeoutMs);
      }
    } catch (err) {
      // Ignore
    }
  }, 200); // Check every 200ms

  try {
    const promises = [
      once(recordProcess, 'exit'),
      once(recordProcess, 'close'),
    ];
    await Promise.race(promises);
  } catch (err) {
    console.error('Recording error:', err);
    throw err;
  } finally {
    clearInterval(monitorVoiceIn);
    if (restartTimeout) {
      clearTimeout(restartTimeout);
      restartTimeout = null;
    }
    currentOperations.delete('activeAsking');
    if (currentRecProcess === recordProcess) {
      currentRecProcess = null;
    }
  }
}

async function transcribe(path: string): Promise<string> {
  const bytes = await fs.readFile(path);
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
    currentOperations.delete('TTSSpeaking');
  }
}

async function speakTTS(text: string) {
  // Stop processing chime when TTS starts
  chimeProcessingStop();

  // Kill any previous playback (safety)
  killCurrentTTS();

  currentOperations.add('TTSSpeaking');
  const audio = await client.audio.speech.create({
    model: TTS_MODEL,
    voice: 'alloy',
    input: text,
    response_format: 'mp3',
  });

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
        currentOperations.add('TTSSpeaking');
        const { done, value } = await reader.read();
        if (done) break;
        if (!speakProcess.stdin.destroyed) {
          speakProcess.stdin.write(Buffer.from(value));
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

async function activeSession() {
  const messages = conversationHistory;

  const historyAge =
    lastInteractionTime > 0 ? Date.now() - lastInteractionTime : 0;
  const isReturning = messages.length > 1 && historyAge < HISTORY_TIMEOUT_MS;

  await chimeWakeDetected();

  if (isReturning) {
    speakTTS('Welcome back! How can I assist you further?');
  } else {
    speakTTS('Whats up?');
  }

  // Track pending work so we don't pile up requests
  let abortPending = false;

  while (true) {
    console.log('\n--- Speak now (auto-stops on silence) ---');
    const wavPath = `/tmp/winterfresh-in-${Date.now()}.wav`;

    await recordUntilSilence(wavPath, IDLE_TIMEOUT_MS, '1.0');

    if (!isAppRunning) return;

    // Signal any previous pending work to abort
    abortPending = true;

    // Start processing transcribe and chat in background, don't await
    (async () => {
      // This task is now the "current" one, reset flag
      abortPending = false;

      chimeProcessingStart();

      try {
        const text = await transcribe(wavPath);
        if (!isAppRunning || abortPending) return; // Check if newer task started

        if (!text) {
          chimeProcessingStop();
          return;
        }

        console.log('You:', text);

        messages.push({ role: 'user', content: text });
        trimHistory(messages);

        const reply = await chat(messages);
        if (!isAppRunning || abortPending) return; // Check if newer task started

        console.log('Winterfresh:', reply);

        messages.push({ role: 'assistant', content: reply });
        trimHistory(messages);

        restartHistoryTimeout();

        speakTTS(reply);
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
  await new Promise((resolve) => setTimeout(resolve, 1000)); // Wait for cleanup
  await start();
}

async function stop() {
  console.log('\n🛑 Stopping Winterfresh...');
  isAppRunning = false;

  // Kill TTS process if running
  if (currentTtsProcess) {
    currentTtsProcess.kill('SIGKILL');
    currentTtsProcess = null;
  }

  // Kill recording process if running
  if (currentRecProcess) {
    currentRecProcess.kill('SIGKILL');
    currentRecProcess = null;
  }

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

      await activeSession();
      if (!isAppRunning) break;

      recordSuccess(); // Session completed successfully
    } catch (err) {
      if (isAppRunning) {
        const shouldRestart = recordError(err);
        if (shouldRestart) {
          console.log('🔄 Attempting recovery restart...');
          await restart();
          return; // Exit this start() call, restart() will call start() again
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
    await stop();
    process.exit(0);
  });

  process.on('SIGTERM', async () => {
    await stop();
    process.exit(0);
  });

  await start();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
