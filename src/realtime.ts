import 'dotenv/config';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import path from 'node:path';
import OpenAI from 'openai';
import { OpenAIRealtimeWS } from 'openai/realtime/ws';

import { runPreflightChecks, cleanupZombieProcesses } from './cleanup.js';
import { getCachedAudio, cacheAudio } from './tts-cache.js';

/**
 * Winterfresh — Realtime (speech-to-speech) variant.
 *
 * Architecture goal: keep "wake whenever" and "shut down whenever" exactly as
 * reliable as the classic pipeline (local, offline Vosk), but replace the
 * serial transcribe -> chat -> TTS hop with a single full-duplex OpenAI
 * Realtime session for a fluid conversation while awake.
 *
 *   wake.py (Vosk, local)  --WAKE-->  Realtime WS session
 *        ^                                 |
 *        |  shutdown_listener.py (Vosk) ---/  (SHUTDOWN / idle / error)
 *
 * The EMEET speakerphone does hardware echo cancellation, so the open mic does
 * not hear the model's own voice — full duplex + barge-in works without AEC
 * code here.
 */

const IS_LINUX = process.platform === 'linux';
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const ASSISTANT_NAME = process.env.ASSISTANT_NAME ?? 'Winter fresh';
const REALTIME_MODEL = process.env.REALTIME_MODEL ?? 'gpt-realtime-2';
// marin / cedar are the highest quality GA voices.
const REALTIME_VOICE = process.env.REALTIME_VOICE ?? 'marin';
// semantic_vad uses a turn-detection model, so static / non-speech noise
// doesn't get treated as an interruption (server_vad is pure energy and
// false-triggers on noise). Default to semantic; REALTIME_VAD=server opts out.
const REALTIME_VAD =
  process.env.REALTIME_VAD === 'server' ? 'server_vad' : 'semantic_vad';
// semantic_vad: lower eagerness = waits longer / less trigger-happy.
const REALTIME_VAD_EAGERNESS = (process.env.REALTIME_VAD_EAGERNESS ??
  'auto') as 'low' | 'medium' | 'high' | 'auto';
// server_vad: raise threshold (0..1, default 0.5) to ignore quieter noise.
const REALTIME_VAD_THRESHOLD = Number(
  process.env.REALTIME_VAD_THRESHOLD ?? '0.5',
);
const REALTIME_VAD_SILENCE_MS = Number(
  process.env.REALTIME_VAD_SILENCE_MS ?? '500',
);
const turnDetection =
  REALTIME_VAD === 'semantic_vad'
    ? ({
        type: 'semantic_vad',
        create_response: true,
        eagerness: REALTIME_VAD_EAGERNESS,
      } as const)
    : ({
        type: 'server_vad',
        create_response: true,
        threshold: REALTIME_VAD_THRESHOLD,
        silence_duration_ms: REALTIME_VAD_SILENCE_MS,
      } as const);

const DEFAULT_RULES = [
  'Keep replies short and conversational — usually one sentence.',
  "Speak naturally and don't over-explain.",
  'Only ask a brief follow-up question when you genuinely need clarification.',
].join(' ');
const ASSISTANT_RULES = process.env.ASSISTANT_RULES ?? DEFAULT_RULES;
const INSTRUCTIONS = [
  `You are ${ASSISTANT_NAME}, a warm, concise voice assistant.`,
  ASSISTANT_RULES,
].join(' ');

// Deterministic spoken bookends so start/stop are always recognizable.
const GREETING = process.env.REALTIME_GREETING ?? 'Hi, what’s up?';
const GOODBYE = process.env.REALTIME_GOODBYE ?? 'Okay, see ya next time.';
// Voice for the locally-played goodbye (OpenAI tts-1; cached after first use).
const GOODBYE_VOICE = process.env.REALTIME_GOODBYE_VOICE ?? 'alloy';
const sayExactly = (line: string) =>
  `Say exactly and only this, then stop. Do not add anything: "${line}"`;

// Realtime PCM is fixed at 24kHz mono, 16-bit, little-endian.
const RT_RATE = 24000;

// Mic capture (mirrors app.ts device handling).
// Pi playback goes to the USB speakerphone via aplay, same as volume.py.
const ALSA_PLAY_DEVICE = process.env.ALSA_PLAY_DEVICE ?? 'plughw:2,0';
const LINUX_ARECORD_DEVICE = process.env.ARECORD_DEVICE ?? 'mic_share';
const MAC_GAIN_DB = 6;

// Close the session after this much inactivity to bound cost (always-on device).
const IDLE_TIMEOUT_MS = Number(process.env.REALTIME_IDLE_MS ?? '7000');

let isAppRunning = false;
// Hard guard: never allow two Realtime sessions at once.
let sessionActive = false;

function ms(n: number) {
  return `${Math.round(n)}ms`;
}

// ---------------------------------------------------------------------------
// Local wake word (reuse the proven offline Vosk listener)
// ---------------------------------------------------------------------------
function waitForWake(): Promise<void> {
  return new Promise((resolve, reject) => {
    const pythonPath = path.join(process.cwd(), '.venv', 'bin', 'python');
    const wakePath = path.join(process.cwd(), 'wake.py');
    const proc = spawn(pythonPath, ['-u', wakePath], {
      stdio: ['inherit', 'pipe', 'pipe'],
    });

    let settled = false;
    proc.stdout?.on('data', (d: Buffer) => {
      const text = d.toString();
      process.stdout.write(text);
      if (!settled && text.includes('WAKE')) {
        settled = true;
        proc.kill('SIGTERM');
        resolve();
      }
    });
    proc.stderr?.on('data', (d: Buffer) => process.stderr.write(d));
    proc.on('error', (err) => {
      if (!settled) {
        settled = true;
        reject(err);
      }
    });
    proc.on('close', (code) => {
      if (settled) return;
      settled = true;
      if (code === 0 || !isAppRunning) resolve();
      else reject(new Error(`wake.py exited with code ${code}`));
    });
  });
}

// ---------------------------------------------------------------------------
// Local shutdown phrase listener (reuse the proven offline Vosk listener).
// Runs concurrently with the Realtime session so "stop / go to sleep" works
// instantly and offline, independent of the model.
// ---------------------------------------------------------------------------
function startShutdownListener(onShutdown: () => void): () => void {
  const pythonPath = path.join(process.cwd(), '.venv', 'bin', 'python');
  const listenerPath = path.join(process.cwd(), 'shutdown_listener.py');
  let stopped = false;
  let proc: ReturnType<typeof spawn> | null = null;

  const spawnProc = () => {
    proc = spawn(pythonPath, ['-u', listenerPath], {
      stdio: ['inherit', 'pipe', 'pipe'],
    });

    proc.stdout?.on('data', (d: Buffer) => {
      const text = d.toString().trim();
      if (text && !text.includes('SHUTDOWN')) {
        process.stdout.write(`[shutdown] ${text}\n`);
      }
      if (text.includes('SHUTDOWN')) {
        console.log('\n🛑 Shutdown phrase detected');
        onShutdown();
      }
    });
    proc.stderr?.on('data', (d: Buffer) => {
      const m = d.toString().trim();
      if (m) console.error(`[shutdown:err] ${m}`);
    });
    proc.on('error', (err) => {
      console.error('[shutdown] spawn error:', err);
    });
    // STOP must never be silently off during a session — relaunch if it dies.
    proc.on('close', (code) => {
      if (stopped) return;
      console.warn(`[shutdown] listener exited (code ${code}), restarting...`);
      setTimeout(() => {
        if (!stopped) spawnProc();
      }, 300);
    });
  };

  spawnProc();
  console.log('🎧 STOP listener armed');

  return () => {
    stopped = true;
    try {
      proc?.kill('SIGTERM');
    } catch {}
  };
}

// ---------------------------------------------------------------------------
// Audio I/O processes
// ---------------------------------------------------------------------------
function startMic(): ReturnType<typeof spawn> {
  if (IS_LINUX) {
    // No sox: ALSA's `plug` plugin resamples/downmixes the shared mic_share
    // (dsnoop) device to exactly 24kHz mono raw, so arecord emits the format
    // Realtime needs directly. Spawned like shutdown_listener.py's arecord,
    // which is the proven-working primitive on the Pi.
    return spawn(
      'arecord',
      [
        '-q',
        '-D', `plug:${LINUX_ARECORD_DEVICE}`,
        '-f', 'S16_LE',
        '-c', '1',
        '-r', String(RT_RATE),
        '-t', 'raw',
      ],
      { stdio: ['ignore', 'pipe', 'pipe'] },
    );
  }
  // macOS: rec directly at 24kHz mono raw.
  return spawn(
    'rec',
    [
      '-q',
      '-c',
      '1',
      '-r',
      String(RT_RATE),
      '-b',
      '16',
      '-t',
      'raw',
      '-',
      'gain',
      String(MAC_GAIN_DB),
    ],
    { stdio: ['ignore', 'pipe', 'inherit'] },
  );
}

function startPlayer(): ReturnType<typeof spawn> {
  // Stream raw PCM16 24kHz mono from stdin to the speaker. On the Pi, sox's
  // default sink doesn't reach the USB speakerphone — use aplay on the same
  // ALSA device volume.py uses, otherwise no audio is heard.
  const p = IS_LINUX
    ? spawn(
        'aplay',
        [
          '-q',
          '-D',
          ALSA_PLAY_DEVICE,
          '-t',
          'raw',
          '-f',
          'S16_LE',
          '-r',
          String(RT_RATE),
          '-c',
          '1',
          // Small buffer (~150ms) so playback latency stays low and any
          // residual after a barge-in is short. Big enough to ride out
          // delta bursts without underrunning.
          '--buffer-time=150000',
          '--period-time=30000',
          '-',
        ],
        { stdio: ['pipe', 'inherit', 'inherit'] },
      )
    : spawn(
        'play',
        [
          '-q',
          '-t',
          'raw',
          '-r',
          String(RT_RATE),
          '-e',
          'signed-integer',
          '-b',
          '16',
          '-c',
          '1',
          '-',
        ],
        { stdio: ['pipe', 'inherit', 'inherit'] },
      );
  p.stdin?.on('error', () => {});
  p.on('error', (err) => console.error('🔇 player spawn error:', err));
  p.on('exit', (code, sig) => {
    if (code) console.error(`🔇 player exited code=${code} sig=${sig}`);
  });
  console.log(
    `🔊 player started (${IS_LINUX ? `aplay ${ALSA_PLAY_DEVICE}` : 'play'})`,
  );
  return p;
}

function killProc(p: ReturnType<typeof spawn> | null) {
  if (!p) return;
  try {
    if (IS_LINUX && p.pid !== undefined && p.spawnargs[0] === 'bash') {
      process.kill(-p.pid, 'SIGKILL'); // kill the arecord|sox group
    } else {
      p.kill('SIGKILL');
    }
  } catch {}
}

// ---------------------------------------------------------------------------
// One Realtime conversation session. Resolves when it ends (shutdown / idle /
// error / disconnect); the outer loop then returns to the wake word.
// ---------------------------------------------------------------------------
async function runSession(): Promise<void> {
  // Hard guard: never allow two concurrent sessions.
  if (sessionActive) {
    console.warn('⚠️ Session already active, ignoring duplicate start');
    return;
  }
  sessionActive = true;

  // STOP first: armed before we even connect, so the shutdown phrase covers
  // the entire session lifetime (and self-heals if it crashes).
  let endSession: ((why: string) => void) | null = null;
  const stopShutdown = startShutdownListener(() =>
    endSession?.('shutdown phrase'),
  );

  console.log(
    `\n🔌 Connecting Realtime (${REALTIME_MODEL}, voice=${REALTIME_VOICE})...`,
  );
  const rt = new OpenAIRealtimeWS({ model: REALTIME_MODEL }, client);

  let mic: ReturnType<typeof spawn> | null = null;
  // Opened lazily on the first audio delta so it doesn't hold the single-open
  // EMEET device during the wake chime (which would fail "device busy").
  let player: ReturnType<typeof spawn> | null = null;
  const ensurePlayer = () => {
    if (!player) player = startPlayer();
    return player;
  };
  let idleTimer: NodeJS.Timeout | null = null;
  let ended = false;

  // In-progress assistant audio, tracked so we can truncate on barge-in.
  let activeResponse = false;
  let curItemId: string | null = null;
  let curContentIndex = 0;
  let playedBytes = 0;
  // After a barge-in we stop feeding the player the cancelled response's
  // tail, but we do NOT kill the player — respawning aplay on the Pi's
  // single-open hw device collides and kills all further audio.
  let suppressAudio = false;

  let closing = false; // a goodbye is being spoken before teardown

  return new Promise<void>((resolve) => {
    // Hard teardown. Used directly when we can't speak (socket gone, error).
    const end = (why: string) => {
      if (ended) return;
      ended = true;
      sessionActive = false;
      console.log(`\n💤 Session ended (${why})`);
      if (idleTimer) clearTimeout(idleTimer);
      stopShutdown();
      killProc(mic);
      mic = null;
      try {
        player?.stdin?.end();
      } catch {}
      setTimeout(() => killProc(player), 1500);
      player = null;
      try {
        rt.close();
      } catch {}
      resolve();
    };

    // Shut down with a spoken confirmation. The goodbye is played LOCALLY
    // (cached/synthesized WAV via `play`), fully decoupled from the Realtime
    // socket — no cancel/response races, works even if the model is mid-reply.
    const goodbyeAndEnd = async (why: string) => {
      if (ended || closing) return;
      closing = true;
      console.log(`\n👋 Shutting down (${why}) — saying goodbye`);
      if (idleTimer) clearTimeout(idleTimer);
      stopShutdown(); // already shutting down; don't let STOP re-trigger
      killProc(mic); // stop listening so the user can't keep talking
      mic = null;
      // Tear the model down right now; the goodbye no longer needs it.
      killProc(player);
      player = null;
      try {
        rt.close();
      } catch {}

      try {
        let file = await getCachedAudio(GOODBYE, GOODBYE_VOICE);
        if (!file) {
          const audio = await client.audio.speech.create({
            model: 'tts-1',
            voice: GOODBYE_VOICE,
            input: GOODBYE,
            response_format: 'wav',
          });
          const buf = Buffer.from(await audio.arrayBuffer());
          file = await cacheAudio(GOODBYE, GOODBYE_VOICE, buf);
        }
        const p = IS_LINUX
          ? spawn('aplay', ['-q', '-D', ALSA_PLAY_DEVICE, file], {
              stdio: ['ignore', 'inherit', 'inherit'],
            })
          : spawn('play', ['-q', file], {
              stdio: ['ignore', 'inherit', 'inherit'],
            });
        await Promise.race([
          once(p, 'close'),
          once(p, 'exit'),
          new Promise((r) => setTimeout(r, 8000)),
        ]);
      } catch (err) {
        console.warn('goodbye playback failed:', err);
      }

      end(why);
    };
    endSession = goodbyeAndEnd;

    // Idle timeout like app.ts: it only counts down while we're waiting for
    // the user. Any activity (user speaking, model replying) clears it; it is
    // re-armed when we go back to waiting. It only fires after a full
    // IDLE_TIMEOUT_MS with no activity at all.
    const clearIdle = () => {
      if (idleTimer) {
        clearTimeout(idleTimer);
        idleTimer = null;
      }
    };
    const armIdle = () => {
      if (closing || ended) return;
      clearIdle();
      idleTimer = setTimeout(
        () => goodbyeAndEnd('idle timeout'),
        IDLE_TIMEOUT_MS,
      );
    };

    // --- Configure the session once the socket is open ---
    rt.socket.on('open', () => {
      console.log('🟢 Realtime socket open');
      rt.send({
        type: 'session.update',
        session: {
          type: 'realtime',
          instructions: INSTRUCTIONS,
          output_modalities: ['audio'],
          // Backstop against an occasional runaway monologue (cost + brevity).
          max_output_tokens: 512,
          audio: {
            input: {
              format: { type: 'audio/pcm', rate: RT_RATE },
              // Conference mic on a Pi -> far_field noise reduction.
              noise_reduction: { type: 'far_field' },
              // Cheap async transcript, useful for logging only.
              transcription: { model: 'whisper-1' },
              turn_detection: turnDetection,
            },
            output: {
              format: { type: 'audio/pcm', rate: RT_RATE },
              voice: REALTIME_VOICE,
            },
          },
        },
      });
    });

    rt.on('session.created', () => console.log('✅ Realtime session created'));

    rt.on('response.created', () => {
      activeResponse = true;
      suppressAudio = false; // a new reply has begun; play it
      console.log('🧠 response.created');
      clearIdle(); // model is active
    });
    rt.on('input_audio_buffer.speech_stopped', () => {
      console.log('🗣️ speech_stopped (your turn ended)');
      // Waiting on the model now; re-arm as a fallback in case no reply comes.
      armIdle();
    });

    rt.on('session.updated', async () => {
      if (mic) return; // configure once
      console.log('🎤 Streaming mic — speak now (barge-in supported)');
      // No wake chime here: it's a separate sox `play` that fights the
      // single-open EMEET device right before the greeting. The spoken
      // greeting is the readiness signal instead.
      // Deterministic greeting (otherwise the model ad-libs random openers).
      rt.send({
        type: 'response.create',
        response: {
          instructions: sayExactly(GREETING),
          output_modalities: ['audio'],
        },
      });
      armIdle(); // safety until the greeting response starts

      mic = startMic();
      let micBytes = 0;
      mic.stdout?.on('data', (buf: Buffer) => {
        const first = micBytes === 0;
        micBytes += buf.length;
        if (first) console.log('🎤 mic producing audio');
        rt.send({
          type: 'input_audio_buffer.append',
          audio: buf.toString('base64'),
        });
      });
      mic.stderr?.on('data', (d: Buffer) => {
        const m = d.toString().trim();
        if (m) console.error(`[mic:err] ${m}`);
      });
      // Heartbeat: prove whether the pipeline is actually producing samples.
      const micWatch = setInterval(() => {
        console.log(`🎤 mic bytes so far: ${micBytes}`);
      }, 2000);
      mic.on('error', (err) => console.error('[mic] spawn error:', err));
      mic.on('close', (code) => {
        clearInterval(micWatch);
        console.log(`🎤 mic process closed (code=${code}, bytes=${micBytes})`);
        // Don't hard-end while a goodbye is being spoken — goodbyeAndEnd
        // intentionally kills the mic first.
        if (!ended && !closing && isAppRunning) end('mic stopped');
      });
    });

    // Model speech -> speaker. Track item + bytes for barge-in truncation.
    rt.on('response.output_audio.delta', (e) => {
      if (e.item_id !== curItemId) {
        curItemId = e.item_id;
        curContentIndex = e.content_index;
        playedBytes = 0;
        console.log('🔊 first audio delta for', e.item_id);
      }
      activeResponse = true;
      if (suppressAudio) return; // tail of a cancelled (barged-in) response
      const p = ensurePlayer();
      if (p?.stdin && !p.stdin.destroyed) {
        const chunk = Buffer.from(e.delta, 'base64');
        playedBytes += chunk.length;
        p.stdin.write(chunk);
      }
    });

    // Barge-in: user started talking while the model was speaking.
    rt.on('input_audio_buffer.speech_started', () => {
      console.log('🎙️ speech_started (mic heard you)');
      clearIdle(); // user is active
      if (activeResponse) {
        // Only cancel a response that actually exists (avoids noisy errors).
        rt.send({ type: 'response.cancel' });
        if (curItemId) {
          // 24kHz mono 16-bit PCM => 48 bytes per ms. Tell the server how
          // much the user actually heard so its context stays truthful.
          rt.send({
            type: 'conversation.item.truncate',
            item_id: curItemId,
            content_index: curContentIndex,
            audio_end_ms: Math.floor(playedBytes / 48),
          });
        }
        activeResponse = false;
      }
      // Flush instantly: kill the player so the audio already queued in
      // aplay's ALSA buffer stops *now* (otherwise it keeps talking ~1-2s).
      // We deliberately do NOT respawn here — ensurePlayer() lazily makes a
      // fresh one on the next reply's first delta, which is seconds away
      // (you finish talking -> server turn-end -> model responds), giving
      // the Pi device ample time to release. The earlier breakage was
      // kill-then-IMMEDIATELY-respawn; deferred respawn avoids the collision.
      suppressAudio = true;
      killProc(player);
      player = null;
      playedBytes = 0;
      curItemId = null;
    });

    rt.on('response.done', () => {
      activeResponse = false;
      console.log('✅ response.done');
      armIdle(); // back to waiting for the user
    });

    rt.on('conversation.item.input_audio_transcription.completed', (e) => {
      if (e.transcript) console.log('You:', e.transcript.trim());
    });
    rt.on('response.output_audio_transcript.done', (e) => {
      if (e.transcript) console.log(`${ASSISTANT_NAME}:`, e.transcript.trim());
    });

    rt.on('error', (err) => {
      console.error('❌ Realtime error:', (err as Error)?.message ?? err);
    });
    rt.socket.on('close', () => end('socket closed'));
  });
}

// ---------------------------------------------------------------------------
// Supervisor loop: wake -> session -> back to wake.
// ---------------------------------------------------------------------------
async function start() {
  isAppRunning = true;
  while (isAppRunning) {
    try {
      console.log('\n🎤 Listening for wake word (local Vosk)...');
      await waitForWake();
      if (!isAppRunning) break;

      const t0 = performance.now();
      await runSession();
      console.log(`⏱️ session duration=${ms(performance.now() - t0)}`);
    } catch (err) {
      console.error('Loop error:', err);
      await new Promise((r) => setTimeout(r, 1000));
    }
  }
}

async function main() {
  console.log(`🌨️  ${ASSISTANT_NAME} (realtime) starting...`);
  if (!process.env.OPENAI_API_KEY) {
    console.error('❌ OPENAI_API_KEY is required.');
    process.exit(1);
  }
  if (!runPreflightChecks()) {
    console.error('❌ Pre-flight checks failed. Exiting.');
    process.exit(1);
  }

  const shutdown = async () => {
    console.log(`\n🛑 Shutting down ${ASSISTANT_NAME} (realtime)...`);
    isAppRunning = false;
    cleanupZombieProcesses();
    process.exit(0);
  };
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);

  // Pre-warm the goodbye clip so the first shutdown is also instant + offline.
  void (async () => {
    try {
      if (await getCachedAudio(GOODBYE, GOODBYE_VOICE)) return;
      const audio = await client.audio.speech.create({
        model: 'tts-1',
        voice: GOODBYE_VOICE,
        input: GOODBYE,
        response_format: 'wav',
      });
      await cacheAudio(
        GOODBYE,
        GOODBYE_VOICE,
        Buffer.from(await audio.arrayBuffer()),
      );
      console.log('🗣️  Goodbye clip cached');
    } catch (err) {
      console.warn('goodbye pre-warm failed (will synth on demand):', err);
    }
  })();

  await start();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
