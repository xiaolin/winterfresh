import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';

const CACHE_DIR = path.join(process.cwd(), 'audio_cache');

/**
 * Normalize text for consistent cache keys.
 */
function normalizeForCache(text: string): string {
  return text
    .toLowerCase()
    .replace(/['']/g, "'") // normalize apostrophes
    .replace(/[""]/g, '"') // normalize quotes
    .trim();
}

/**
 * Get a cache key for a TTS phrase (hash of text + voice).
 */
function getCacheKey(text: string, voice: string): string {
  const normalized = normalizeForCache(text);
  const hash = crypto
    .createHash('sha256')
    .update(`${voice}:${normalized}`)
    .digest('hex')
    .slice(0, 16);
  return `${hash}.wav`;
}

/**
 * Get the cached audio file path, or null if not cached.
 */
export async function getCachedAudio(
  text: string,
  voice: string,
): Promise<string | null> {
  const cacheFile = path.join(CACHE_DIR, getCacheKey(text, voice));
  try {
    await fs.access(cacheFile);
    return cacheFile;
  } catch {
    return null;
  }
}

/**
 * Save audio data to cache and return the file path.
 */
export async function cacheAudio(
  text: string,
  voice: string,
  audioData: Buffer,
): Promise<string> {
  await fs.mkdir(CACHE_DIR, { recursive: true });
  const cacheFile = path.join(CACHE_DIR, getCacheKey(text, voice));
  await fs.writeFile(cacheFile, audioData);
  return cacheFile;
}

/**
 * Common phrases that should be pre-cached or always cached.
 */
export const CACHED_PHRASES = [
  'Alright, going back to sleep.',
  "What's up?",
  'Welcome back! How can I assist you further?',
];
