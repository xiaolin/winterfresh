import fs from 'node:fs/promises';

function sleep(ms: number) {
  return new Promise<void>((resolve) => setTimeout(resolve, ms));
}

/**
 * Wait until a file's size stops changing for a few consecutive checks.
 * This helps avoid sending partially-written WAVs to transcription.
 */
export async function waitForStableFileSize(
  filePath: string,
  {
    intervalMs = 80,
    stableChecks = 3,
    timeoutMs = 1500,
  }: { intervalMs?: number; stableChecks?: number; timeoutMs?: number } = {},
): Promise<void> {
  const start = Date.now();
  let lastSize = -1;
  let stableCount = 0;

  while (Date.now() - start < timeoutMs) {
    const st = await fs.stat(filePath).catch(() => null);
    const size = st?.size ?? 0;

    if (size > 0 && size === lastSize) {
      stableCount++;
      if (stableCount >= stableChecks) return;
    } else {
      stableCount = 0;
      lastSize = size;
    }

    await sleep(intervalMs);
  }
}

/**
 * Minimal RIFF/WAVE validation: checks RIFF/WAVE header and presence of fmt/data chunks.
 * Keeps false positives low; intended to catch truncation/corruption before API call.
 */
export function isLikelyValidWav(buf: Buffer): boolean {
  if (buf.length < 44) return false;
  if (buf.toString('ascii', 0, 4) !== 'RIFF') return false;
  if (buf.toString('ascii', 8, 12) !== 'WAVE') return false;

  let offset = 12;
  let sawFmt = false;
  let sawData = false;

  while (offset + 8 <= buf.length) {
    const id = buf.toString('ascii', offset, offset + 4);
    const size = buf.readUInt32LE(offset + 4);
    const dataStart = offset + 8;
    const dataEnd = dataStart + size;

    if (dataEnd > buf.length) return false;

    if (id === 'fmt ') {
      if (size < 16) return false;

      const audioFormat = buf.readUInt16LE(dataStart + 0); // 1=PCM, 3=float, 65534=extensible
      const channels = buf.readUInt16LE(dataStart + 2);
      const sampleRate = buf.readUInt32LE(dataStart + 4);
      const bitsPerSample = buf.readUInt16LE(dataStart + 14);

      if (![1, 3, 65534].includes(audioFormat)) return false;
      if (channels < 1 || channels > 2) return false;
      if (sampleRate < 8000 || sampleRate > 96000) return false;
      if (![8, 16, 24, 32].includes(bitsPerSample)) return false;

      sawFmt = true;
    } else if (id === 'data') {
      if (size <= 0) return false;
      sawData = true;
    }

    // chunks are word-aligned
    offset = dataEnd + (size % 2);

    if (sawFmt && sawData) return true;
  }

  return false;
}
