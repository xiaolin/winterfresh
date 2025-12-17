import { execSync } from 'node:child_process';
import os from 'node:os';

// Recovery settings
const MAX_CONSECUTIVE_ERRORS = 5;

// Error tracking state
let consecutiveErrors = 0;

// Detect platform
const IS_MAC = os.platform() === 'darwin';

/**
 * Check if audio devices are available
 */
export function checkAudioDevices(): boolean {
  try {
    execSync('which rec', { stdio: 'pipe' });
    execSync('which play', { stdio: 'pipe' });
    return true;
  } catch {
    console.error('❌ Audio tools (sox) not available. Install with:');
    if (IS_MAC) {
      console.error('   brew install sox');
    } else {
      console.error('   sudo apt-get install sox libsox-fmt-all');
    }
    return false;
  }
}

/**
 * Clean up any zombie processes
 */
export function cleanupZombieProcesses() {
  try {
    execSync('pkill -f "python.*wake.py" 2>/dev/null || true', {
      stdio: 'pipe',
      shell: '/bin/bash',
    });
  } catch {
    // Ignore
  }
}

/**
 * Record success to reset error counter
 */
export function recordSuccess() {
  consecutiveErrors = 0;
}

/**
 * Record error and check if we should restart
 */
export function recordError(error: Error | unknown): boolean {
  consecutiveErrors++;
  console.error(
    `❌ Error (${consecutiveErrors}/${MAX_CONSECUTIVE_ERRORS}):`,
    error,
  );

  if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
    console.error('🔄 Too many errors, will attempt full restart...');
    return true;
  }

  return false;
}

/**
 * Reset error counter
 */
export function resetErrorCounter() {
  consecutiveErrors = 0;
}

/**
 * Run pre-flight checks
 */
export function runPreflightChecks(): boolean {
  console.log('🔍 Pre-flight checks...');

  if (!process.env.OPENAI_API_KEY) {
    console.error('❌ OPENAI_API_KEY not set in .env');
    return false;
  }

  if (!checkAudioDevices()) {
    return false;
  }

  console.log('✅ Ready\n');
  return true;
}
