// ============================================================================
// RainAI Neural Audio Engine Initialization & Autoplay Unlocking
// ============================================================================
window.__rainAudioContext = new (window.AudioContext || window.webkitAudioContext)();
window.__rainEngine = null;

// ============================================================================
// Chunked Fetch API & Live ETA Progress Bar Engine
// ============================================================================
async function fetchWithChunkedProgress(url, estimatedTotalBytes = 15500000, stepLabel = "Downloading Engine...") {
  const stepEl = document.getElementById('progress_step');
  const fillEl = document.getElementById('progress_fill');
  const pctEl = document.getElementById('progress_pct');
  const bytesEl = document.getElementById('progress_bytes');
  const speedEl = document.getElementById('progress_speed');
  const etaEl = document.getElementById('progress_eta');

  if (stepEl) stepEl.textContent = stepLabel;

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`);
  }

  const contentLength = response.headers.get('content-length');
  const totalBytes = contentLength ? parseInt(contentLength, 10) : estimatedTotalBytes;

  if (!response.body) {
    const buf = await response.arrayBuffer();
    return buf;
  }

  const reader = response.body.getReader();
  const chunks = [];
  let receivedBytes = 0;
  const startTime = performance.now();
  let lastSpeedUpdate = startTime;
  let lastSpeedBytes = 0;
  let rollingSpeedBps = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    chunks.push(value);
    receivedBytes += value.length;

    const now = performance.now();
    const elapsedSecSinceUpdate = (now - lastSpeedUpdate) / 1000.0;

    if (elapsedSecSinceUpdate >= 0.25 || receivedBytes === totalBytes) {
      const bytesSinceLast = receivedBytes - lastSpeedBytes;
      const currentSpeed = bytesSinceLast / Math.max(0.001, elapsedSecSinceUpdate);
      rollingSpeedBps = rollingSpeedBps === 0 ? currentSpeed : (rollingSpeedBps * 0.7 + currentSpeed * 0.3);

      lastSpeedUpdate = now;
      lastSpeedBytes = receivedBytes;

      const pct = Math.min(100, Math.round((receivedBytes / totalBytes) * 100));
      const remainingBytes = Math.max(0, totalBytes - receivedBytes);
      const etaSeconds = rollingSpeedBps > 1000 ? Math.ceil(remainingBytes / rollingSpeedBps) : 0;

      if (fillEl) fillEl.style.width = `${pct}%`;
      if (pctEl) pctEl.textContent = `${pct}%`;
      if (bytesEl) {
        bytesEl.textContent = `${(receivedBytes / 1048576).toFixed(1)} MB / ${(totalBytes / 1048576).toFixed(1)} MB`;
      }
      if (speedEl) {
        speedEl.textContent = `Speed: ${(rollingSpeedBps / 1048576).toFixed(2)} MB/s`;
      }
      if (etaEl) {
        etaEl.textContent = etaSeconds > 0 ? `ETA: ~${etaSeconds}s` : 'ETA: finalizing...';
      }
    }
  }

  // Concatenate all chunks into a unified Uint8Array
  const combined = new Uint8Array(receivedBytes);
  let position = 0;
  for (const chunk of chunks) {
    combined.set(chunk, position);
    position += chunk.length;
  }

  if (fillEl) fillEl.style.width = '100%';
  if (pctEl) pctEl.textContent = '100%';
  if (stepEl) stepEl.textContent = 'Engine Ready!';

  return combined.buffer;
}

async function initializeAudioEngine() {
  if (window.__rainEngine) return window.__rainEngine;

  const audioCtx = window.__rainAudioContext;
  
  // 1. Load the AudioWorklet module
  await audioCtx.audioWorklet.addModule('inference_worklet.js');
  
  // 2. Fetch and compile the WebAssembly binary dynamically using chunked progress tracking
  const wasmBytes = await fetchWithChunkedProgress('./pkg/web_bg.wasm', 15500000, "Downloading Neural WASM Engine...");
  const wasmModule = await WebAssembly.compile(wasmBytes);
  
  // 3. Create the Inference Node (0 inputs, 1 output with 4 channels)
  const inferenceNode = new AudioWorkletNode(audioCtx, 'rain-inference-processor', {
      numberOfInputs: 0,
      numberOfOutputs: 1,
      outputChannelCount: [4]
  });
  
  // 4. Initialize lock-free memory mapping (554 f32s = 2216 bytes)
  const sharedBuffer = new SharedArrayBuffer(554 * 4);
  const telemetryArray = new Float32Array(sharedBuffer);
  
  // 5. Send initialization payloads to the Worklet
  inferenceNode.port.postMessage({ type: 'INIT_WASM', payload: { wasmModule } });
  inferenceNode.port.postMessage({ type: 'SET_SHARED_BUFFER', payload: { sharedBuffer } });
  
  // 6. Connect the 4-channel FOA output to the destination (or a Binaural downmixer)
  inferenceNode.connect(audioCtx.destination);
  
  window.__rainEngine = { audioCtx, inferenceNode, telemetryArray };
  console.log("RainAI Audio Engine Initialized");

  // Dismiss loading overlay smoothly
  hideLoadingOverlay();
  
  return window.__rainEngine;
}

(function () {
  const unlockAudio = async () => {
    if (window.__rainAudioContext) {
      if (window.__rainAudioContext.state === 'suspended') {
        try {
          await window.__rainAudioContext.resume();
          // Mobile Safari AudioContext unlock: play 1 frame of silence
          const buffer = window.__rainAudioContext.createBuffer(1, 1, 22050);
          const source = window.__rainAudioContext.createBufferSource();
          source.buffer = buffer;
          source.connect(window.__rainAudioContext.destination);
          source.start(0);
        } catch (_) {}
      }
    }
    // Initialize the WASM engine on user interaction if not yet started
    if (!window.__rainEngine) {
      await initializeAudioEngine().catch(e => console.error("Audio Engine Init Failed:", e));
    }
  };
  ['click', 'touchstart', 'touchend', 'pointerdown', 'keydown'].forEach((evt) => {
    window.addEventListener(evt, unlockAudio, { passive: true });
  });
})();

// ============================================================================
// Service Worker Registration for Offline PWA Capabilities
// ============================================================================
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker
      .register('./sw.js')
      .then((reg) => console.log('ServiceWorker active on scope:', reg.scope))
      .catch((err) => console.log('ServiceWorker registration deferred:', err));
  });
}

// ============================================================================
// Progressive Web App (PWA) Install Prompt Engine
// ============================================================================
window.__pwaInstallPrompt = null;
window.__pwaInstallAvailable = false;
window.__pwaInstalled = window.matchMedia('(display-mode: standalone)').matches || window.navigator.standalone === true;

window.addEventListener('beforeinstallprompt', (e) => {
  e.preventDefault();
  window.__pwaInstallPrompt = e;
  window.__pwaInstallAvailable = true;
  window.dispatchEvent(new CustomEvent('pwa-install-available'));
  console.log('PWA installation prompt captured and ready');
});

window.addEventListener('appinstalled', () => {
  window.__pwaInstallPrompt = null;
  window.__pwaInstallAvailable = false;
  window.__pwaInstalled = true;
  window.dispatchEvent(new CustomEvent('pwa-installed'));
  console.log('PWA successfully installed by user');
  if (window.__requestPersistentStorage) {
    window.__requestPersistentStorage();
  }
});

window.__triggerPWAInstall = async function () {
  if (!window.__pwaInstallPrompt) {
    console.warn('PWA install prompt is not available at this time');
    return false;
  }
  try {
    window.__pwaInstallPrompt.prompt();
    const { outcome } = await window.__pwaInstallPrompt.userChoice;
    console.log(`User response to PWA install prompt: ${outcome}`);
    if (outcome === 'accepted') {
      window.__pwaInstallPrompt = null;
      window.__pwaInstallAvailable = false;
      return true;
    }
    return false;
  } catch (err) {
    console.error('Error triggering PWA install:', err);
    return false;
  }
};

// ============================================================================
// StorageManager Persistence API Bridge
// ============================================================================
window.__storagePersisted = false;

window.__checkStoragePersisted = async function () {
  if (navigator.storage && navigator.storage.persisted) {
    try {
      const persisted = await navigator.storage.persisted();
      window.__storagePersisted = persisted;
      return persisted;
    } catch (e) {
      console.warn('Failed to check storage persistence:', e);
      return false;
    }
  }
  return false;
};

window.__requestPersistentStorage = async function () {
  if (navigator.storage && navigator.storage.persist) {
    try {
      const granted = await navigator.storage.persist();
      window.__storagePersisted = granted;
      window.dispatchEvent(new CustomEvent('storage-persistence-changed', { detail: { granted } }));
      console.log(`Persistent storage requested. Granted: ${granted}`);
      return granted;
    } catch (e) {
      console.error('Error requesting persistent storage:', e);
      return false;
    }
  }
  return false;
};

window.__getStorageEstimate = async function () {
  if (navigator.storage && navigator.storage.estimate) {
    try {
      const estimate = await navigator.storage.estimate();
      return JSON.stringify({
        usage: estimate.usage || 0,
        quota: estimate.quota || 0,
      });
    } catch (e) {
      return JSON.stringify({ usage: 0, quota: 0 });
    }
  }
  return JSON.stringify({ usage: 0, quota: 0 });
};

// Auto-check persistence status on load
window.addEventListener('load', () => {
  if (window.__checkStoragePersisted) {
    window.__checkStoragePersisted();
  }
});

// ============================================================================
// Robust IndexedDB Fallback / Multi-Tier Storage Engine
// ============================================================================
const IDB_DB_NAME = 'rainai_offline_store';
const IDB_STORE_NAME = 'rainai_kv';
const IDB_VERSION = 1;

function openIdbDatabase() {
  return new Promise((resolve, reject) => {
    if (!window.indexedDB) {
      reject(new Error('IndexedDB is not supported in this environment'));
      return;
    }
    const request = window.indexedDB.open(IDB_DB_NAME, IDB_VERSION);
    request.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains(IDB_STORE_NAME)) {
        db.createObjectStore(IDB_STORE_NAME);
      }
    };
    request.onsuccess = (e) => resolve(e.target.result);
    request.onerror = (e) => reject(e.target.error);
  });
}

window.__saveToIndexedDB = async function (key, value) {
  try {
    const db = await openIdbDatabase();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE_NAME, 'readwrite');
      const store = tx.objectStore(IDB_STORE_NAME);
      const req = store.put(value, key);
      req.onsuccess = () => resolve(true);
      req.onerror = (e) => reject(e.target.error);
    });
  } catch (err) {
    console.error('IndexedDB save error for key:', key, err);
    return false;
  }
};

window.__loadFromIndexedDB = async function (key) {
  try {
    const db = await openIdbDatabase();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE_NAME, 'readonly');
      const store = tx.objectStore(IDB_STORE_NAME);
      const req = store.get(key);
      req.onsuccess = (e) => resolve(e.target.result || null);
      req.onerror = (e) => reject(e.target.error);
    });
  } catch (err) {
    console.error('IndexedDB load error for key:', key, err);
    return null;
  }
};

window.__deleteFromIndexedDB = async function (key) {
  try {
    const db = await openIdbDatabase();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE_NAME, 'readwrite');
      const store = tx.objectStore(IDB_STORE_NAME);
      const req = store.delete(key);
      req.onsuccess = () => resolve(true);
      req.onerror = (e) => reject(e.target.error);
    });
  } catch (err) {
    return false;
  }
};

// ============================================================================
// Screen Wake Lock API (Prevents Mobile Sleep Throttling During Playback)
// ============================================================================
window.__wakeLockSentinel = null;
window.__wakeLockDesired = false;

window.__setWakeLock = async function (active) {
  window.__wakeLockDesired = active;
  if (!('wakeLock' in navigator)) {
    return false;
  }

  try {
    if (active) {
      if (!window.__wakeLockSentinel) {
        window.__wakeLockSentinel = await navigator.wakeLock.request('screen');
        window.__wakeLockSentinel.addEventListener('release', () => {
          window.__wakeLockSentinel = null;
          console.log('[RainAI] Screen Wake Lock released');
        });
        console.log('[RainAI] Screen Wake Lock acquired for continuous audio playback');
      }
      return true;
    } else {
      if (window.__wakeLockSentinel) {
        await window.__wakeLockSentinel.release();
        window.__wakeLockSentinel = null;
      }
      return true;
    }
  } catch (err) {
    console.warn('[RainAI] Screen Wake Lock error:', err);
    return false;
  }
};

// Re-acquire Wake Lock when tab becomes visible if playback was active
document.addEventListener('visibilitychange', async () => {
  if (document.visibilityState === 'visible' && window.__wakeLockDesired) {
    await window.__setWakeLock(true);
  }
});

// ============================================================================
// Hardware WebGPU vs CPU Acceleration Detection
// ============================================================================
window.__webgpuAvailable = false;

async function detectGpuAcceleration() {
  const badge = document.getElementById('hw_badge');
  if (!navigator.gpu) {
    if (badge) {
      badge.textContent = "⚡ CPU Mode (WebGPU unavailable)";
      badge.className = "badge badge-cpu";
    }
    window.__webgpuAvailable = false;
    return false;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      if (badge) {
        badge.textContent = "⚡ CPU Mode (No GPU adapter)";
        badge.className = "badge badge-cpu";
      }
      window.__webgpuAvailable = false;
      return false;
    }
    if (badge) {
      badge.textContent = "🚀 WebGPU Accelerated";
      badge.className = "badge badge-gpu";
    }
    window.__webgpuAvailable = true;
    return true;
  } catch (e) {
    if (badge) {
      badge.textContent = "⚡ CPU Mode (Fallback)";
      badge.className = "badge badge-cpu";
    }
    window.__webgpuAvailable = false;
    return false;
  }
}

// ============================================================================
// Loading Overlay Auto-Dismissal Engine
// ============================================================================
function hideLoadingOverlay() {
  const overlay = document.getElementById('loading_overlay');
  if (overlay && !overlay.classList.contains('fade_out')) {
    overlay.classList.add('fade_out');
    setTimeout(() => {
      overlay.style.display = 'none';
    }, 600);
  }
}

// Watch canvas render activity to automatically dismiss overlay
(function watchCanvasReady() {
  const canvas = document.getElementById('egui_canvas');
  if (!canvas) return;

  let checks = 0;
  const poll = setInterval(() => {
    checks++;
    if (canvas.width > 0 && canvas.height > 0) {
      // Allow egui one frame to paint
      setTimeout(hideLoadingOverlay, 300);
      clearInterval(poll);
    }
    if (checks > 120) {
      clearInterval(poll);
      hideLoadingOverlay();
    }
  }, 100);
})();

// ============================================================================
// Out-of-Memory (OOM) & Crash Resurrection Loop
// ============================================================================
window.__rainRecoveryAttempts = 0;

function setupOOMAndCrashRecovery() {
  const handlePanicOrOOM = (err) => {
    const errStr = String(err || '').toLowerCase();
    const isOomOrPanic =
      errStr.includes('out of memory') ||
      errStr.includes('memory access out of bounds') ||
      errStr.includes('oom') ||
      errStr.includes('unreachable') ||
      errStr.includes('panic') ||
      errStr.includes('allocation failed');

    if (isOomOrPanic) {
      console.warn('[RainAI] Intercepted runtime memory pressure or panic:', err);
      if (window.__rainRecoveryAttempts < 3) {
        window.__rainRecoveryAttempts++;
        const banner = document.getElementById('recovery_banner');
        const overlay = document.getElementById('loading_overlay');
        if (banner) {
          banner.textContent = `⚠ Memory limit reached. Gracefully recovering session with optimized preset (Attempt ${window.__rainRecoveryAttempts}/3)...`;
          banner.classList.remove('hidden');
        }
        if (overlay) {
          overlay.style.display = 'flex';
          overlay.classList.remove('fade_out');
        }

        // Write safe lightweight recovery state to localStorage
        try {
          const safeState = {
            rain: {
              intensity: 0.35,
              droplet_density: 0.25,
              quality_tier: 0,
              is_playing: true,
              master_volume: 0.5
            }
          };
          localStorage.setItem('rainai_app_state', JSON.stringify(safeState));
        } catch (_) {}

        // Reload after a short delay to let the browser release garbage
        setTimeout(() => {
          window.location.reload();
        }, 1200);
      }
    }
  };

  window.addEventListener('error', (e) => handlePanicOrOOM(e.error || e.message));
  window.addEventListener('unhandledrejection', (e) => handlePanicOrOOM(e.reason));
}

// ============================================================================
// Cloudflare Turnstile Helper Bridge
// ============================================================================
window.__turnstileWidgetId = null;

window.__getTurnstileToken = async function (sitekey = "1x00000000000000000000AA") {
  if (!window.turnstile) {
    console.warn('[RainAI] Turnstile API not loaded, checking ad-blockers');
    return null;
  }

  const container = document.getElementById('cf-turnstile-container');
  if (!container) return null;

  return new Promise((resolve) => {
    try {
      container.innerHTML = '';
      window.__turnstileWidgetId = window.turnstile.render('#cf-turnstile-container', {
        sitekey: sitekey,
        callback: (token) => {
          resolve(token);
        },
        'error-callback': () => {
          console.warn('[RainAI] Turnstile challenge execution deferred');
          resolve(null);
        },
      });
    } catch (e) {
      console.warn('[RainAI] Turnstile render exception:', e);
      resolve(null);
    }
  });
};

// Initial triggers on page load
window.addEventListener('DOMContentLoaded', () => {
  detectGpuAcceleration();
  setupOOMAndCrashRecovery();
});

