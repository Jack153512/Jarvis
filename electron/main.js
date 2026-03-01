const { app, BrowserWindow, ipcMain, session } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

// ----------------------------
// Constants & small utilities
// ----------------------------

const DEV_SERVER_PORT = 5173;
const DEV_SERVER_HOSTS = ['localhost', '127.0.0.1'];

const BACKEND_HOST = '127.0.0.1';
const BACKEND_PORT = 8000;
const BACKEND_STATUS_PATH = '/status';

const BACKEND_HEALTHCHECK_TIMEOUT_MS = 1200;
const BACKEND_WAIT_EXISTING_SECONDS = 30;
const BACKEND_WAIT_AFTER_START_SECONDS = 60;
const BACKEND_STARTUP_GRACE_DELAY_MS = 1000;
const BACKEND_WAIT_LOG_EVERY_ATTEMPTS = 5;
const BACKEND_WAIT_INTERVAL_MS = 1000;

const FRONTEND_DIST_INDEX_REL = path.join('..', 'dist', 'index.html');

let sessionSecurityInitialized = false;

function isDevEnvironment() {
    return !app.isPackaged;
}

function getFrontendDevUrl(host = 'localhost') {
    return `http://${host}:${DEV_SERVER_PORT}`;
}

function getFrontendDevUrlCandidates() {
    return DEV_SERVER_HOSTS.map((h) => getFrontendDevUrl(h));
}

function getFrontendDistPath() {
    // Use __dirname (electron/) so it still resolves correctly in packaged scenarios
    return path.join(__dirname, FRONTEND_DIST_INDEX_REL);
}

function getBackendBaseUrl() {
    return `http://${BACKEND_HOST}:${BACKEND_PORT}`;
}

function getBackendStatusUrl() {
    return `${getBackendBaseUrl()}${BACKEND_STATUS_PATH}`;
}

function log(scope, message, ...args) {
    const prefix = `[${scope}]`;
    if (args.length) {
        console.log(prefix, message, ...args);
        return;
    }
    console.log(`${prefix} ${message}`);
}

function logError(scope, message, ...args) {
    const prefix = `[${scope}]`;
    if (args.length) {
        console.error(prefix, message, ...args);
        return;
    }
    console.error(`${prefix} ${message}`);
}

function buildCspHeader({ isDev }) {
    const connectSrc = isDev
        ? `'self' http://localhost:* ws://localhost:* http://127.0.0.1:* ws://127.0.0.1:*`
        : `'self' http://localhost:* ws://localhost:* http://127.0.0.1:* ws://127.0.0.1:* https://cdn.jsdelivr.net https://storage.googleapis.com`;

    // Keep this strict for app content; do not weaken it to accommodate dev tooling.
    return [
        "default-src 'self';",
        "base-uri 'self';",
        "object-src 'none';",
        "frame-ancestors 'none';",
        "script-src 'self' 'wasm-unsafe-eval' blob:;",
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com;",
        "font-src 'self' https://fonts.gstatic.com;",
        "img-src 'self' data: blob:;",
        "media-src 'self' blob:;",
        `connect-src ${connectSrc};`,
        "worker-src 'self' blob:;"
    ].join(' ');
}

function shouldBypassCspForUrl(url) {
    // In development, let the Vite dev server control its own headers (prevents breaking the preamble/HMR).
    return isDevEnvironment() && getFrontendDevUrlCandidates().some((origin) => url.startsWith(origin));
}

function setupSessionSecurity() {
    if (sessionSecurityInitialized) return;
    sessionSecurityInitialized = true;

    const isDev = isDevEnvironment();

    // Permission policy: allow A/V capture prompts for supported features.
    try {
        session.defaultSession.setPermissionRequestHandler((webContents, permission, callback) => {
            if (permission === 'media' || permission === 'camera' || permission === 'microphone') {
                callback(true);
                return;
            }
            callback(false);
        });

        if (typeof session.defaultSession.setPermissionCheckHandler === 'function') {
            session.defaultSession.setPermissionCheckHandler((webContents, permission) => {
                return permission === 'media' || permission === 'camera' || permission === 'microphone';
            });
        }
    } catch (e) {
        logError('Main', 'Failed to set media permission handlers:', e);
    }

    // CSP: strict for our app content; do not override Vite dev server responses in development.
    session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
        if (shouldBypassCspForUrl(details.url)) {
            return callback({ responseHeaders: details.responseHeaders });
        }

        const csp = buildCspHeader({ isDev });
        callback({
            responseHeaders: {
                ...details.responseHeaders,
                'Content-Security-Policy': [csp]
            }
        });
    });

    log('Main', `Session security initialized (dev=${isDev})`);
}

// Use ANGLE D3D11 backend - more stable on Windows while keeping WebGL working
// This fixes "GPU state invalid after WaitForGetOffsetInRange" error
app.commandLine.appendSwitch('use-angle', 'd3d11');
if (process.env.ELECTRON_ENABLE_VULKAN === '1') {
    app.commandLine.appendSwitch('enable-features', 'Vulkan');
}
app.commandLine.appendSwitch('ignore-gpu-blocklist');

let mainWindow;
let pythonProcess;
let cachedDistPath = null;
let cachedDevUrl = null;

// Backend lifecycle state
let backendReady = false;           // true once /status returned 200 at least once
let isShuttingDown = false;         // set to true in will-quit so we skip restarts
let backendRestartCount = 0;        // how many times we have auto-restarted
const BACKEND_MAX_RESTARTS = 5;     // give up after this many consecutive crashes
let backendWaitGeneration = 0;      // bumped each time a new waitForBackend is started;
                                    // old waiters detect the change and cancel themselves

/**
 * Push a backend-status event to the renderer.
 * The renderer's IPC listener responds immediately without any fetch.
 */
function notifyRendererBackendStatus(status) {
    if (mainWindow && !mainWindow.isDestroyed()) {
        try {
            mainWindow.webContents.send('backend-status', status);
        } catch (_) {}
    }
}

function buildDevSplashHtml({ title, message, primaryActionLabel, secondaryActionLabel, showSecondary }) {
    const safeTitle = String(title || 'Jarvis');
    const safeMessage = String(message || '');
    return `<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>${safeTitle}</title>
  </head>
  <body style="margin:0; height:100vh; display:flex; align-items:center; justify-content:center; background:#0b0f14; color:#e7eef7; font-family:system-ui,-apple-system,Segoe UI,Roboto,Inter,sans-serif;">
    <div style="width:min(840px, 92vw); border:1px solid rgba(255,255,255,0.08); background:rgba(15,22,32,0.75); border-radius:14px; padding:24px; box-shadow:0 30px 90px rgba(0,0,0,0.55);">
      <div style="display:flex; justify-content:space-between; align-items:center; gap:16px;">
        <div style="font-size:14px; letter-spacing:0.24em; color:#5cf5ff; font-weight:600;">JARVIS</div>
        <div style="font-size:12px; opacity:0.65;">DEV MODE</div>
      </div>
      <div style="margin-top:18px; font-size:18px; font-weight:600;">${safeTitle}</div>
      <div style="margin-top:10px; opacity:0.75; line-height:1.6;">${safeMessage}</div>
      <pre id="instructions" style="margin-top:14px; background:rgba(0,0,0,0.22); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:12px; color:#a9b7c6; overflow:auto;">npm run dev
// then in another terminal
npx electron .</pre>
    </div>
  </body>
</html>`;
}

function wireWindowEvents(win, { isDev }) {
    win.once('ready-to-show', () => {
        windowWasShown = true;
        win.show();
    });

    // Mirror renderer console messages into the main process for easier debugging
    win.webContents.on('console-message', (event, level, message, line, sourceId) => {
        log('Renderer', `(${level}) ${message} (${sourceId}:${line})`);
    });

    win.webContents.on('dom-ready', () => {
        log('Renderer', 'DOM ready');
    });

    win.webContents.on('did-finish-load', () => {
        log('Renderer', 'Finished load');
    });

    win.webContents.on('did-fail-load', (event, errorCode, errorDescription, validatedURL) => {
        logError('Renderer', `did-fail-load (${errorCode}): ${errorDescription} @ ${validatedURL}`);
        if (isDev) {
            win.webContents.openDevTools();
        }
    });

    win.webContents.on('render-process-gone', (event, details) => {
        logError('Renderer', 'render-process-gone:', details);
        if (isDev) {
            win.webContents.openDevTools();
        }
    });

    win.webContents.on('unresponsive', () => {
        logError('Renderer', 'Renderer became unresponsive');
    });

    win.on('closed', () => {
        if (mainWindow === win) {
            mainWindow = null;
        }
    });
}

function createWindow() {
    const isDev = isDevEnvironment();

    mainWindow = new BrowserWindow({
        width: 1920,
        height: 1080,
        webPreferences: {
            // Keep the renderer as a browser environment so web libraries (e.g. MediaPipe) behave correctly.
            // IPC is exposed via `electron/preload.js`.
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.js'),
        },
        backgroundColor: '#000000',
        frame: false, // Frameless for custom UI
        titleBarStyle: 'hidden',
        show: false, // We'll show early via a splash so the user always sees something
    });

    log('Main', `BrowserWindow created (dev=${isDev})`);
    wireWindowEvents(mainWindow, { isDev });

    // In dev, load Vite server. In prod, load index.html
    const loadFrontend = (retries = 3) => {
        const url = isDev ? getFrontendDevUrl('localhost') : null;
        const distPath = getFrontendDistPath();
        const hasDist = fs.existsSync(distPath);
        cachedDevUrl = url;
        cachedDistPath = hasDist ? distPath : null;
        if (isDev) {
            const splashHtml = buildDevSplashHtml({
                title: 'Starting UI…',
                message: 'Trying to connect to the Vite dev server. If it is not running, start it and click Retry.',
                primaryActionLabel: 'Retry',
                secondaryActionLabel: 'Load built UI',
                showSecondary: Boolean(hasDist)
            });
            try {
                mainWindow.loadURL(`data:text/html,${encodeURIComponent(splashHtml)}`);
            } catch (_) {
            }
            if (!windowWasShown) {
                windowWasShown = true;
                mainWindow.show();
            }
        }

        const loadPromise = isDev ? mainWindow.loadURL(url) : mainWindow.loadFile(distPath);

        loadPromise
            .then(() => {
                log('Main', 'Frontend loaded successfully');
                if (isDev) {
                    mainWindow.webContents.openDevTools();
                }
            })
            .catch((err) => {
                logError('Main', `Failed to load frontend: ${err.message}`);
                if (retries > 0) {
                    log('Main', `Retrying in 1 second... (${retries} retries left)`);
                    setTimeout(() => loadFrontend(retries - 1), 1000);
                } else {
                    if (isDev) {
                        const message = hasDist
                            ? 'Vite dev server is not reachable. You can start it and Retry, or load the built UI (it may be outdated).'
                            : 'Vite dev server is not reachable and no production build was found.';
                        logError('Main', message);
                        const splashHtml = buildDevSplashHtml({
                            title: 'Unable to load UI',
                            message,
                            primaryActionLabel: 'Retry',
                            secondaryActionLabel: 'Load built UI (may be outdated)',
                            showSecondary: Boolean(hasDist)
                        });
                        mainWindow.loadURL(`data:text/html,${encodeURIComponent(splashHtml)}`);
                    } else {
                        logError('Main', 'Failed to load frontend after all retries. Keeping window open.');
                    }

                    windowWasShown = true;
                    mainWindow.show();
                    if (isDev) {
                        mainWindow.webContents.openDevTools();
                    }
                }
            });
    };

    loadFrontend();
}

function checkBackendHealth(timeoutMs = 1000) {
    return new Promise((resolve) => {
        const http = require('http');
        const req = http.get(getBackendStatusUrl(), (res) => {
            const ok = res.statusCode === 200;
            res.resume();
            resolve(ok);
        });
        req.on('error', () => resolve(false));
        req.setTimeout(timeoutMs, () => {
            try {
                req.destroy();
            } catch (_) {
            }
            resolve(false);
        });
    });
}

function startPythonBackend() {
    const scriptPath = path.join(__dirname, '../backend/server.py');
    log('Backend', `Starting Python backend: ${scriptPath}`);

    // On Windows use the 'py' launcher with an explicit version flag so we
    // always get the Python that has all ML packages installed (3.11).
    // The bare 'python' command can resolve to a different version (e.g. 3.14)
    // whose site-packages lack torch / diffusers / vosk / etc.
    // On non-Windows fall back to 'python3'.
    const [pyExe, pyArgPrefix] = process.platform === 'win32'
        ? ['py', ['-3.11']]
        : ['python3', []];

    pythonProcess = spawn(pyExe, [...pyArgPrefix, '-u', scriptPath], {
        cwd: path.join(__dirname, '../backend'),
        stdio: 'pipe',
        env: {
            ...process.env,
            // Force line-buffered stdout so every print() appears in the
            // Electron terminal immediately (equivalent to `python -u` but
            // also covers libraries that check the env var directly).
            PYTHONUNBUFFERED: '1',
            // Disable ALL tqdm progress bars globally at the OS-env level.
            // tqdm checks this variable at import time — setting it here means
            // tqdm never attempts to initialise a progress bar or flush stderr,
            // completely preventing the "OSError: [Errno 22] Invalid argument"
            // that occurs when stderr is a Windows named pipe (not a real TTY).
            TQDM_DISABLE: '1',
        }
    });

    pythonProcess.stdout.on('data', (data) => {
        const output = data.toString();
        log('Backend', output.trimEnd());
    });

    pythonProcess.stderr.on('data', (data) => {
        const output = data.toString();
        logError('Backend', output.trimEnd());
    });
    
    pythonProcess.on('error', (err) => {
        logError('Backend', `Failed to start Python backend: ${err.message}`);
        logError('Backend', 'Make sure Python is installed and in your PATH');
    });

    pythonProcess.on('exit', (code, signal) => {
        if (code !== null && code !== 0) {
            logError('Backend', `Backend exited with code ${code}`);
        }
        if (signal) {
            logError('Backend', `Backend killed with signal ${signal}`);
        }

        // Notify renderer so it shows "Backend Offline" immediately
        backendReady = false;
        notifyRendererBackendStatus('offline');

        // Auto-restart on unexpected crashes (not on clean app shutdown)
        if (isShuttingDown) return;
        if (code === 0 && !signal) return; // intentional clean exit

        // Exit code 1 = Python raised an exception at import/startup time
        // (ModuleNotFoundError, SyntaxError, etc.).  Restarting will NEVER fix
        // this — the user must install missing packages or fix the environment.
        if (code === 1) {
            logError('Backend', '════════════════════════════════════════════════');
            logError('Backend', 'Backend exited with Python error (exit code 1).');
            logError('Backend', 'This is a missing dependency or configuration');
            logError('Backend', 'problem that restarting cannot fix.');
            logError('Backend', 'Run this command to install all required packages:');
            logError('Backend', '  py -3.11 -m pip install -r requirements.txt');
            logError('Backend', 'Then restart the application.');
            logError('Backend', '════════════════════════════════════════════════');
            return; // do not restart
        }

        if (backendRestartCount < BACKEND_MAX_RESTARTS) {
            backendRestartCount += 1;
            const delayMs = Math.min(3000 * backendRestartCount, 15000);
            log('Backend', `Restarting in ${delayMs / 1000}s (attempt ${backendRestartCount}/${BACKEND_MAX_RESTARTS})`);
            setTimeout(() => {
                if (isShuttingDown) return;
                startPythonBackend();
                setTimeout(() => {
                    waitForBackend(BACKEND_WAIT_AFTER_START_SECONDS)
                        .then(() => {
                            backendRestartCount = 0; // successful restart resets counter
                            backendReady = true;
                            notifyRendererBackendStatus('ready');
                        })
                        .catch(err => {
                            if (err.message === 'Superseded') return;
                            logError('Backend', `Backend restart failed to become ready: ${err.message}`);
                        });
                }, BACKEND_STARTUP_GRACE_DELAY_MS);
            }, delayMs);
        } else {
            logError('Backend', `Backend has crashed ${BACKEND_MAX_RESTARTS} times in a row — giving up.`);
            logError('Backend', 'Restart the application to try again.');
        }
    });
}

/**
 * Kill any process currently listening on `port`.
 *
 * IMPORTANT: uses async `exec` (not `execSync`) so the Electron main thread
 * is never blocked.  execSync inside a Promise constructor still runs
 * synchronously — it freezes the entire event loop and delays the window
 * from appearing for 2-4 s on Windows while netstat.exe runs.
 */
function killPortProcess(port) {
    const { exec } = require('child_process');
    const systemRoot = process.env.SystemRoot || 'C:\\Windows';

    return new Promise((resolve) => {
        if (process.platform === 'win32') {
            const netstatPath = path.join(systemRoot, 'System32', 'netstat.exe');
            const taskkillPath = path.join(systemRoot, 'System32', 'taskkill.exe');

            exec(`"${netstatPath}" -ano`, { encoding: 'utf8' }, (err, stdout) => {
                if (err || !stdout) { resolve(); return; }

                const pids = new Set();
                for (const line of stdout.split('\n')) {
                    if (line.includes(`:${port}`) && (line.includes('LISTENING') || line.includes('ESTABLISHED'))) {
                        const parts = line.trim().split(/\s+/);
                        const pid = parseInt(parts[parts.length - 1], 10);
                        if (!isNaN(pid) && pid > 0) pids.add(pid);
                    }
                }

                if (pids.size === 0) { resolve(); return; }

                const kills = [...pids].map(
                    (pid) => new Promise((r) => {
                        exec(`"${taskkillPath}" /pid ${pid} /f /t`, () => {
                            log('Backend', `Killed orphaned process on port ${port} (PID ${pid})`);
                            r();
                        });
                    })
                );
                Promise.all(kills).then(() => resolve());
            });
        } else {
            exec(`lsof -ti :${port}`, { encoding: 'utf8' }, (err, stdout) => {
                if (err || !stdout) { resolve(); return; }
                const pids = stdout.trim().split('\n').filter(Boolean);
                const kills = pids.map(
                    (pid) => new Promise((r) => {
                        exec(`kill -9 ${pid}`, () => {
                            log('Backend', `Killed orphaned process on port ${port} (PID ${pid})`);
                            r();
                        });
                    })
                );
                Promise.all(kills).then(() => resolve());
            });
        }
    });
}

function ensureBackendRunning() {
    // Always kill any orphaned process on the backend port so we never
    // accidentally reuse a stale Python process from a previous session.
    killPortProcess(BACKEND_PORT).then(() => {
        startPythonBackend();
        setTimeout(() => {
            waitForBackend(BACKEND_WAIT_AFTER_START_SECONDS)
                .then(() => {
                    backendReady = true;
                    notifyRendererBackendStatus('ready');
                    log('Backend', 'Backend ready — notified renderer');
                })
                .catch((err) => {
                    if (err.message === 'Superseded') return; // a newer waiter took over
                    logError('Backend', `Backend failed to start: ${err.message}`);
                    logError('Backend', 'Check the backend output above for errors');
                });
        }, BACKEND_STARTUP_GRACE_DELAY_MS);
    });
}

app.whenReady().then(() => {
    setupSessionSecurity();

    // Renderer can call this synchronously on mount to get current backend state.
    // Handles the race where the renderer reloads after main already sent 'ready'.
    ipcMain.handle('get-backend-status', () => backendReady ? 'ready' : 'offline');

    ipcMain.on('renderer-load-dev', async () => {
        if (!mainWindow) return;
        const url = cachedDevUrl || getFrontendDevUrl('127.0.0.1');
        try {
            await mainWindow.loadURL(url);
        } catch (e) {
            const splashHtml = buildDevSplashHtml({
                title: 'Dev server not reachable',
                message: `Could not load ${url}. Start Vite (npm run dev) and retry.`,
                primaryActionLabel: 'Retry',
                secondaryActionLabel: cachedDistPath ? 'Load built UI (may be outdated)' : 'Load built UI',
                showSecondary: Boolean(cachedDistPath)
            });
            try {
                await mainWindow.loadURL(`data:text/html,${encodeURIComponent(splashHtml)}`);
            } catch (_) {
            }
            mainWindow.show();
        }
    });

    ipcMain.on('renderer-load-dist', async () => {
        if (!mainWindow) return;
        if (!cachedDistPath) {
            const splashHtml = buildDevSplashHtml({
                title: 'No build found',
                message: 'dist/index.html was not found. Run: npm run build',
                primaryActionLabel: 'Retry',
                showSecondary: false
            });
            try {
                await mainWindow.loadURL(`data:text/html,${encodeURIComponent(splashHtml)}`);
            } catch (_) {
            }
            mainWindow.show();
            return;
        }

        try {
            await mainWindow.loadFile(cachedDistPath);
        } catch (e) {
            const splashHtml = buildDevSplashHtml({
                title: 'Failed to load build',
                message: `Could not load dist UI: ${e.message}`,
                primaryActionLabel: 'Retry',
                showSecondary: false
            });
            try {
                await mainWindow.loadURL(`data:text/html,${encodeURIComponent(splashHtml)}`);
            } catch (_) {
            }
            mainWindow.show();
        }
    });

    ipcMain.on('window-minimize', () => {
        if (mainWindow) mainWindow.minimize();
    });

    ipcMain.on('window-maximize', () => {
        if (mainWindow) {
            if (mainWindow.isMaximized()) {
                mainWindow.unmaximize();
            } else {
                mainWindow.maximize();
            }
        }
    });

    ipcMain.on('window-close', () => {
        if (mainWindow) mainWindow.close();
    });

    createWindow();
    ensureBackendRunning();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

function checkBackendPort(port) {
    return new Promise((resolve) => {
        const net = require('net');
        const server = net.createServer();
        server.once('error', (err) => {
            if (err.code === 'EADDRINUSE') {
                resolve(true);
            } else {
                resolve(false);
            }
        });
        server.once('listening', () => {
            server.close();
            resolve(false);
        });
        server.listen(port);
    });
}

function waitForBackend(maxWaitSeconds = 60) {
    // Increment the global generation so any previously running waitForBackend
    // call detects it has been superseded and quietly cancels itself.
    // This prevents multiple parallel countdown timers piling up when the
    // backend crashes and restarts multiple times in quick succession.
    const myGeneration = ++backendWaitGeneration;

    return new Promise((resolve, reject) => {
        let attempts = 0;
        const maxAttempts = maxWaitSeconds;
        const check = () => {
            // If a newer waitForBackend was started, abandon this one silently.
            if (backendWaitGeneration !== myGeneration) {
                reject(new Error('Superseded'));
                return;
            }
            attempts++;
            const http = require('http');
            http.get(getBackendStatusUrl(), (res) => {
                if (res.statusCode === 200) {
                    log('Backend', 'Backend is ready');
                    resolve();
                } else {
                    if (attempts >= maxAttempts) {
                        logError('Backend', `Backend failed to start after ${maxAttempts} seconds`);
                        reject(new Error('Backend timeout'));
                    } else {
                        log('Backend', `Backend not ready (attempt ${attempts}/${maxAttempts}), retrying...`);
                        setTimeout(check, BACKEND_WAIT_INTERVAL_MS);
                    }
                }
            }).on('error', (err) => {
                if (attempts >= maxAttempts) {
                    logError('Backend', `Backend failed to start after ${maxAttempts} seconds: ${err.message}`);
                    logError('Backend', 'Check the backend output above for errors');
                    reject(new Error('Backend timeout'));
                } else {
                    if (attempts % BACKEND_WAIT_LOG_EVERY_ATTEMPTS === 0) {
                        log('Backend', `Waiting for backend... (${attempts}/${maxAttempts} seconds)`);
                    }
                    setTimeout(check, BACKEND_WAIT_INTERVAL_MS);
                }
            });
        };
        check();
    });
}

let windowWasShown = false;

app.on('window-all-closed', () => {
    // Only quit if the window was actually shown at least once
    // This prevents quitting during startup if window creation fails
    if (process.platform !== 'darwin' && windowWasShown) {
        app.quit();
    } else if (!windowWasShown) {
        log('Main', 'Window was never shown - keeping app alive to allow retries');
    }
});

app.on('will-quit', () => {
    isShuttingDown = true; // prevent auto-restart when we intentionally kill the backend
    log('Main', 'App closing... Killing Python backend.');
    if (pythonProcess) {
        if (process.platform === 'win32') {
            // Windows: Force kill the process tree synchronously
            try {
                const { execSync } = require('child_process');
                const systemRoot = process.env.SystemRoot || 'C:\\Windows';
                const taskkillPath = path.join(systemRoot, 'System32', 'taskkill.exe');
                execSync(`"${taskkillPath}" /pid ${pythonProcess.pid} /f /t`);
            } catch (e) {
                logError('Backend', `Failed to kill python process: ${e.message}`);
                try {
                    pythonProcess.kill();
                } catch (_) {
                    // ignore
                }
            }
        } else {
            // Unix: SIGKILL
            pythonProcess.kill('SIGKILL');
        }
        pythonProcess = null;
    }
});
