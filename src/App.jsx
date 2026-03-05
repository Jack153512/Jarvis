import React, { useEffect, useState, useRef, useCallback } from 'react';
import io from 'socket.io-client';

import Visualizer from './components/Visualizer';
import TopAudioBar from './components/TopAudioBar';
import CadWindow from './components/CadWindow';
import BrowserWindow from './components/BrowserWindow';
import ImageWindow from './components/ImageWindow';
import ConsoleChat from './components/ConsoleChat';
import ChatModule from './components/ChatModule';
import ToolsModule from './components/ToolsModule';
import { Minus, X, Box, Globe, ImageIcon } from 'lucide-react';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import { AnimatePresence, motion } from 'framer-motion';
import ConfirmationPopup from './components/ConfirmationPopup';
import SettingsWindow from './components/SettingsWindow';
import NavigationRail from './components/NavigationRail';
import AlertPanel from './components/AlertPanel';
import EventTimeline from './components/EventTimeline';
import CommandPalette from './components/CommandPalette';
import OverlayHeader from './components/OverlayHeader';
import ConversationSidebar from './components/ConversationSidebar';
import LiveClock from './components/LiveClock';

const SOCKET_URL = import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:8000';
const getSocket = () => {
    try {
        const g = globalThis;
        const existing = g.__jarvis_socketio;
        if (existing && typeof existing.emit === 'function') {
            return existing;
        }
        const created = io(SOCKET_URL, {
            autoConnect: false,   // connect only after /status probe confirms backend is ready
            reconnection: false,  // probe handles all reconnection — eliminates WS error spam
            transports: ['websocket', 'polling'],
            timeout: 10000,
        });
        g.__jarvis_socketio = created;
        return created;
    } catch (_) {
        return io(SOCKET_URL, {
            autoConnect: false,
            reconnection: false,
            transports: ['websocket', 'polling'],
            timeout: 10000,
        });
    }
};
const socket = getSocket();
const ipcRenderer = window?.electronAPI?.ipcRenderer;

const getHandLandmarkerSingleton = () => {
    const g = globalThis;
    if (g.__jarvis_hand_landmarker_promise) return g.__jarvis_hand_landmarker_promise;

    g.__jarvis_hand_landmarker_promise = (async () => {
        const vision = await FilesetResolver.forVisionTasks(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm'
        );

        const modelPath =
            'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

        // createFromOptions instead of createFromModelPath so we can tune
        // the three settings that most affect performance:
        //
        //  runningMode "VIDEO"  — keeps an internal tracker between frames so
        //    only the first frame pays full-detection cost (~70 ms).  Subsequent
        //    frames use the cheap tracking path (~5–15 ms).  The default "IMAGE"
        //    mode re-runs the full detector every frame, which is why FPS drops
        //    to ~14 even with frame-skipping.
        //
        //  delegate "GPU"  — routes preprocessing and model execution through
        //    WebGL (Electron uses Chromium's GPU process).  Without this the
        //    WASM CPU path is used, which is 3–8× slower.
        //
        //  numHands 1  — halves detection/landmark cost; a second hand can be
        //    re-enabled if multi-hand tracking is needed later.
        const landmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: modelPath,
                delegate: 'GPU',
            },
            runningMode: 'VIDEO',
            numHands: 1,
            minHandDetectionConfidence: 0.5,
            minHandPresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
        });
        g.__jarvis_hand_landmarker_instance = landmarker;
        return landmarker;
    })();

    return g.__jarvis_hand_landmarker_promise;
};


function App() {
    // Boot sequence — shows a full-screen overlay for 1.8 s on first mount.
    // Uses a single state bool; the overlay is pure CSS after that.
    const [isBooting, setIsBooting] = useState(true);
    useEffect(() => {
        const t = setTimeout(() => setIsBooting(false), 1800);
        return () => clearTimeout(t);
    }, []);

    const [status, setStatus] = useState('Disconnected');
    const [socketConnected, setSocketConnected] = useState(socket.connected);
    
    // No face authentication - always authenticated
    const [isAuthenticated, setIsAuthenticated] = useState(true);

    const [isConnected, setIsConnected] = useState(true); // Power state DEFAULT ON
    const [isMuted, setIsMuted] = useState(true); // Mic state DEFAULT MUTED
    const [isVideoOn, setIsVideoOn] = useState(false); // Video state
    const [messages, setMessages] = useState([]);
    const [inputValue, setInputValue] = useState('');
    const [cadData, setCadData] = useState(null);
    const [cadThoughts, setCadThoughts] = useState(''); // Streaming AI thoughts
    const [cadRetryInfo, setCadRetryInfo] = useState({ attempt: 1, maxAttempts: 3, error: null }); // Retry status
    const [browserData, setBrowserData] = useState({ image: null, logs: [] });
    const [confirmationRequest, setConfirmationRequest] = useState(null); // { id, tool, args }
    const [showCadWindow, setShowCadWindow] = useState(false);
    const [showBrowserWindow, setShowBrowserWindow] = useState(false);
    const [showImageWindow, setShowImageWindow] = useState(false);
    const [imageData, setImageData] = useState(null);

    // Conversation management
    const [conversations, setConversations] = useState([]);
    const [activeConversationId, setActiveConversationId] = useState(null);
    const [showConversationSidebar, setShowConversationSidebar] = useState(false);
    // currentTime removed — see LiveClock component above App for the clock display.

    const [isTyping, setIsTyping] = useState(false);
    const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
    const [activityLog, setActivityLog] = useState([]);
    const [systemLogs, setSystemLogs] = useState([]); // Dynamic Terminal Feed

    // Specular highlight — update CSS variable directly, NO React state.
    // Using setState here caused App to re-render on every single mouse move event.
    useEffect(() => {
        const handleMouseMove = (e) => {
            const x = (e.clientX / window.innerWidth) * 100 + '%';
            const y = (e.clientY / window.innerHeight) * 100 + '%';
            document.documentElement.style.setProperty('--specular-x', x);
            document.documentElement.style.setProperty('--specular-y', y);
        };
        window.addEventListener('mousemove', handleMouseMove, { passive: true });
        return () => window.removeEventListener('mousemove', handleMouseMove);
    }, []);

    // Adaptive lighting — update CSS variable directly, NO React state.
    useEffect(() => {
        let color = 'rgba(92, 245, 255, 0.05)'; // Default Cyan
        if (showCadWindow) color = 'rgba(34, 197, 94, 0.05)'; // Green
        else if (showBrowserWindow) color = 'rgba(59, 130, 246, 0.05)'; // Blue
        document.documentElement.style.setProperty('--adaptive-color', color);
    }, [showCadWindow, showBrowserWindow]);

    const addSystemLog = useCallback((msg) => {
        const timestamp = new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
        setSystemLogs(prev => [{ id: Date.now(), msg, timestamp }, ...prev].slice(0, 5));
    }, []);

    const toolLogByIdRef = useRef(new Map());
    const toolLogOrderRef = useRef([]);

    // RESTORED STATE
    const [aiAudioData, setAiAudioData] = useState(new Array(64).fill(0));
    const [micAudioData, setMicAudioData] = useState(new Array(32).fill(0));
    const [fps, setFps] = useState(0);

    // Device states - microphones, speakers, webcams
    const [micDevices, setMicDevices] = useState([]);
    const [speakerDevices, setSpeakerDevices] = useState([]);
    const [webcamDevices, setWebcamDevices] = useState([]);

    // Selected device IDs - restored from localStorage
    const [selectedMicId, setSelectedMicId] = useState(() => localStorage.getItem('selectedMicId') || '');
    const [selectedSpeakerId, setSelectedSpeakerId] = useState(() => localStorage.getItem('selectedSpeakerId') || '');
    const [selectedWebcamId, setSelectedWebcamId] = useState(() => localStorage.getItem('selectedWebcamId') || '');
    const [showSettings, setShowSettings] = useState(false);
    const [currentProject, setCurrentProject] = useState('default');
    const [identity, setIdentity] = useState({ user_name: 'User', assistant_name: 'JARVIS' });
    const identityRef = useRef({ user_name: 'User', assistant_name: 'JARVIS' });
    const sessionIdRef = useRef(`s_${Date.now()}`);
    const lastAiMsgRef = useRef(null);
    const aiSaveTimerRef = useRef(null);
    const activeConvIdRef = useRef(null);
    // true while the active conversation has no stored messages yet (used for auto-title)
    const isNewConvRef = useRef(true);
    // Stores the ID of the last recommendation so feedback commands can reference it.
    const lastRecommendationIdRef = useRef(null);
    // Refs that always hold the latest handleSend / handleSendButton implementations.
    // The stable useCallback wrappers below never change reference, allowing
    // React.memo(ConsoleChat) to skip re-renders when unrelated App state changes.
    const _handleSendImpl       = useRef(null);
    const _handleSendButtonImpl = useRef(null);
    const [settings, setSettings] = useState(null);
    const ttsEnabledRef = useRef(true);

    // Modular Mode State
    const [isModularMode, setIsModularMode] = useState(false);
    const [elementPositions, setElementPositions] = useState({
        video: { x: 40, y: 80 },
        visualizer: { x: window.innerWidth / 2, y: window.innerHeight / 2 - 150 },
        chat: { x: window.innerWidth / 2, y: window.innerHeight / 2 + 100 },
        cad: { x: window.innerWidth / 2 + 300, y: window.innerHeight / 2 },
        browser: { x: window.innerWidth / 2 - 300, y: window.innerHeight / 2 },
        image: { x: window.innerWidth / 2 - 150, y: window.innerHeight / 2 },
        tools: { x: window.innerWidth / 2, y: window.innerHeight - 100 }
    });

    const [elementSizes, setElementSizes] = useState({
        visualizer: { w: 550, h: 350 },
        chat: { w: 550, h: 220 },
        tools: { w: 500, h: 80 },
        cad: { w: 400, h: 400 },
        browser: { w: 550, h: 380 },
        image: { w: 480, h: 420 },
        video: { w: 320, h: 180 }
    });
    const [activeDragElement, setActiveDragElement] = useState(null);

    // Z-Index Stacking Order
    const [zIndexOrder, setZIndexOrder] = useState([
        'visualizer', 'chat', 'tools', 'video', 'cad', 'browser'
    ]);

    // Hand Control State
    // cursorPos is intentionally NOT useState — updating a React state 20 fps
    // forces the entire App tree to re-render on every frame.  Instead we keep
    // a ref and write directly to the cursor div's style in predictWebcam.
    const cursorDivRef = useRef(null);
    const [isPinching, setIsPinching] = useState(false);
    const [isHandTrackingEnabled, setIsHandTrackingEnabled] = useState(false);
    const [cursorSensitivity, setCursorSensitivity] = useState(2.0);
    const [isCameraFlipped, setIsCameraFlipped] = useState(false);

    // Refs for Loop Access
    const isHandTrackingEnabledRef = useRef(false);
    const cursorSensitivityRef = useRef(2.0);
    const isCameraFlippedRef = useRef(false);
    const handLandmarkerRef = useRef(null);
    const cursorTrailRef = useRef([]);
    const [ripples, setRipples] = useState([]);

    // Web Audio Context for Mic Visualization
    const audioContextRef = useRef(null);
    const analyserRef = useRef(null);
    const sourceRef = useRef(null);
    const animationFrameRef = useRef(null);
    const micStreamRef = useRef(null);
    
    const ttsAudioContextRef = useRef(null);
    const ttsAudioQueueRef = useRef([]);       // Queue of { audio: base64, index: number }
    const isTTSPlayingRef = useRef(false);
    const currentTTSAudioRef = useRef(null);   // Current playing Audio element
    const ttsStoppedRef = useRef(false);       // Flag to stop playback

    // Push-to-talk MediaRecorder refs (replaces Web Speech API)
    const mediaRecorderRef = useRef(null);
    const pttChunksRef = useRef([]);
    const pttMimeTypeRef = useRef('audio/webm');
    const [isRecording, setIsRecording] = useState(false);
    const lastSentUtteranceRef = useRef('');
    const micLevelRef = useRef(0);
    const ttsEndedRef = useRef(false);
    // Ref-copy of selectedMicId so async PTT callbacks don't capture stale state
    const selectedMicIdRef = useRef(null);
    
    // Initialization guards to prevent duplicate initialization
    const handLandmarkerInitializedRef = useRef(false);
    const isInitializingRef = useRef(false);

    // Video Refs
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const transmissionCanvasRef = useRef(null);
    const videoIntervalRef = useRef(null);
    const lastFrameTimeRef = useRef(0);
    const frameCountRef = useRef(0);
    const lastVideoTimeRef = useRef(-1);
    // Cached camera stream — kept alive across toggles for instant restart
    const streamRef = useRef(null);
    // Cached snap-target positions — rebuilt every 500 ms instead of per-frame DOM query
    const snapTargetsRef = useRef([]);
    // Throttle ML inference to every other frame (~30 fps) while canvas renders at full display rate
    const inferenceFrameSkipRef = useRef(0);
    // Track previous pinch state so setIsPinching only fires on a real change
    const prevIsPinchRef = useRef(false);
    // Ensures WebGL/WASM warm-up runs exactly once when hand tracking is first enabled
    const warmupDoneRef = useRef(false);

    // Refs for state tracking
    const isVideoOnRef = useRef(false);
    const isModularModeRef = useRef(false);
    const elementPositionsRef = useRef(elementPositions);
    const elementSizesRef = useRef(elementSizes);
    const activeDragElementRef = useRef(null);
    const lastActiveDragElementRef = useRef(null);
    const lastCursorPosRef = useRef({ x: 0, y: 0 });
    const lastWristPosRef = useRef({ x: 0, y: 0 });

    // Smoothing and Snapping Refs
    const smoothedCursorPosRef = useRef({ x: 0, y: 0 });
    const snapStateRef = useRef({ isSnapped: false, element: null, snapPos: { x: 0, y: 0 } });
    const oneEuroRef = useRef({
        lastTimeMs: 0,
        x: 0,
        y: 0,
        dx: 0,
        dy: 0,
        initialized: false,
    });

    // Mouse Drag Refs
    const dragOffsetRef = useRef({ x: 0, y: 0 });
    const isDraggingRef = useRef(false);
    const dragTargetPosRef = useRef({ x: 0, y: 0 });
    const dragCurrentPosRef = useRef({ x: 0, y: 0 });
    const dragRafIdRef = useRef(null);
    const dragLastFrameMsRef = useRef(0);

    // Update refs when state changes
    useEffect(() => {
        isModularModeRef.current = isModularMode;
        elementPositionsRef.current = elementPositions;
        elementSizesRef.current = elementSizes;
        isHandTrackingEnabledRef.current = isHandTrackingEnabled;
        cursorSensitivityRef.current = cursorSensitivity;
        isCameraFlippedRef.current = isCameraFlipped;
    }, [isModularMode, elementPositions, isHandTrackingEnabled, cursorSensitivity, isCameraFlipped]);

    // Clock is now handled by the standalone <LiveClock /> component above App.

    // Build and refresh snap-target cache every 500 ms — but ONLY while hand tracking
    // is active.  getBoundingClientRect() on every button/input forces a layout reflow
    // twice per second even when the camera is off, contributing to idle Electron lag.
    useEffect(() => {
        if (!isHandTrackingEnabled) {
            snapTargetsRef.current = [];
            return;
        }
        const rebuild = () => {
            snapTargetsRef.current = Array.from(
                document.querySelectorAll('button, input, select, .draggable')
            ).map(el => {
                const rect = el.getBoundingClientRect();
                return { el, centerX: rect.left + rect.width / 2, centerY: rect.top + rect.height / 2 };
            });
        };
        rebuild();
        const id = setInterval(rebuild, 500);
        return () => clearInterval(id);
    }, [isHandTrackingEnabled]);

    // Warm up WebGL/WASM the first time hand tracking is enabled.
    // MediaPipe compiles shaders on the very first detectForVideo call, causing a
    // visible freeze. Running a dummy inference now moves that cost to a deliberate
    // moment instead of mid-stream.
    useEffect(() => {
        if (
            isHandTrackingEnabled &&
            handLandmarkerRef.current &&
            videoRef.current &&
            isVideoOnRef.current &&
            !warmupDoneRef.current
        ) {
            warmupDoneRef.current = true;
            try {
                handLandmarkerRef.current.detectForVideo(videoRef.current, performance.now());
            } catch (_) {}
        }
    }, [isHandTrackingEnabled]);

    useEffect(() => {
        const onKeyDown = (e) => {
            const isCmdOrCtrl = e.ctrlKey || e.metaKey;
            if (isCmdOrCtrl && (e.key === 'k' || e.key === 'K')) {
                e.preventDefault();
                setCommandPaletteOpen(true);
            }
            if (e.key === 'Escape') {
                setCommandPaletteOpen(false);
            }
        };
        window.addEventListener('keydown', onKeyDown, true);
        return () => window.removeEventListener('keydown', onKeyDown, true);
    }, []);

    const pushActivity = (line) => {
        const msg = String(line || '').trim();
        if (!msg) return;
        setActivityLog(prev => {
            const next = [...prev, msg];
            return next.length > 80 ? next.slice(next.length - 80) : next;
        });
    };

    const safeEmit = (event, payload) => {
        if (socket && socket.connected) {
            try {
                socket.emit(event, payload);
                return true;
            } catch (_) {
                return false;
            }
        }
        const now = Date.now();
        if (now - lastSocketOfflineNoticeMsRef.current > 2500) {
            lastSocketOfflineNoticeMsRef.current = now;
            pushActivity('Backend offline');
        }
        setStatus('Backend Offline');
        setSocketConnected(false);
        return false;
    };

    function stopTTSPlayback() {
        ttsStoppedRef.current = true;
        ttsEndedRef.current = true;
        try {
            const cur = currentTTSAudioRef.current;
            if (cur) {
                try {
                    cur.pause();
                } catch (_) {
                }
                try {
                    cur.src = '';
                } catch (_) {
                }
            }
        } catch (_) {
        }
        currentTTSAudioRef.current = null;
        isTTSPlayingRef.current = false;

        try {
            const q = ttsAudioQueueRef.current;
            if (Array.isArray(q)) {
                for (const item of q) {
                    if (item && item.url) {
                        try {
                            URL.revokeObjectURL(item.url);
                        } catch (_) {
                        }
                    }
                }
            }
        } catch (_) {
        }
        ttsAudioQueueRef.current = [];
    }

    function playNextTTSChunk() {
        if (ttsStoppedRef.current) {
            isTTSPlayingRef.current = false;
            return;
        }

        const queue = ttsAudioQueueRef.current;
        if (!Array.isArray(queue) || queue.length === 0) {
            isTTSPlayingRef.current = false;
            return;
        }

        const next = queue.shift();
        if (!next || !next.url) {
            playNextTTSChunk();
            return;
        }

        const audio = new Audio(next.url);
        currentTTSAudioRef.current = audio;
        isTTSPlayingRef.current = true;

        audio.onended = () => {
            try {
                URL.revokeObjectURL(next.url);
            } catch (_) {
            }
            currentTTSAudioRef.current = null;
            playNextTTSChunk();
        };

        audio.onerror = () => {
            try {
                URL.revokeObjectURL(next.url);
            } catch (_) {
            }
            currentTTSAudioRef.current = null;
            playNextTTSChunk();
        };

        try {
            const p = audio.play();
            if (p && typeof p.catch === 'function') {
                p.catch(() => {
                    currentTTSAudioRef.current = null;
                    isTTSPlayingRef.current = false;
                });
            }
        } catch (_) {
            currentTTSAudioRef.current = null;
            isTTSPlayingRef.current = false;
        }
    }

    function enqueueTTSMp3Base64(audioB64, chunkIndex) {
        if (!audioB64) return;

        let bytes;
        try {
            const binary = atob(audioB64);
            const len = binary.length;
            bytes = new Uint8Array(len);
            for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
        } catch (_) {
            return;
        }

        const blob = new Blob([bytes], { type: 'audio/mpeg' });
        const url = URL.createObjectURL(blob);

        const q = Array.isArray(ttsAudioQueueRef.current) ? ttsAudioQueueRef.current : [];
        q.push({ url, index: Number.isFinite(chunkIndex) ? chunkIndex : q.length });
        q.sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
        ttsAudioQueueRef.current = q;

        if (!isTTSPlayingRef.current && !ttsStoppedRef.current) {
            playNextTTSChunk();
        }
    }

    // Centering Logic (Startup & Resize)
    useEffect(() => {
        const centerElements = () => {
            const width = window.innerWidth;
            const height = window.innerHeight;

            const toolsCenterY = height - 100;
            const gap = 20;
            const chatBottomLimit = height - 140;

            let vizH = 400;
            let chatH = 250;
            const topBarHeight = 60;

            const totalNeeded = topBarHeight + vizH + gap + chatH + gap + 140;

            if (height < totalNeeded) {
                const available = height - topBarHeight - 140 - (gap * 2);
                vizH = available * 0.6;
                chatH = available * 0.4;
            }

            const vizY = topBarHeight + (vizH / 2);
            const chatY = topBarHeight + vizH + gap;

            setElementSizes(prev => ({
                ...prev,
                visualizer: { w: Math.min(600, width * 0.8), h: vizH },
                chat: { w: Math.min(600, width * 0.9), h: chatH }
            }));

            setElementPositions(prev => ({
                ...prev,
                visualizer: { x: width / 2, y: vizY },
                chat: { x: width / 2, y: chatY },
                tools: { x: width / 2, y: toolsCenterY }
            }));
        };

        centerElements();
        window.addEventListener('resize', centerElements);
        return () => window.removeEventListener('resize', centerElements);
    }, []);

    // Utility: Clamp position to viewport
    const clampToViewport = (pos, size) => {
        const margin = 10;
        const topBarHeight = 60;
        const width = window.innerWidth;
        const height = window.innerHeight;

        return {
            x: Math.max(size.w / 2 + margin, Math.min(width - size.w / 2 - margin, pos.x)),
            y: Math.max(size.h / 2 + margin + topBarHeight, Math.min(height - size.h / 2 - margin, pos.y))
        };
    };

    // Utility: Get z-index for an element
    const getZIndex = (id) => {
        const baseZ = 30;
        const index = zIndexOrder.indexOf(id);
        return baseZ + (index >= 0 ? index : 0);
    };

    // Utility: Bring element to front
    const bringToFront = (id) => {
        setZIndexOrder(prev => {
            const filtered = prev.filter(el => el !== id);
            return [...filtered, id];
        });
    };

    const hasAutoConnectedRef = useRef(false);
    const connectErrorCountRef = useRef(0);
    const lastConnectErrorLogMsRef = useRef(0);
    const lastSocketOfflineNoticeMsRef = useRef(0);
    const reconnectPausedRef = useRef(false);
    const reconnectResumeTimerRef = useRef(null);
    const reconnectCooldownMsRef = useRef(5000);
    // Polls GET /status every 3 s when disconnected; calls socket.connect() on 200 OK.
    const backendProbeRef = useRef(null);

    // Auto-Connect Model on Start
    useEffect(() => {
        if (isConnected && isAuthenticated && socketConnected && micDevices.length > 0 && !hasAutoConnectedRef.current) {
            hasAutoConnectedRef.current = true;

            const timer = setTimeout(() => {
                const index = micDevices.findIndex(d => d.deviceId === selectedMicId);
                const queryDevice = micDevices.find(d => d.deviceId === selectedMicId);
                const deviceName = queryDevice ? queryDevice.label : null;
                console.log("Auto-connecting to model with device:", deviceName, "Index:", index);

                setStatus('Connecting...');
                safeEmit('start_audio', {
                    device_index: index >= 0 ? index : null,
                    device_name: deviceName,
                    muted: isMuted
                });
            }, 500);
        }
    }, [isConnected, isAuthenticated, socketConnected, micDevices, selectedMicId]);

    useEffect(() => {
        // ── Backend probe helpers ──────────────────────────────────────────────
        // When running inside Electron the main process tracks backend health
        // via Node.js http.get (invisible to the renderer DevTools) and pushes
        // status events over IPC.  We use ipcRenderer.invoke as a poll fallback
        // and ipcRenderer.on('backend-status') for immediate push notifications.
        //
        // When running in a plain browser (no electronAPI) we fall back to a
        // fetch-based probe — those ERR_CONNECTION_REFUSED entries will appear
        // in DevTools but are harmless.
        const electronIpc = window?.electronAPI?.ipcRenderer;

        const _stopProbe = () => {
            if (backendProbeRef.current) {
                clearInterval(backendProbeRef.current);
                backendProbeRef.current = null;
            }
        };

        const _connectSocket = () => {
            _stopProbe();
            if (!socket.connected) {
                console.log('[Socket] Backend ready — connecting');
                try { socket.connect(); } catch (_) {}
            }
        };

        const _startProbe = () => {
            if (backendProbeRef.current) return; // already running
            if (electronIpc?.invoke) {
                // Electron path: IPC invoke — no renderer-side network request,
                // so nothing ever appears in the DevTools console.
                backendProbeRef.current = setInterval(() => {
                    electronIpc.invoke('get-backend-status')
                        .then(status => { if (status === 'ready') _connectSocket(); })
                        .catch(() => {});
                }, 3000);
            } else {
                // Browser fallback: fetch probe (produces console noise but works)
                backendProbeRef.current = setInterval(() => {
                    fetch(`${SOCKET_URL}/status`, { signal: AbortSignal.timeout(2000) })
                        .then(r => { if (r.ok) _connectSocket(); })
                        .catch(() => {});
                }, 3000);
            }
        };
        // ─────────────────────────────────────────────────────────────────────

        socket.on('connect', () => {
            setStatus('Connected');
            setSocketConnected(true);
            connectErrorCountRef.current = 0;
            reconnectPausedRef.current = false;
            reconnectCooldownMsRef.current = 5000;
            if (reconnectResumeTimerRef.current) {
                clearTimeout(reconnectResumeTimerRef.current);
                reconnectResumeTimerRef.current = null;
            }
            _stopProbe();
            safeEmit('get_settings');
            safeEmit('get_conversations');
        });
        // ── Conversation management ────────────────────────────────────────
        socket.on('conversations_list', (data) => {
            const convs = Array.isArray(data?.conversations) ? data.conversations : [];
            setConversations(convs);
            // Auto-load/create only on the very first connect (no active conversation yet).
            // Subsequent conversations_list events (e.g. after rename) must not trigger this.
            if (activeConvIdRef.current) return;
            if (convs.length > 0) {
                const most_recent = convs[0]; // sorted by updated_at desc
                socket.emit('load_conversation', { id: most_recent.id });
            } else {
                // First ever launch with no history — start a fresh conversation
                socket.emit('create_conversation', { title: 'New Chat' });
            }
        });

        socket.on('conversation_created', (data) => {
            const conv = data?.conversation;
            const convs = Array.isArray(data?.conversations) ? data.conversations : [];
            if (conv) {
                setActiveConversationId(conv.id);
                activeConvIdRef.current = conv.id;
                isNewConvRef.current = true; // fresh — no messages yet
                setConversations(convs);
                setMessages([]);
            }
        });

        socket.on('conversation_loaded', (data) => {
            const conv = data?.conversation;
            const msgs = Array.isArray(data?.messages) ? data.messages : [];
            if (conv) {
                setActiveConversationId(conv.id);
                activeConvIdRef.current = conv.id;
                // Only auto-title if the conversation is actually empty
                isNewConvRef.current = msgs.length === 0;
                // Guarantee every history message has a stable id so ConsoleChat
                // never falls back to the array-index key (which causes duplicate renders
                // when the list is prepended or the component re-mounts).
                const stamped = msgs.map((m, i) =>
                    m.id ? m : { ...m, id: `hist_${conv.id}_${i}`, fromHistory: true }
                );
                setMessages(stamped);
            }
        });

        socket.on('conversation_deleted', (data) => {
            const convs = Array.isArray(data?.conversations) ? data.conversations : [];
            setConversations(convs);
            if (data?.was_active) {
                if (convs.length > 0) {
                    // Load the most recent remaining conversation
                    const next = convs[0];
                    activeConvIdRef.current = next.id;
                    socket.emit('load_conversation', { id: next.id });
                } else {
                    // No conversations remain — show empty state, don't auto-create
                    setActiveConversationId(null);
                    activeConvIdRef.current = null;
                    setMessages([]);
                }
            }
        });
        socket.on('disconnect', () => {
            setStatus('Disconnected');
            setSocketConnected(false);
            _startProbe();
        });
        socket.on('connect_error', (err) => {
            setStatus('Backend Offline');
            setSocketConnected(false);
            const now = Date.now();
            if (now - lastConnectErrorLogMsRef.current > 8000) {
                lastConnectErrorLogMsRef.current = now;
                console.warn('[Socket] connect_error:', err?.message || err, '— polling /status for backend');
            }
            _startProbe();
        });
        socket.io.on('reconnect_failed', () => {
            setStatus('Backend Offline');
            setSocketConnected(false);
        });
        socket.on('status', (data) => {
            addMessage('System', data.msg);
            if (data.msg === 'J.A.R.V.I.S Started') {
                setStatus('Model Connected');
            } else if (data.msg === 'J.A.R.V.I.S Stopped') {
                setStatus('Connected');
            }
        });
        socket.on('audio_data', (data) => {
            setAiAudioData(data.data);
        });

        socket.on('settings', (nextSettings) => {
            setSettings(nextSettings);
            try {
                const ident = nextSettings?.identity;
                if (ident && typeof ident === 'object') {
                    const user_name = typeof ident.user_name === 'string' ? ident.user_name : identityRef.current.user_name;
                    const assistant_name = typeof ident.assistant_name === 'string' ? ident.assistant_name : identityRef.current.assistant_name;
                    const next = { user_name, assistant_name };
                    identityRef.current = next;
                    setIdentity(next);
                }
            } catch (_) {
            }

            try {
                const enabled = nextSettings?.tts?.enabled;
                if (typeof enabled === 'boolean') {
                    ttsEnabledRef.current = enabled;
                    if (!enabled) {
                        stopTTSPlayback();
                    }
                }
            } catch (_) {
            }
        });

        socket.on('tts_chunk', (payload) => {
            if (!ttsEnabledRef.current) return;
            ttsStoppedRef.current = false;
            ttsEndedRef.current = false;
            enqueueTTSMp3Base64(payload?.audio, payload?.chunk_index);
        });

        socket.on('tts_end', () => {
            ttsEndedRef.current = true;
            setIsTyping(false);
        });

        socket.on('tts_stop', () => {
            stopTTSPlayback();
            setIsTyping(false);
        });

        socket.on('cad_data', (data) => {
            console.log("Received CAD Data:", data);
            setCadData(data);
            setCadThoughts('');
            setShowCadWindow(true);
            if (!elementPositions.cad) {
                const size = { w: 400, h: 400 };
                const clamped = clampToViewport({ x: window.innerWidth / 2 + 150, y: window.innerHeight / 2 }, size);
                setElementPositions(prev => ({ ...prev, cad: clamped }));
            }
        });
        socket.on('cad_status', (data) => {
            console.log("Received CAD Status:", data);
            if (data.attempt) {
                setCadRetryInfo({
                    attempt: data.attempt,
                    maxAttempts: data.max_attempts || 3,
                    error: data.error
                });
            }
            if (data.status === 'generating' || data.status === 'retrying') {
                setCadData({ format: 'loading' });
                setShowCadWindow(true);
                if (data.status === 'generating' && data.attempt === 1) {
                    setCadThoughts('');
                }
                if (!elementPositions.cad) {
                    const size = { w: 400, h: 400 };
                    const clamped = clampToViewport({ x: window.innerWidth / 2 + 150, y: window.innerHeight / 2 }, size);
                    setElementPositions(prev => ({ ...prev, cad: clamped }));
                }
            } else if (data.status === 'failed') {
                setCadData({ format: 'loading' });
            }
        });
        socket.on('cad_thought', (data) => {
            setCadThoughts(prev => prev + data.text);
        });

        socket.on('image_status', (data) => {
            if (data.status === 'generating') {
                setImageData({ status: 'loading' });
                setShowImageWindow(true);
                if (!elementPositions.image) {
                    const size = { w: 480, h: 420 };
                    const clamped = clampToViewport({ x: window.innerWidth / 2 - 150, y: window.innerHeight / 2 }, size);
                    setElementPositions(prev => ({ ...prev, image: clamped }));
                }
            }
        });
        socket.on('image_result', (data) => {
            setImageData(data);
            if (data.status !== 'error') {
                setShowImageWindow(true);
                if (!elementPositions.image) {
                    const size = { w: 480, h: 420 };
                    const clamped = clampToViewport({ x: window.innerWidth / 2 - 150, y: window.innerHeight / 2 }, size);
                    setElementPositions(prev => ({ ...prev, image: clamped }));
                }
            }
        });

        socket.on('browser_frame', (data) => {
            setBrowserData(prev => ({
                image: data.image,
                logs: [...prev.logs, data.log].filter(l => l).slice(-50)
            }));
            setShowBrowserWindow(true);
            if (!elementPositions.browser) {
                const size = { w: 550, h: 380 };
                const clamped = clampToViewport({ x: window.innerWidth / 2 - 200, y: window.innerHeight / 2 }, size);
                setElementPositions(prev => ({ ...prev, browser: clamped }));
            }
        });

        socket.on('transcription', (data) => {
            const sender = data?.sender;
            const rawText = data?.text ?? '';
            const text = rawText.trim ? rawText.trim() : String(rawText).trim();
            const error = data?.error;

            // ── Voice transcription result (no sender = user's spoken words) ──────
            // audio_transcribe in server.py emits without a sender field.
            // This must be treated as the user's message and added to the chat.
            if (!sender) {
                if (error && !text) {
                    console.error('[STT] Backend error:', error);
                    pushActivity(`Voice error: ${error}`);
                    return;
                }
                if (!text) return;
                // Dedup: process_text_input also emits a 'transcription' with
                // sender=user_name for the same text — ignore that duplicate below.
                // Here we guard against the no-sender copy arriving twice.
                if (text === lastSentUtteranceRef.current) return;
                lastSentUtteranceRef.current = text;

                if (isTTSPlayingRef.current) {
                    stopTTSPlayback();
                    safeEmit('stop_tts');
                }

                setMessages(prev => [
                    ...prev,
                    { sender: 'You', text, time: new Date().toLocaleTimeString() },
                ]);
                pushActivity('Voice message sent');
                return;
            }

            // ── Filter duplicate user-side messages from process_text_input ───────
            // jarvis.py emits on_transcription({"sender": user_name, "text": text})
            // at the start of process_text_input — we already showed the message above.
            if (sender === 'User' || sender === identityRef.current.user_name) {
                return;
            }

            // ── AI response streaming — accumulate all tokens into one bubble ─────
            if (!rawText) return;
            setIsTyping(true);
            const now = new Date().toLocaleTimeString();
            // Capture the active conversation ID at event time so the debounce save
            // always targets the right conversation, even if the user switches away.
            const convIdAtEventTime = activeConvIdRef.current;

            setMessages(prev => {
                const lastMsg = prev[prev.length - 1];
                let updated;
                if (lastMsg && lastMsg.sender === sender) {
                    // Append token to the existing AI bubble — never create a new one.
                    updated = {
                        ...lastMsg,
                        text: lastMsg.text + rawText,
                        _convId: lastMsg._convId ?? convIdAtEventTime,
                    };
                    // Setting a ref inside an updater is safe: Strict Mode runs the
                    // updater twice with the same `prev`, so both writes are identical.
                    lastAiMsgRef.current = updated;
                    return [...prev.slice(0, -1), updated];
                } else {
                    // First token for this AI turn — open a new bubble.
                    updated = { sender, text: rawText, time: now, _convId: convIdAtEventTime };
                    lastAiMsgRef.current = updated;
                    return [...prev, updated];
                }
            });

            // Debounce: persist the fully-merged message after 1 s of silence
            if (aiSaveTimerRef.current) clearTimeout(aiSaveTimerRef.current);
            aiSaveTimerRef.current = setTimeout(() => {
                const msg = lastAiMsgRef.current;
                const targetConvId = msg?._convId;
                if (msg && targetConvId && socket?.connected) {
                    socket.emit('add_conversation_message', {
                        conversation_id: targetConvId,
                        sender: msg.sender,
                        text: msg.text,
                        timestamp: new Date().toISOString(),
                    });
                    lastAiMsgRef.current = null;
                    aiSaveTimerRef.current = null;
                }
            }, 1000);
        });

        socket.on('tool_confirmation_request', (data) => {
            console.log("Received Confirmation Request:", data);
            setConfirmationRequest(data);
        });

        socket.on('tool_activity', (data) => {
            try {
                const tool = data?.tool ? String(data.tool) : 'tool';
                const evt = data?.event ? String(data.event) : 'event';
                const id = data?.id ? String(data.id).slice(0, 8) : null;
                const msg = data?.message ? String(data.message) : null;

                const summary = data?.summary ? String(data.summary) : null;

                const shortId = id || 'unknown';
                let line = `[TOOL] ${tool}: ${evt}`;
                if (shortId) line += ` (${shortId})`;
                if (evt === 'done' && summary) line += ` - ${summary}`;
                if (evt === 'error' && msg) line += ` - ${msg}`;

                const map = toolLogByIdRef.current;
                const order = toolLogOrderRef.current;
                const existing = map.get(shortId);
                if (!existing) {
                    map.set(shortId, line);
                    order.push(shortId);
                    if (order.length > 20) {
                        const drop = order.shift();
                        if (drop) map.delete(drop);
                    }
                } else {
                    map.set(shortId, line);
                }

                const merged = order.map(k => map.get(k)).filter(Boolean);
                setActivityLog(prev => {
                    const head = prev.slice(0, Math.max(0, prev.length - merged.length));
                    const next = [...head, ...merged];
                    return next.length > 80 ? next.slice(next.length - 80) : next;
                });
            } catch (_) {
            }
        });

        socket.on('project_update', (data) => {
            console.log("Project Update:", data.project);
            setCurrentProject(data.project);
            addMessage('System', `Switched to project: ${data.project}`);
        });

        // Get All Media Devices
        navigator.mediaDevices.enumerateDevices().then(devs => {
            const audioInputs = devs.filter(d => d.kind === 'audioinput');
            const audioOutputs = devs.filter(d => d.kind === 'audiooutput');
            const videoInputs = devs.filter(d => d.kind === 'videoinput');

            setMicDevices(audioInputs);
            setSpeakerDevices(audioOutputs);
            setWebcamDevices(videoInputs);

            const savedMicId = localStorage.getItem('selectedMicId');
            if (savedMicId && audioInputs.some(d => d.deviceId === savedMicId)) {
                setSelectedMicId(savedMicId);
            } else if (audioInputs.length > 0) {
                setSelectedMicId(audioInputs[0].deviceId);
            }

            const savedSpeakerId = localStorage.getItem('selectedSpeakerId');
            if (savedSpeakerId && audioOutputs.some(d => d.deviceId === savedSpeakerId)) {
                setSelectedSpeakerId(savedSpeakerId);
            } else if (audioOutputs.length > 0) {
                setSelectedSpeakerId(audioOutputs[0].deviceId);
            }

            const savedWebcamId = localStorage.getItem('selectedWebcamId');
            if (savedWebcamId && videoInputs.some(d => d.deviceId === savedWebcamId)) {
                setSelectedWebcamId(savedWebcamId);
            } else if (videoInputs.length > 0) {
                setSelectedWebcamId(videoInputs[0].deviceId);
            }
        });

        // Initialize Hand Landmarker with guards for React Strict Mode / Fast Refresh
        const initHandLandmarker = async () => {
            if (handLandmarkerInitializedRef.current || isInitializingRef.current) {
                return;
            }

            isInitializingRef.current = true;
            try {
                const landmarker = await getHandLandmarkerSingleton();
                if (handLandmarkerInitializedRef.current) {
                    isInitializingRef.current = false;
                    return;
                }
                handLandmarkerRef.current = landmarker;
                handLandmarkerInitializedRef.current = true;
                isInitializingRef.current = false;
                addMessage('System', 'Hand Tracking Ready');
            } catch (error) {
                isInitializingRef.current = false;
                console.error("[HandLandmarker] Failed to initialize:", error);
                addMessage('System', `Hand Tracking Error: ${error.message}`);
            }
        };
        initHandLandmarker();

        // Note: voice transcription and AI streaming are both handled by the
        // single 'transcription' socket.on listener registered above (line ~915).
        // A second listener here was the root cause of split/duplicate messages.

        // ── IPC backend-status push listener (Electron only) ─────────────────
        // Main process sends 'backend-status' the instant the backend becomes
        // ready or goes offline — no polling delay, no DevTools noise.
        let _backendStatusWrapper = null;
        if (electronIpc?.on) {
            _backendStatusWrapper = electronIpc.on('backend-status', (status) => {
                if (status === 'ready') {
                    _connectSocket();
                } else if (status === 'offline') {
                    setSocketConnected(false);
                    setStatus('Backend Offline');
                    _startProbe(); // fall back to poll until main pushes 'ready' again
                }
            });
        }

        // Immediate synchronous query — handles the race where the renderer
        // reloaded after main already sent the 'ready' event.
        if (electronIpc?.invoke) {
            electronIpc.invoke('get-backend-status')
                .then(status => { if (status === 'ready') _connectSocket(); })
                .catch(() => {});
        }

        // Start the interval probe as a belt-and-suspenders fallback for both
        // the initial cold-start and the browser (non-Electron) code path.
        _startProbe();

        return () => {
            // Socket cleanup
            socket.off('connect');
            socket.off('connect_error');
            socket.off('disconnect');
            socket.off('status');
            socket.off('audio_data');
            socket.off('settings');
            socket.off('tts_chunk');
            socket.off('tts_end');
            socket.off('tts_stop');
            socket.off('cad_data');
            socket.off('cad_status');
            socket.off('cad_thought');
            socket.off('browser_frame');
            socket.off('transcription');
            socket.off('tool_confirmation_request');
            socket.off('tool_activity');
            socket.off('error');
            socket.off('project_update');
            socket.off('conversations_list');
            socket.off('conversation_created');
            socket.off('conversation_loaded');
            socket.off('conversation_deleted');
            socket.off('image_status');
            socket.off('image_result');
            socket.off('browser_frame');

            stopTTSPlayback();
            stopPTT();
            
            // Note: HandLandmarker cleanup is intentionally not reset here
            // to allow the instance to persist across hot reloads.
            // Release camera hardware when the component unmounts.
            if (streamRef.current) {
                streamRef.current.getTracks().forEach(t => t.stop());
                streamRef.current = null;
            }
            if (reconnectResumeTimerRef.current) {
                clearTimeout(reconnectResumeTimerRef.current);
                reconnectResumeTimerRef.current = null;
            }
            if (backendProbeRef.current) {
                clearInterval(backendProbeRef.current);
                backendProbeRef.current = null;
            }
            if (_backendStatusWrapper && electronIpc?.removeListener) {
                electronIpc.removeListener('backend-status', _backendStatusWrapper);
            }
        };
    }, []);

    // Push-to-talk: start recording when mic is unmuted, stop when muted
    useEffect(() => {
        if (!isConnected) return;
        if (!isMuted) {
            startPTT();
        } else {
            stopPTT();
        }
    }, [isMuted, isConnected]);

    useEffect(() => {
        if (socket.connected) {
            setStatus('Connected');
            safeEmit('get_settings');
        }
    }, []);

    // Persist device selections
    useEffect(() => {
        if (selectedMicId) {
            localStorage.setItem('selectedMicId', selectedMicId);
            selectedMicIdRef.current = selectedMicId;
        }
    }, [selectedMicId]);

    useEffect(() => {
        if (selectedSpeakerId) localStorage.setItem('selectedSpeakerId', selectedSpeakerId);
    }, [selectedSpeakerId]);

    useEffect(() => {
        if (selectedWebcamId) localStorage.setItem('selectedWebcamId', selectedWebcamId);
    }, [selectedWebcamId]);

    // Start/Stop Mic Visualizer
    useEffect(() => {
        if (selectedMicId) startMicVisualizer(selectedMicId);
    }, [selectedMicId]);

    // ── Push-to-talk helpers (faster-whisper via backend) ─────────────────────

    /**
     * Decode an audio blob and re-encode as 16 kHz mono 16-bit PCM WAV.
     * Uses OfflineAudioContext (purely in-memory, no hardware access) so it
     * never conflicts with the live AudioContext used by the mic visualizer.
     */
    const _blobToWav = async (blob) => {
        const arrayBuffer = await blob.arrayBuffer();

        // Step 1: Decode at the blob's native sample rate using a temporary live context.
        // We create it, decode, then immediately close it to free hardware resources.
        const decodeCtx = new (window.AudioContext || window.webkitAudioContext)();
        let decoded;
        try {
            // slice(0) forces a detached copy so decodeAudioData can take ownership safely
            decoded = await decodeCtx.decodeAudioData(arrayBuffer.slice(0));
        } finally {
            try { decodeCtx.close(); } catch (_) {}
        }

        if (!decoded || decoded.length === 0) throw new Error('decoded buffer is empty');

        // Step 2: Resample to 16 kHz mono via OfflineAudioContext (no device, no conflicts).
        const SR = 16000;
        const numSamples = Math.ceil(decoded.duration * SR);
        if (numSamples <= 0) throw new Error('audio duration is zero');

        const offCtx = new OfflineAudioContext(1, numSamples, SR);
        const src = offCtx.createBufferSource();
        src.buffer = decoded;
        src.connect(offCtx.destination);
        src.start(0);
        const rendered = await offCtx.startRendering();
        const samples = rendered.getChannelData(0);

        // Step 3: Write WAV header + int16 PCM samples
        const wavBuffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(wavBuffer);
        const writeStr = (off, str) => {
            for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i));
        };
        writeStr(0, 'RIFF');
        view.setUint32(4,  36 + samples.length * 2, true);
        writeStr(8, 'WAVE');
        writeStr(12, 'fmt ');
        view.setUint32(16, 16,       true);
        view.setUint16(20, 1,        true);  // PCM
        view.setUint16(22, 1,        true);  // mono
        view.setUint32(24, SR,       true);
        view.setUint32(28, SR * 2,   true);
        view.setUint16(32, 2,        true);
        view.setUint16(34, 16,       true);
        writeStr(36, 'data');
        view.setUint32(40, samples.length * 2, true);
        let off = 44;
        for (let i = 0; i < samples.length; i++) {
            const s = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            off += 2;
        }
        return wavBuffer;
    };

    const startPTT = async () => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') return;
        try {
            // CRITICAL: Stop the mic visualizer BEFORE opening a new getUserMedia stream.
            // Having two simultaneous captures on the same hardware device causes an
            // access violation (0xC0000005) and crashes the Electron renderer.
            stopMicVisualizer();

            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

            const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? 'audio/webm;codecs=opus'
                : MediaRecorder.isTypeSupported('audio/webm')
                    ? 'audio/webm'
                    : 'audio/ogg';
            pttMimeTypeRef.current = mimeType;
            pttChunksRef.current = [];

            // Capture current mic device id so the onstop callback can restart the visualizer
            const capturedMicId = selectedMicIdRef.current;

            const recorder = new MediaRecorder(stream, { mimeType });
            recorder.ondataavailable = (e) => {
                if (e.data && e.data.size > 0) pttChunksRef.current.push(e.data);
            };
            recorder.onstop = async () => {
                // Release the recording stream immediately
                stream.getTracks().forEach(t => t.stop());

                // Restart the visualizer now that the hardware is free
                if (capturedMicId) startMicVisualizer(capturedMicId);

                const chunks = pttChunksRef.current;
                pttChunksRef.current = [];
                if (!chunks.length) return;
                const rawBlob = new Blob(chunks, { type: pttMimeTypeRef.current });
                if (rawBlob.size < 1000) return;

                try {
                    const wavBuffer = await _blobToWav(rawBlob);
                    const uint8 = new Uint8Array(wavBuffer);
                    let binary = '';
                    const CHUNK = 8192;
                    for (let i = 0; i < uint8.length; i += CHUNK) {
                        binary += String.fromCharCode(...uint8.subarray(i, i + CHUNK));
                    }
                    safeEmit('audio_transcribe', { audio: btoa(binary), mime_type: 'audio/wav' });
                    pushActivity('Processing voice…');
                } catch (err) {
                    console.error('[PTT] encode/convert error:', err);
                    pushActivity(`Voice error: ${err.message}`);
                }
            };

            recorder.start(250);
            mediaRecorderRef.current = recorder;
            setIsRecording(true);
            pushActivity('Microphone active');
        } catch (err) {
            console.error('[PTT] start error:', err.name, err.message);
            pushActivity(`Mic error: ${err.name}`);
            // Restore visualizer if startPTT failed mid-way
            const mid = selectedMicIdRef.current;
            if (mid) startMicVisualizer(mid);
        }
    };

    const stopPTT = () => {
        const rec = mediaRecorderRef.current;
        if (!rec) return;
        try {
            if (rec.state !== 'inactive') rec.stop();
        } catch (_) {}
        mediaRecorderRef.current = null;
        setIsRecording(false);
    };
    // ──────────────────────────────────────────────────────────────────────────

    const startMicVisualizer = async (deviceId) => {
        // Clean up any existing resources first
        stopMicVisualizer();
        
        try {
            let stream;
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    audio: deviceId ? { deviceId: { exact: deviceId } } : true
                });
            } catch (e) {
                stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            }

            // Store stream reference for cleanup
            micStreamRef.current = stream;

            // Create new AudioContext
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            audioContextRef.current = new AudioContextClass();
            analyserRef.current = audioContextRef.current.createAnalyser();
            analyserRef.current.fftSize = 64;

            sourceRef.current = audioContextRef.current.createMediaStreamSource(stream);
            sourceRef.current.connect(analyserRef.current);

            const updateMicData = () => {
                // Guard: check if refs still exist (component may have unmounted)
                if (!analyserRef.current || !audioContextRef.current || audioContextRef.current.state === 'closed') {
                    return;
                }
                const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
                analyserRef.current.getByteFrequencyData(dataArray);
                const avg = dataArray.length ? dataArray.reduce((a, b) => a + b, 0) / dataArray.length : 0;
                micLevelRef.current = avg;
                setMicAudioData(Array.from(dataArray));
                animationFrameRef.current = requestAnimationFrame(updateMicData);
            };

            updateMicData();
        } catch (err) {
            console.error("[MicVisualizer] Error accessing microphone:", err);
        }
    };

    const stopMicVisualizer = () => {
        // Cancel animation frame first
        if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current);
            animationFrameRef.current = null;
        }
        
        // Disconnect source
        if (sourceRef.current) {
            try {
                sourceRef.current.disconnect();
            } catch (e) {
                // Source may already be disconnected
            }
            sourceRef.current = null;
        }
        
        // Stop mic stream tracks
        if (micStreamRef.current) {
            try {
                micStreamRef.current.getTracks().forEach(track => track.stop());
            } catch (e) {
                // Stream may already be stopped
            }
            micStreamRef.current = null;
        }
        
        // Close AudioContext only if not already closed
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
            try {
                audioContextRef.current.close();
            } catch (e) {
                console.warn('[AudioContext] Error closing:', e.message);
            }
        }
        audioContextRef.current = null;
        analyserRef.current = null;
    };

    const startVideo = async () => {
        try {
            // Re-use the cached stream when it is still alive and the device hasn't changed.
            // This eliminates the OS-level cold-start delay on every toggle.
            if (streamRef.current?.active) {
                const liveTrack = streamRef.current.getVideoTracks()[0];
                const liveDeviceId = liveTrack?.getSettings()?.deviceId;
                const wantedDeviceId = selectedWebcamId || liveDeviceId;
                if (!selectedWebcamId || liveDeviceId === wantedDeviceId) {
                    if (videoRef.current) {
                        videoRef.current.srcObject = streamRef.current;
                        await videoRef.current.play().catch(() => {});
                    }
                    setIsVideoOn(true);
                    isVideoOnRef.current = true;
                    requestAnimationFrame(predictWebcam);
                    return;
                }
                // Device changed — release old stream and fall through to create a new one.
                streamRef.current.getTracks().forEach(t => t.stop());
                streamRef.current = null;
            }

            // 640×480 is the MediaPipe sweet spot for hand tracking:
            //  • The model's internal input tile is 224×224 — feeding 640×480
            //    gives the GPU preprocessor a 3× easier downsample than 1280×720.
            //  • Camera cold-start is faster and memory bandwidth is lower.
            //  • Cap frame rate at 30 fps — the human eye can't track fingers
            //    faster, and it halves GPU texture-upload work vs 60 fps.
            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    frameRate: { ideal: 30, max: 30 },
                    ...(selectedWebcamId ? { deviceId: { exact: selectedWebcamId } } : {}),
                },
            };

            let stream;
            try {
                stream = await navigator.mediaDevices.getUserMedia(constraints);
            } catch (e) {
                stream = await navigator.mediaDevices.getUserMedia({ video: true });
            }

            streamRef.current = stream;

            if (!transmissionCanvasRef.current) {
                transmissionCanvasRef.current = document.createElement('canvas');
                transmissionCanvasRef.current.width = 640;
                transmissionCanvasRef.current.height = 360;
            }

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                // Wait for the video element to be actually ready before starting the loop,
                // replacing the polling approach (readyState < 2) in predictWebcam.
                if (videoRef.current.readyState < 2) {
                    await new Promise(resolve =>
                        videoRef.current.addEventListener('canplay', resolve, { once: true })
                    );
                }
                await videoRef.current.play().catch(() => {});
            }

            setIsVideoOn(true);
            isVideoOnRef.current = true;
            requestAnimationFrame(predictWebcam);

        } catch (err) {
            console.error("Error accessing camera:", err);
            addMessage('System', 'Error accessing camera');
        }
    };

    const predictWebcam = () => {
        if (!videoRef.current || !canvasRef.current || !isVideoOnRef.current) return;

        if (videoRef.current.readyState < 2 || videoRef.current.videoWidth === 0 || videoRef.current.videoHeight === 0) {
            requestAnimationFrame(predictWebcam);
            return;
        }

        const ctx = canvasRef.current.getContext('2d');

        if (canvasRef.current.width !== videoRef.current.videoWidth || canvasRef.current.height !== videoRef.current.videoHeight) {
            canvasRef.current.width = videoRef.current.videoWidth;
            canvasRef.current.height = videoRef.current.videoHeight;
        }

        ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);

        if (isConnected) {
            if (frameCountRef.current % 5 === 0) {
                const transCanvas = transmissionCanvasRef.current;
                if (transCanvas) {
                    const transCtx = transCanvas.getContext('2d');
                    transCtx.drawImage(videoRef.current, 0, 0, transCanvas.width, transCanvas.height);
                    transCanvas.toBlob((blob) => {
                        if (blob) safeEmit('video_frame', { image: blob });
                    }, 'image/jpeg', 0.6);
                }
            }
        }

        let startTimeMs = performance.now();
        // With runningMode="VIDEO" the tracker maintains internal state between
        // calls, so calling every frame is efficient (~5–15 ms/frame on GPU).
        // We still skip every other rAF tick to stay well under the 16 ms frame
        // budget on lower-end hardware.  The canvas draw above runs every tick so
        // the video overlay remains smooth at the display's full refresh rate.
        inferenceFrameSkipRef.current = (inferenceFrameSkipRef.current + 1) % 2;
        const shouldRunInference =
            inferenceFrameSkipRef.current === 0 &&
            isHandTrackingEnabledRef.current &&
            handLandmarkerRef.current &&
            videoRef.current.currentTime !== lastVideoTimeRef.current;

        if (shouldRunInference) {
            lastVideoTimeRef.current = videoRef.current.currentTime;
            const results = handLandmarkerRef.current.detectForVideo(videoRef.current, startTimeMs);

            if (results.landmarks && results.landmarks.length > 0) {
                const landmarks = results.landmarks[0];
                const indexTip = landmarks[8];
                const thumbTip = landmarks[4];

                const SENSITIVITY = cursorSensitivityRef.current;
                const rawX = isCameraFlippedRef.current ? indexTip.x : (1 - indexTip.x);

                let normX = (rawX - 0.5) * SENSITIVITY + 0.5;
                normX = Math.max(0, Math.min(1, normX));

                let normY = (indexTip.y - 0.5) * SENSITIVITY + 0.5;
                normY = Math.max(0, Math.min(1, normY));

                const targetX = normX * window.innerWidth;
                const targetY = normY * window.innerHeight;

                const lowPass = (prev, value, alpha) => alpha * value + (1 - alpha) * prev;
                const alphaForCutoff = (cutoff, dt) => {
                    const tau = 1 / (2 * Math.PI * cutoff);
                    return 1 / (1 + tau / dt);
                };

                const state = oneEuroRef.current;
                let dt = state.lastTimeMs ? (startTimeMs - state.lastTimeMs) / 1000 : (1 / 60);
                if (!Number.isFinite(dt) || dt <= 0) dt = 1 / 60;
                dt = Math.min(dt, 0.2);

                const minCutoff = 1.0;
                const beta = 1.2;
                const dCutoff = 1.0;

                if (!state.initialized) {
                    state.x = targetX;
                    state.y = targetY;
                    state.dx = 0;
                    state.dy = 0;
                    state.initialized = true;
                } else {
                    const dx = (targetX - state.x) / dt;
                    const dy = (targetY - state.y) / dt;

                    const aD = alphaForCutoff(dCutoff, dt);
                    state.dx = lowPass(state.dx, dx, aD);
                    state.dy = lowPass(state.dy, dy, aD);

                    const cutoffX = minCutoff + beta * Math.abs(state.dx);
                    const cutoffY = minCutoff + beta * Math.abs(state.dy);
                    const aX = alphaForCutoff(cutoffX, dt);
                    const aY = alphaForCutoff(cutoffY, dt);

                    state.x = lowPass(state.x, targetX, aX);
                    state.y = lowPass(state.y, targetY, aY);
                }

                state.lastTimeMs = startTimeMs;
                smoothedCursorPosRef.current.x = state.x;
                smoothedCursorPosRef.current.y = state.y;

                let finalX = smoothedCursorPosRef.current.x;
                let finalY = smoothedCursorPosRef.current.y;

                const SNAP_THRESHOLD = 50;
                const UNSNAP_THRESHOLD = 100;

                if (snapStateRef.current.isSnapped) {
                    const dist = Math.sqrt(
                        Math.pow(finalX - snapStateRef.current.snapPos.x, 2) +
                        Math.pow(finalY - snapStateRef.current.snapPos.y, 2)
                    );

                    if (dist > UNSNAP_THRESHOLD) {
                        if (snapStateRef.current.element) {
                            snapStateRef.current.element.classList.remove('snap-highlight');
                            snapStateRef.current.element.style.boxShadow = '';
                            snapStateRef.current.element.style.backgroundColor = '';
                            snapStateRef.current.element.style.borderColor = '';
                        }
                        snapStateRef.current = { isSnapped: false, element: null, snapPos: { x: 0, y: 0 } };
                    } else {
                        finalX = snapStateRef.current.snapPos.x;
                        finalY = snapStateRef.current.snapPos.y;
                    }
                } else {
                    // Use pre-built cache instead of querySelectorAll on every frame.
                    const targets = snapTargetsRef.current;
                    let closest = null;
                    let minDist = Infinity;

                    for (const entry of targets) {
                        const dist = Math.sqrt(
                            Math.pow(finalX - entry.centerX, 2) + Math.pow(finalY - entry.centerY, 2)
                        );
                        if (dist < minDist) {
                            minDist = dist;
                            closest = entry;
                        }
                    }

                    if (closest && minDist < SNAP_THRESHOLD) {
                        snapStateRef.current = {
                            isSnapped: true,
                            element: closest.el,
                            snapPos: { x: closest.centerX, y: closest.centerY }
                        };
                        finalX = closest.centerX;
                        finalY = closest.centerY;

                        closest.el.classList.add('snap-highlight');
                        closest.el.style.boxShadow = '0 0 20px rgba(34, 211, 238, 0.6)';
                        closest.el.style.backgroundColor = 'rgba(6, 182, 212, 0.2)';
                        closest.el.style.borderColor = 'rgba(34, 211, 238, 1)';
                    }
                }

                // Write cursor position directly to the DOM — no React state update,
                // no re-render of the full App tree on every animation frame.
                if (cursorDivRef.current) {
                    cursorDivRef.current.style.left = finalX + 'px';
                    cursorDivRef.current.style.top  = finalY + 'px';
                }

                const distance = Math.sqrt(
                    Math.pow(indexTip.x - thumbTip.x, 2) + Math.pow(indexTip.y - thumbTip.y, 2)
                );

                const isPinchNow = distance < 0.05;
                // Only fire a React state update when the pinch state actually changes,
                // not 60 times per second regardless.
                if (isPinchNow !== prevIsPinchRef.current) {
                    prevIsPinchRef.current = isPinchNow;
                    if (isPinchNow) {
                        const el = document.elementFromPoint(finalX, finalY);
                        if (el) {
                            const clickable = el.closest('button, input, a, [role="button"]');
                            if (clickable && typeof clickable.click === 'function') {
                                clickable.click();
                            } else if (typeof el.click === 'function') {
                                el.click();
                            }
                        }
                    }
                    setIsPinching(isPinchNow);
                }

                const isFingerFolded = (tipIdx, mcpIdx) => {
                    const tip = landmarks[tipIdx];
                    const mcp = landmarks[mcpIdx];
                    const wrist = landmarks[0];
                    const distTip = Math.sqrt(Math.pow(tip.x - wrist.x, 2) + Math.pow(tip.y - wrist.y, 2));
                    const distMcp = Math.sqrt(Math.pow(mcp.x - wrist.x, 2) + Math.pow(mcp.y - wrist.y, 2));
                    return distTip < distMcp;
                };

                const isFist = isFingerFolded(8, 5) && isFingerFolded(12, 9) && isFingerFolded(16, 13) && isFingerFolded(20, 17);

                const wrist = landmarks[0];
                const wristRawX = isCameraFlippedRef.current ? wrist.x : (1 - wrist.x);
                const wristNormX = Math.max(0, Math.min(1, (wristRawX - 0.5) * SENSITIVITY + 0.5));
                const wristNormY = Math.max(0, Math.min(1, (wrist.y - 0.5) * SENSITIVITY + 0.5));
                const wristScreenX = wristNormX * window.innerWidth;
                const wristScreenY = wristNormY * window.innerHeight;

                if (isFist) {
                    if (!activeDragElementRef.current) {
                        const draggableElements = ['cad', 'browser'];

                        for (const id of draggableElements) {
                            const el = document.getElementById(id);
                            if (el) {
                                const rect = el.getBoundingClientRect();
                                if (finalX >= rect.left && finalX <= rect.right && finalY >= rect.top && finalY <= rect.bottom) {
                                    activeDragElementRef.current = id;
                                    bringToFront(id);
                                    lastWristPosRef.current = { x: wristScreenX, y: wristScreenY };
                                    break;
                                }
                            }
                        }
                    }

                    if (activeDragElementRef.current) {
                        const dx = wristScreenX - lastWristPosRef.current.x;
                        const dy = wristScreenY - lastWristPosRef.current.y;

                        if (Math.abs(dx) > 0.5 || Math.abs(dy) > 0.5) {
                            updateElementPosition(activeDragElementRef.current, dx, dy);
                        }

                        lastWristPosRef.current = { x: wristScreenX, y: wristScreenY };
                    }
                } else {
                    activeDragElementRef.current = null;
                }

                if (activeDragElementRef.current !== lastActiveDragElementRef.current) {
                    setActiveDragElement(activeDragElementRef.current);
                    lastActiveDragElementRef.current = activeDragElementRef.current;
                }

                lastCursorPosRef.current = { x: finalX, y: finalY };
                drawSkeleton(ctx, landmarks);
            }
        }

        const now = performance.now();
        frameCountRef.current++;
        if (now - lastFrameTimeRef.current >= 1000) {
            setFps(frameCountRef.current);
            frameCountRef.current = 0;
            lastFrameTimeRef.current = now;
        }

        if (isVideoOnRef.current) requestAnimationFrame(predictWebcam);
    };

    const drawSkeleton = (ctx, landmarks) => {
        ctx.strokeStyle = '#00FFFF';
        ctx.lineWidth = 2;

        const connections = HandLandmarker.HAND_CONNECTIONS;
        for (const connection of connections) {
            const start = landmarks[connection.start];
            const end = landmarks[connection.end];
            ctx.beginPath();
            ctx.moveTo(start.x * canvasRef.current.width, start.y * canvasRef.current.height);
            ctx.lineTo(end.x * canvasRef.current.width, end.y * canvasRef.current.height);
            ctx.stroke();
        }
    };

    const stopVideo = () => {
        // Detach the stream from the video element but keep it alive in streamRef
        // so the next startVideo() can reuse it without a hardware cold-start.
        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }
        setIsVideoOn(false);
        isVideoOnRef.current = false;
        setFps(0);
    };

    const toggleVideo = () => {
        if (isVideoOn) stopVideo();
        else startVideo();
    };

    const msgSeqRef = useRef(0);
    const addMessage = (sender, text) => {
        const time = new Date().toLocaleTimeString();
        const isoTime = new Date().toISOString();
        // Stable id = prevents duplicate keys and lets AnimatePresence skip re-animation.
        const id = `msg_${++msgSeqRef.current}_${Date.now()}`;

        // Pure state update — NO side-effects inside the updater (Strict Mode double-invokes it)
        setMessages(prev => [...prev, { id, sender, text, time }]);

        // Side-effects outside the updater: emit exactly once per call
        if (sender !== 'System' && socket?.connected && activeConvIdRef.current) {
            const convId = activeConvIdRef.current;
            socket.emit('add_conversation_message', {
                conversation_id: convId,
                sender,
                text,
                timestamp: isoTime,
            });
            // Auto-title: fire only on the very first message of a new conversation
            if (isNewConvRef.current) {
                isNewConvRef.current = false;
                const autoTitle = text.slice(0, 48).trim() + (text.length > 48 ? '…' : '');
                socket.emit('update_conversation_title', { id: convId, title: autoTitle });
                setConversations(cs =>
                    cs.map(c => c.id === convId ? { ...c, title: autoTitle } : c)
                );
            }
        }
    };

    const clearHistory = useCallback(() => {
        if (socket?.connected) {
            socket.emit('clear_chat_history');
        } else {
            setMessages([]);
        }
    }, [socket]);

    // Immediately save any pending AI message before switching conversations,
    // using the conversation ID that was captured when the message arrived.
    const flushPendingAiMessage = useCallback(() => {
        if (aiSaveTimerRef.current) {
            clearTimeout(aiSaveTimerRef.current);
            aiSaveTimerRef.current = null;
        }
        const msg = lastAiMsgRef.current;
        const targetConvId = msg?._convId;
        if (msg && targetConvId && socket?.connected) {
            socket.emit('add_conversation_message', {
                conversation_id: targetConvId,
                sender: msg.sender,
                text: msg.text,
                timestamp: new Date().toISOString(),
            });
        }
        lastAiMsgRef.current = null;
    }, [socket]);

    const createNewConversation = useCallback(() => {
        if (socket?.connected) {
            flushPendingAiMessage();
            socket.emit('create_conversation', { title: 'New Chat' });
        }
    }, [socket, flushPendingAiMessage]);

    const loadConversation = useCallback((id) => {
        if (socket?.connected && id !== activeConvIdRef.current) {
            flushPendingAiMessage();
            socket.emit('load_conversation', { id });
        }
    }, [socket, flushPendingAiMessage]);

    const renameConversation = useCallback((id, title) => {
        if (socket?.connected) {
            socket.emit('update_conversation_title', { id, title });
            setConversations(cs => cs.map(c => c.id === id ? { ...c, title } : c));
        }
    }, [socket]);

    const deleteConversation = useCallback((id) => {
        if (socket?.connected) {
            socket.emit('delete_conversation', { id });
        }
    }, [socket]);

    // ── Stable ConsoleChat send callbacks (ref-of-latest-fn pattern) ────────
    // handleSend / handleSendButton close over inputValue and other frequently-
    // changing values, so they can't be safely wrapped in useCallback directly.
    // Instead, we keep a ref pointing at the latest implementation and expose a
    // stable wrapper — its reference never changes, satisfying React.memo.
    const stableHandleSend       = useCallback((e) => _handleSendImpl.current?.(e), []);
    const stableHandleSendButton = useCallback(() => _handleSendButtonImpl.current?.(), []);
    // ────────────────────────────────────────────────────────────────────────

    // ── Stable NavigationRail toggle callbacks ──────────────────────────────
    // These use the functional-updater form of setState (v => !v) so they never
    // capture stale state.  Empty dep arrays mean the references never change,
    // which is what allows React.memo(NavigationRail) to skip re-renders.
    const handleToggleHand              = useCallback(() => setIsHandTrackingEnabled(v => !v), []);
    const handleToggleCad               = useCallback(() => setShowCadWindow(v => !v), []);
    const handleToggleBrowser           = useCallback(() => setShowBrowserWindow(v => !v), []);
    const handleToggleImage             = useCallback(() => setShowImageWindow(v => !v), []);
    const handleToggleSettings          = useCallback(() => setShowSettings(v => !v), []);
    const handleToggleConversationSidebar = useCallback(() => setShowConversationSidebar(v => !v), []);
    const handleOpenCommandPalette      = useCallback(() => setCommandPaletteOpen(true), []);
    // ────────────────────────────────────────────────────────────────────────

    const handleFileUpload = useCallback((e) => {
        const file = e?.target?.files?.[0];
        if (!file) return;

        const reader = new FileReader();

        reader.onload = (event) => {
            try {
                const textContent = event?.target?.result;
                if (typeof textContent === 'string' && textContent.length > 0) {
                    if (socket && socket.connected) {
                        try {
                            socket.emit('upload_memory', { memory: textContent });
                            addMessage('System', 'Uploading memory...');
                        } catch (_) {
                            addMessage('System', 'Backend Offline - could not upload memory');
                            setStatus('Backend Offline');
                            setSocketConnected(false);
                        }
                    } else {
                        addMessage('System', 'Backend Offline - could not upload memory');
                        setStatus('Backend Offline');
                        setSocketConnected(false);
                    }
                } else {
                    addMessage('System', 'Empty or invalid memory file');
                }
            } catch (err) {
                console.error('Error reading file:', err);
                addMessage('System', 'Error reading memory file');
            } finally {
                try {
                    if (e?.target) e.target.value = '';
                } catch (_) {
                }
            }
        };

        reader.onerror = () => {
            addMessage('System', 'Error reading memory file');
            try {
                if (e?.target) e.target.value = '';
            } catch (_) {
            }
        };

        reader.readAsText(file);
    }, []);

    const togglePower = () => {
        if (isConnected) {
            safeEmit('stop_audio');
            setIsConnected(false);
            setIsMuted(false);
            pushActivity('Session stopped');
        } else {
            const selected = micDevices.find(d => d.deviceId === selectedMicId);
            const deviceName = selected?.label || null;
            const index = micDevices.findIndex(d => d.deviceId === selectedMicId);
            safeEmit('start_audio', {
                device_name: deviceName,
                device_index: index >= 0 ? index : null,
                muted: false,
            });
            setIsConnected(true);
            setIsMuted(false);
            pushActivity('Session started');
        }
    };

    const toggleMute = () => {
        if (!isConnected) return;

        if (isMuted) {
            setIsMuted(false);
            pushActivity('Microphone unmuted');
        } else {
            stopPTT();
            setIsMuted(true);
            pushActivity('Microphone muted');
        }
    };

    const handleSend = (e) => {
        if (e.key === 'Enter' && inputValue.trim()) {
            const raw = String(inputValue || '').trim();
            if (raw.toLowerCase().startsWith('/feedback')) {
                const parts = raw.split(' ').filter(Boolean);
                const outcome = (parts[1] || '').toLowerCase();
                const note = parts.length > 2 ? parts.slice(2).join(' ') : null;

                safeEmit('recommendation_feedback', {
                    outcome,
                    rec_id: lastRecommendationIdRef.current,
                    note,
                });
                addMessage('You', raw);
                pushActivity('Feedback sent');
                setInputValue('');
                return;
            }

            if (raw.toLowerCase() === '/learning') {
                safeEmit('get_learning_summary', {});
                addMessage('You', raw);
                pushActivity('Learning summary requested');
                setInputValue('');
                return;
            }

            safeEmit('user_input', { text: raw });
            addMessage('You', raw);
            pushActivity('User message sent');
            setInputValue('');
        }
    };

    const handleSendButton = () => {
        if (!inputValue.trim()) return;
        const raw = String(inputValue || '').trim();
        if (raw.toLowerCase().startsWith('/feedback')) {
            const parts = raw.split(' ').filter(Boolean);
            const outcome = (parts[1] || '').toLowerCase();
            const note = parts.length > 2 ? parts.slice(2).join(' ') : null;
            safeEmit('recommendation_feedback', { outcome, rec_id: lastRecommendationIdRef.current, note });
            addMessage('You', raw);
            pushActivity('Feedback sent');
            setInputValue('');
            return;
        }
        if (raw.toLowerCase() === '/learning') {
            safeEmit('get_learning_summary', {});
            addMessage('You', raw);
            pushActivity('Learning summary requested');
            setInputValue('');
            return;
        }
        safeEmit('user_input', { text: raw });
        addMessage('You', raw);
        pushActivity('User message sent');
        setInputValue('');
    };

    // Sync ref-of-latest-fn on every render so the stable wrappers always
    // dispatch to the freshest implementation (correct closures, no stale state).
    _handleSendImpl.current       = handleSend;
    _handleSendButtonImpl.current = handleSendButton;

    const handleMinimize = () => ipcRenderer?.send('window-minimize');
    const handleMaximize = () => ipcRenderer?.send('window-maximize');

    const closeWindow = () => {
        try {
            ipcRenderer?.send('window-close');
        } catch (_) {
        }
    };

    const handleCloseRequest = () => {
        try {
            if (socket && socket.connected) {
                safeEmit('shutdown', {});
                setTimeout(() => closeWindow(), 300);
                return;
            }
        } catch (_) {
        }
        closeWindow();
    };

    const handleConfirmTool = () => {
        if (confirmationRequest) {
            safeEmit('confirm_tool', { id: confirmationRequest.id, confirmed: true });
            setConfirmationRequest(null);
        }
    };

    const handleDenyTool = () => {
        if (confirmationRequest) {
            safeEmit('confirm_tool', { id: confirmationRequest.id, confirmed: false });
            setConfirmationRequest(null);
        }
    };

    const updateElementPosition = (id, dx, dy) => {
        setElementPositions(prev => {
            const currentPos = prev[id];
            const size = elementSizes[id] || { w: 100, h: 100 };
            let newX = currentPos.x + dx;
            let newY = currentPos.y + dy;

            const width = window.innerWidth;
            const height = window.innerHeight;
            const margin = 0;

            if (id === 'chat') {
                newX = Math.max(size.w / 2 + margin, Math.min(width - size.w / 2 - margin, newX));
                newY = Math.max(margin, Math.min(height - size.h - margin, newY));
            } else if (id === 'video') {
                newX = Math.max(margin, Math.min(width - size.w - margin, newX));
                newY = Math.max(margin, Math.min(height - size.h - margin, newY));
            } else {
                newX = Math.max(size.w / 2 + margin, Math.min(width - size.w / 2 - margin, newX));
                newY = Math.max(size.h / 2 + margin, Math.min(height - size.h / 2 - margin, newY));
            }

            return { ...prev, [id]: { x: newX, y: newY } };
        });
    };

    const clampDraggedPosition = (id, rawX, rawY) => {
        const size = (elementSizesRef.current && elementSizesRef.current[id]) || { w: 100, h: 100 };
        let newX = rawX;
        let newY = rawY;

        const width = window.innerWidth;
        const height = window.innerHeight;
        const margin = 0;

        if (id === 'chat') {
            newX = Math.max(size.w / 2 + margin, Math.min(width - size.w / 2 - margin, newX));
            newY = Math.max(margin, Math.min(height - size.h - margin, newY));
        } else if (id === 'video') {
            newX = Math.max(margin, Math.min(width - size.w - margin, newX));
            newY = Math.max(margin, Math.min(height - size.h - margin, newY));
        } else {
            newX = Math.max(size.w / 2 + margin, Math.min(width - size.w / 2 - margin, newX));
            newY = Math.max(size.h / 2 + margin, Math.min(height - size.h / 2 - margin, newY));
        }

        return { x: newX, y: newY };
    };

    const stopMouseDragLoop = () => {
        if (dragRafIdRef.current != null) {
            cancelAnimationFrame(dragRafIdRef.current);
            dragRafIdRef.current = null;
        }
        dragLastFrameMsRef.current = 0;
    };

    const startMouseDragLoop = () => {
        if (dragRafIdRef.current != null) return;

        const tick = (nowMs) => {
            if (!isDraggingRef.current || !activeDragElementRef.current) {
                stopMouseDragLoop();
                return;
            }

            const id = activeDragElementRef.current;
            const lastMs = dragLastFrameMsRef.current || nowMs;
            const dtMs = Math.max(0, nowMs - lastMs);
            dragLastFrameMsRef.current = nowMs;

            const target = dragTargetPosRef.current;
            const cur = dragCurrentPosRef.current;

            const tauMs = 45;
            const alpha = 1 - Math.exp(-dtMs / tauMs);

            const nextX = cur.x + (target.x - cur.x) * alpha;
            const nextY = cur.y + (target.y - cur.y) * alpha;
            const clamped = clampDraggedPosition(id, nextX, nextY);

            dragCurrentPosRef.current = clamped;

            setElementPositions(prev => {
                const prevPos = prev[id];
                if (!prevPos) return prev;
                if (Math.abs(prevPos.x - clamped.x) < 0.01 && Math.abs(prevPos.y - clamped.y) < 0.01) {
                    return prev;
                }
                return { ...prev, [id]: { x: clamped.x, y: clamped.y } };
            });

            dragRafIdRef.current = requestAnimationFrame(tick);
        };

        dragRafIdRef.current = requestAnimationFrame(tick);
    };

    const handleMouseDown = (e, id) => {
        const fixedElements = ['visualizer', 'chat', 'video', 'tools'];
        if (fixedElements.includes(id)) return;

        bringToFront(id);

        const tagName = e.target.tagName.toLowerCase();
        if (tagName === 'input' || tagName === 'button' || tagName === 'textarea' || tagName === 'canvas' || e.target.closest('button')) {
            return;
        }

        const isDragHandle = e.target.closest('[data-drag-handle]');
        if (!isDragHandle && !isModularModeRef.current) return;

        const elPos = elementPositions[id];
        if (!elPos) return;

        dragOffsetRef.current = { x: e.clientX - elPos.x, y: e.clientY - elPos.y };

        setActiveDragElement(id);
        activeDragElementRef.current = id;
        isDraggingRef.current = true;

        dragCurrentPosRef.current = { x: elPos.x, y: elPos.y };
        const rawX = e.clientX - dragOffsetRef.current.x;
        const rawY = e.clientY - dragOffsetRef.current.y;
        const clamped = clampDraggedPosition(id, rawX, rawY);
        dragTargetPosRef.current = clamped;

        startMouseDragLoop();

        window.addEventListener('mousemove', handleMouseDrag, { passive: true });
        window.addEventListener('mouseup', handleMouseUp, { passive: true });
    };

    const handleMouseDrag = (e) => {
        if (!isDraggingRef.current || !activeDragElementRef.current) return;

        const id = activeDragElementRef.current;
        const rawNewX = e.clientX - dragOffsetRef.current.x;
        const rawNewY = e.clientY - dragOffsetRef.current.y;
        dragTargetPosRef.current = clampDraggedPosition(id, rawNewX, rawNewY);
    };

    const handleMouseUp = () => {
        isDraggingRef.current = false;
        setActiveDragElement(null);
        activeDragElementRef.current = null;
        stopMouseDragLoop();
        window.removeEventListener('mousemove', handleMouseDrag);
        window.removeEventListener('mouseup', handleMouseUp);
    };

    const audioAmp = aiAudioData.length ? aiAudioData.reduce((a, b) => a + b, 0) / aiAudioData.length / 255 : 0;

    const commandActions = [
        {
            id: 'toggle-settings',
            label: showSettings ? 'Close Settings' : 'Open Settings',
            shortcut: 'Ctrl+,',
            run: () => setShowSettings(v => !v),
        },
        {
            id: 'toggle-cad',
            label: showCadWindow ? 'Hide CAD' : 'Show CAD',
            shortcut: 'C',
            run: () => setShowCadWindow(v => !v),
        },
        {
            id: 'toggle-browser',
            label: showBrowserWindow ? 'Hide Web Agent' : 'Show Web Agent',
            shortcut: 'B',
            run: () => setShowBrowserWindow(v => !v),
        },
        {
            id: 'toggle-image',
            label: showImageWindow ? 'Hide Image Creator' : 'Show Image Creator',
            shortcut: 'I',
            run: () => setShowImageWindow(v => !v),
        },
        {
            id: 'toggle-session',
            label: isConnected ? 'Stop Session' : 'Start Session',
            shortcut: 'P',
            run: () => togglePower(),
        },
        {
            id: 'toggle-mute',
            label: isMuted ? 'Unmute Mic' : 'Mute Mic',
            shortcut: 'M',
            run: () => toggleMute(),
        },
    ];

    return (
        <div className="h-screen w-screen bg-premium-dark text-[var(--text-1)] overflow-hidden flex flex-col relative font-sans hex-grid holographic-overlay">

            {/* Boot sequence overlay — fades out after 1.8 s, pointer-events:none so it never blocks UI */}
            <AnimatePresence>
                {isBooting && (
                    <motion.div
                        key="boot"
                        initial={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.5 }}
                        className="fixed inset-0 z-[300] bg-[#010409] flex flex-col items-center justify-center gap-6 pointer-events-none"
                    >
                        {/* Arc decoration */}
                        <svg width="80" height="80" viewBox="0 0 80 80" aria-hidden="true">
                            <circle cx="40" cy="40" r="34" fill="none" stroke="rgba(92,245,255,0.1)" strokeWidth="1" />
                            <circle cx="40" cy="40" r="34" fill="none" stroke="rgba(92,245,255,0.6)" strokeWidth="1"
                                strokeDasharray="64 150" strokeLinecap="round" transform="rotate(-90 40 40)" />
                            <circle cx="40" cy="40" r="24" fill="none" stroke="rgba(92,245,255,0.15)" strokeWidth="1" />
                            <circle cx="40" cy="40" r="24" fill="none" stroke="rgba(92,245,255,0.4)" strokeWidth="1"
                                strokeDasharray="40 110" strokeLinecap="round" transform="rotate(30 40 40)" />
                            <circle cx="40" cy="40" r="5" fill="rgba(92,245,255,0.7)" />
                        </svg>
                        <div className="text-hud text-[var(--accent)] text-2xl tracking-[0.6em]">J.A.R.V.I.S</div>
                        <div className="text-hud text-[var(--text-3)] text-[11px] tracking-[0.4em]">INITIALIZING SYSTEMS</div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Hand Cursor — position written directly via cursorDivRef to avoid 20 fps App re-renders */}
            {isVideoOn && isHandTrackingEnabled && (
                <div
                    ref={cursorDivRef}
                    className={`fixed w-6 h-6 border-2 rounded-full pointer-events-none z-[100] transition-transform duration-75 ${isPinching ? 'bg-cyan-400 border-cyan-400 scale-75 shadow-[0_0_20px_rgba(92,245,255,0.8)]' : 'border-cyan-400 shadow-[0_0_12px_rgba(92,245,255,0.4)]'}`}
                    style={{ left: 0, top: 0, transform: 'translate(-50%, -50%)' }}
                >
                    <div className="absolute top-1/2 left-1/2 w-1.5 h-1.5 bg-white rounded-full -translate-x-1/2 -translate-y-1/2 shadow-sm" />
                </div>
            )}

            {/* Background */}
            <div
                className="absolute inset-0 z-0 pointer-events-none transition-opacity duration-1000"
                style={{
                    background: 'radial-gradient(1000px circle at 50% 30%, rgba(10, 15, 29, 0.4), rgba(1, 4, 9, 1) 80%)'
                }}
            />
            <div className="noise-overlay absolute inset-0 opacity-[0.02] z-0 pointer-events-none mix-blend-overlay" />

            {/* Ambient Glow — static radial gradient, no blur filter, no animation.
                blur-[160px] on an 800×800 div with animate-pulse caused an expensive
                filter repaint on every pulse frame. A static gradient is free. */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[700px] rounded-full pointer-events-none"
                style={{ background: 'radial-gradient(circle, rgba(92,245,255,0.03) 0%, transparent 70%)' }} />

            {/* ── TOP BAR ─────────────────────────────────────────────────────── */}
            <div
                className="z-50 flex items-center justify-between px-4 py-2 border-b border-white/5 bg-premium-dark/80 backdrop-blur-2xl select-none sticky top-0 shadow-2xl"
                style={{ WebkitAppRegion: 'drag' }}
            >
                {/* Left — brand + tags */}
                <div className="flex items-center gap-3 pl-1 shrink-0">
                    <h1 className="text-transparent bg-clip-text bg-gradient-to-r from-white via-cyan-400 to-white/50 text-hud hud-title leading-none">
                        J.A.R.V.I.S
                    </h1>
                    <div className="hidden sm:flex flex-col gap-0.5">
                        <span className="text-[8px] font-mono text-[var(--text-3)] tracking-[0.3em] leading-none">LOCAL AI ASSISTANT</span>
                        <span className="text-[8px] font-mono text-cyan-400/40 tracking-[0.2em] leading-none">
                            {String(currentProject || 'default').toUpperCase()}
                        </span>
                    </div>
                    <div className="text-[9px] font-mono text-cyan-400/50 border border-cyan-400/10 bg-cyan-400/5 px-2 py-0.5 rounded tracking-widest">
                        MODEL: QWEN
                    </div>
                </div>

                {/* Center — activity waveform */}
                <div className="flex-1 flex justify-center items-center mx-4 relative">
                    <div className={`transition-opacity duration-300 ${isMuted ? 'opacity-30' : 'opacity-100'}`}>
                        <TopAudioBar audioData={micAudioData} />
                    </div>
                    {isTyping && (
                        <div className="absolute right-0 flex items-center gap-1.5 text-[9px] font-mono text-cyan-400/70 tracking-widest">
                            <div className="w-1 h-1 rounded-full bg-cyan-400 animate-pulse" />
                            PROCESSING
                        </div>
                    )}
                </div>

                {/* Right — status + clock + window controls */}
                <div className="flex items-center gap-2 shrink-0" style={{ WebkitAppRegion: 'no-drag' }}>
                    {/* Backend status badge */}
                    <div className={`flex items-center gap-1.5 text-[9px] font-mono px-2 py-1 rounded border transition-colors ${
                        socketConnected
                            ? 'text-[#2ee59d] border-[rgba(46,229,157,0.2)] bg-[rgba(46,229,157,0.06)]'
                            : 'text-[#f85149] border-[rgba(248,81,73,0.2)] bg-[rgba(248,81,73,0.06)]'
                    }`}>
                        <div className={`w-1.5 h-1.5 rounded-full ${socketConnected ? 'bg-[#2ee59d] animate-pulse' : 'bg-[#f85149]'}`} />
                        {socketConnected ? 'ONLINE' : 'OFFLINE'}
                    </div>
                    <div className="text-[10px] font-mono text-[var(--text-3)] tabular-nums">
                        <LiveClock />
                    </div>
                    <div className="w-px h-4 bg-white/10 mx-1" />
                    <button onClick={handleMinimize} className="p-1.5 hover:bg-white/10 rounded text-white/50 transition-all hover:text-white/80">
                        <Minus size={14} />
                    </button>
                    <button onClick={handleMaximize} className="p-1.5 hover:bg-white/10 rounded text-white/50 transition-all hover:text-white/80">
                        <div className="w-[12px] h-[12px] border-[1.5px] border-current rounded-sm" />
                    </button>
                    <button onClick={handleCloseRequest} className="p-1.5 hover:bg-red-500/20 rounded text-white/40 hover:text-red-400 transition-all">
                        <X size={14} />
                    </button>
                </div>
            </div>

            {/* ── MAIN + FOOTER ──────────────────────────────────────────────── */}
            <div className="flex-1 relative z-10 flex flex-col min-h-0">

                {/* Row: nav + center + right panel */}
                <div className="flex flex-1 min-h-0 overflow-hidden">

                    {/* Navigation Rail */}
                    <NavigationRail
                        isConnected={isConnected}
                        isMuted={isMuted}
                        isVideoOn={isVideoOn}
                        isHandTrackingEnabled={isHandTrackingEnabled}
                        showCadWindow={showCadWindow}
                        showBrowserWindow={showBrowserWindow}
                        showImageWindow={showImageWindow}
                        showSettings={showSettings}
                        onTogglePower={togglePower}
                        onToggleMute={toggleMute}
                        onToggleVideo={toggleVideo}
                        onToggleHand={handleToggleHand}
                        onToggleCad={handleToggleCad}
                        onToggleBrowser={handleToggleBrowser}
                        onToggleImage={handleToggleImage}
                        onToggleSettings={handleToggleSettings}
                        onOpenCommandPalette={handleOpenCommandPalette}
                        showConversations={showConversationSidebar}
                        onToggleConversations={handleToggleConversationSidebar}
                    />

                    {/* Conversation Sidebar */}
                    <AnimatePresence>
                        {showConversationSidebar && (
                            <ConversationSidebar
                                conversations={conversations}
                                activeConversationId={activeConversationId}
                                onNew={createNewConversation}
                                onLoad={loadConversation}
                                onRename={renameConversation}
                                onDelete={deleteConversation}
                            />
                        )}
                    </AnimatePresence>

                    {/* ── CAMERA SIDE PANEL — collapses when camera is off ─── */}
                    {/* Hidden video element must always exist in the DOM */}
                    <video ref={videoRef} autoPlay muted className="absolute -left-[9999px] -top-[9999px] w-px h-px opacity-0" />

                    <AnimatePresence>
                        {(isVideoOn || isHandTrackingEnabled) && (
                            <motion.div
                                key="camera-panel"
                                initial={{ width: 0, opacity: 0 }}
                                animate={{ width: 360, opacity: 1 }}
                                exit={{ width: 0, opacity: 0 }}
                                transition={{ duration: 0.22, ease: 'easeInOut' }}
                                className="shrink-0 h-full border-r border-[rgba(255,255,255,0.06)] bg-[rgba(5,9,17,0.7)] flex flex-col overflow-hidden"
                            >
                                {/* Camera feed — locked to 16:9 so it never stretches into portrait */}
                                <div
                                    className="shrink-0 relative bg-black overflow-hidden w-full"
                                    style={{ aspectRatio: '16 / 9' }}
                                >
                                    <canvas
                                        ref={canvasRef}
                                        className="absolute inset-0 w-full h-full object-cover"
                                        style={{ transform: isCameraFlipped ? 'scaleX(-1)' : 'none' }}
                                    />

                                    {/* LIVE badge */}
                                    {isVideoOn && (
                                        <div className="absolute top-2 left-2 z-10 flex items-center gap-1.5 text-[9px] font-mono text-cyan-400/90 bg-[rgba(0,0,0,0.72)] backdrop-blur-md px-2 py-1 rounded border border-white/10 tracking-wider">
                                            <div className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
                                            LIVE
                                        </div>
                                    )}

                                    {/* Hand tracking badge */}
                                    {isHandTrackingEnabled && (
                                        <div className="absolute top-2 right-2 z-10 text-[9px] font-mono text-cyan-400/70 bg-[rgba(0,0,0,0.72)] backdrop-blur-md px-2 py-1 rounded border border-cyan-400/20 tracking-wider">
                                            TRACKING
                                        </div>
                                    )}
                                </div>

                                {/* Visualizer — fills remaining vertical space after camera */}
                                <div className="flex-1 min-h-0 flex items-center justify-center border-t border-[rgba(255,255,255,0.06)] bg-[rgba(4,7,15,0.55)]">
                                    <Visualizer
                                        audioData={aiAudioData}
                                        isListening={isConnected && !isMuted}
                                        intensity={audioAmp}
                                        width={352}
                                        height={100}
                                    />
                                </div>

                                {/* System log strip */}
                                <div className="shrink-0 px-3 py-2 border-t border-[rgba(255,255,255,0.06)] flex flex-col gap-0.5">
                                    <div className="flex items-center gap-1.5 mb-0.5">
                                        <div className="w-1 h-1 rounded-full bg-cyan-400/60 animate-pulse" />
                                        <span className="text-[8px] font-mono text-[var(--text-3)] tracking-widest">
                                            {String(currentProject || 'default').toUpperCase()}
                                        </span>
                                    </div>
                                    {systemLogs.slice(-2).map(log => (
                                        <div key={log.id} className="flex gap-1.5">
                                            <span className="text-[8px] font-mono text-cyan-400/25 shrink-0">[{log.timestamp}]</span>
                                            <span className="text-[8px] font-mono text-white/25 truncate">{log.msg}</span>
                                        </div>
                                    ))}
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>

                    {/* ── CENTER PANEL — CHAT IS THE HERO ──────────────────── */}
                    <div className="flex-1 min-w-0 flex flex-col min-h-0 overflow-hidden hud-corner">

                        {/* Thin sub-header: AI state + status pill */}
                        <div className="h-9 shrink-0 border-b border-[rgba(255,255,255,0.05)] bg-[rgba(5,9,17,0.45)] flex items-center justify-between px-4 gap-4">
                            {/* Left: AI waveform (idle when no audio) */}
                            <div className={`transition-opacity duration-300 ${isMuted ? 'opacity-25' : 'opacity-70'}`}>
                                <TopAudioBar audioData={aiAudioData} />
                            </div>

                            {/* Right: status chip */}
                            <div className={`shrink-0 flex items-center gap-1.5 text-[9px] font-mono px-2 py-0.5 rounded border tracking-widest ${
                                isConnected
                                    ? 'text-cyan-400/60 border-cyan-400/12 bg-[rgba(92,245,255,0.04)]'
                                    : 'text-white/25 border-white/8'
                            }`}>
                                {isTyping && <div className="w-1 h-1 rounded-full bg-cyan-400 animate-pulse" />}
                                {status.toUpperCase()}
                            </div>
                        </div>

                        {/* Chat — fills all remaining vertical space */}
                        <div className="flex-1 min-h-0 overflow-hidden">
                            <ConsoleChat
                                messages={messages}
                                inputValue={inputValue}
                                setInputValue={setInputValue}
                                onKeyDown={stableHandleSend}
                                onSend={stableHandleSendButton}
                                onClearHistory={clearHistory}
                                isTyping={isTyping}
                            />
                        </div>
                    </div>

                    {/* ── RIGHT PANEL — Alert / Telemetry / Controls ─────── */}
                    <AlertPanel
                        status={status}
                        socketConnected={socketConnected}
                        isConnected={isConnected}
                        isMuted={isMuted}
                        isVideoOn={isVideoOn}
                        isHandTrackingEnabled={isHandTrackingEnabled}
                        recentLogs={activityLog}
                        fps={fps}
                        onTogglePower={togglePower}
                        onToggleMute={toggleMute}
                        onToggleVideo={toggleVideo}
                    />
                </div>

                {/* ── FOOTER — Event Timeline ───────────────────────────── */}
                <EventTimeline events={activityLog} />
            </div>

            {/* Overlays / Modals */}
            <AnimatePresence>
                {showSettings && (
                    <motion.div
                        key="settings"
                        initial={{ opacity: 0, scale: 0.97 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.97 }}
                        transition={{ duration: 0.15 }}
                        className="absolute inset-0 z-[60] flex items-center justify-center"
                        style={{ pointerEvents: 'auto' }}
                    >
                        <SettingsWindow
                            socket={socket}
                            micDevices={micDevices}
                            speakerDevices={speakerDevices}
                            webcamDevices={webcamDevices}
                            selectedMicId={selectedMicId}
                            setSelectedMicId={setSelectedMicId}
                            selectedSpeakerId={selectedSpeakerId}
                            setSelectedSpeakerId={setSelectedSpeakerId}
                            selectedWebcamId={selectedWebcamId}
                            setSelectedWebcamId={setSelectedWebcamId}
                            cursorSensitivity={cursorSensitivity}
                            setCursorSensitivity={setCursorSensitivity}
                            isCameraFlipped={isCameraFlipped}
                            setIsCameraFlipped={setIsCameraFlipped}
                            handleFileUpload={handleFileUpload}
                            onClose={() => setShowSettings(false)}
                        />
                    </motion.div>
                )}
            </AnimatePresence>

            <AnimatePresence>
                {showCadWindow && (
                    <motion.div
                        key="cad"
                        id="cad"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 10 }}
                        transition={{ duration: 0.18 }}
                        className={`absolute flex flex-col backdrop-blur-xl bg-black/40 border border-white/10 shadow-2xl overflow-hidden rounded-2xl
                            ${activeDragElement === 'cad' ? 'ring-2 ring-[rgba(92,245,255,0.35)]' : ''}
                        `}
                        style={{
                            left: elementPositions.cad?.x || window.innerWidth / 2,
                            top: elementPositions.cad?.y || window.innerHeight / 2,
                            transform: 'translate(-50%, -50%)',
                            width: `${elementSizes.cad.w}px`,
                            height: `${elementSizes.cad.h}px`,
                            pointerEvents: 'auto',
                            zIndex: getZIndex('cad')
                        }}
                        onMouseDown={(e) => handleMouseDown(e, 'cad')}
                    >
                        <OverlayHeader
                            icon={<Box size={13} />}
                            title="CAD PROTOTYPE"
                            onClose={() => setShowCadWindow(false)}
                            isDragging={activeDragElement === 'cad'}
                        />
                        <div className="noise-overlay absolute inset-0 opacity-10 pointer-events-none mix-blend-overlay z-10"></div>
                        <div className="relative z-20 flex-1 min-h-0">
                            <CadWindow
                                data={cadData}
                                thoughts={cadThoughts}
                                retryInfo={cadRetryInfo}
                                onClose={() => setShowCadWindow(false)}
                                socket={socket}
                            />
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            <AnimatePresence>
                {showImageWindow && (
                    <motion.div
                        key="image"
                        id="image"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 10 }}
                        transition={{ duration: 0.18 }}
                        className={`absolute flex flex-col backdrop-blur-xl bg-black/40 border border-white/10 shadow-2xl overflow-hidden rounded-2xl
                            ${activeDragElement === 'image' ? 'ring-2 ring-[rgba(139,92,246,0.35)]' : ''}
                        `}
                        style={{
                            left: elementPositions.image?.x || window.innerWidth / 2,
                            top: elementPositions.image?.y || window.innerHeight / 2,
                            transform: 'translate(-50%, -50%)',
                            width: `${elementSizes.image.w}px`,
                            height: `${elementSizes.image.h}px`,
                            pointerEvents: 'auto',
                            zIndex: getZIndex('image')
                        }}
                        onMouseDown={(e) => handleMouseDown(e, 'image')}
                    >
                        <OverlayHeader
                            icon={<ImageIcon size={13} />}
                            title="IMAGE CREATOR"
                            onClose={() => setShowImageWindow(false)}
                            isDragging={activeDragElement === 'image'}
                        />
                        <div className="noise-overlay absolute inset-0 opacity-10 pointer-events-none mix-blend-overlay z-10"></div>
                        <div className="relative z-20 flex-1 min-h-0">
                            <ImageWindow
                                data={imageData}
                                onClose={() => setShowImageWindow(false)}
                                socket={socket}
                            />
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            <AnimatePresence>
                {showBrowserWindow && (
                    <motion.div
                        key="browser"
                        id="browser"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 10 }}
                        transition={{ duration: 0.18 }}
                        className={`absolute flex flex-col backdrop-blur-xl bg-black/40 border border-white/10 shadow-2xl overflow-hidden rounded-lg
                            ${activeDragElement === 'browser' ? 'ring-2 ring-[rgba(92,245,255,0.35)]' : ''}
                        `}
                        style={{
                            left: elementPositions.browser?.x || window.innerWidth / 2 - 200,
                            top: elementPositions.browser?.y || window.innerHeight / 2,
                            transform: 'translate(-50%, -50%)',
                            width: `${elementSizes.browser.w}px`,
                            height: `${elementSizes.browser.h}px`,
                            pointerEvents: 'auto',
                            zIndex: getZIndex('browser')
                        }}
                        onMouseDown={(e) => handleMouseDown(e, 'browser')}
                    >
                        <OverlayHeader
                            icon={<Globe size={13} />}
                            title="WEB AGENT"
                            onClose={() => setShowBrowserWindow(false)}
                            isDragging={activeDragElement === 'browser'}
                        />
                        <div className="noise-overlay absolute inset-0 opacity-10 pointer-events-none mix-blend-overlay z-10"></div>
                        <div className="relative z-20 flex-1 min-h-0">
                            <BrowserWindow
                                imageSrc={browserData.image}
                                logs={browserData.logs}
                                onClose={() => setShowBrowserWindow(false)}
                                socket={socket}
                            />
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {isModularMode && (
                <>
                    <ChatModule
                        messages={messages}
                        inputValue={inputValue}
                        setInputValue={setInputValue}
                        handleSend={handleSend}
                        isModularMode={isModularMode}
                        activeDragElement={activeDragElement}
                        position={elementPositions.chat}
                        width={elementSizes.chat.w}
                        height={elementSizes.chat.h}
                        onMouseDown={(e) => handleMouseDown(e, 'chat')}
                    />

                    <div className="z-20 flex justify-center pb-10 pointer-events-none">
                        <ToolsModule
                            isConnected={isConnected}
                            isMuted={isMuted}
                            isVideoOn={isVideoOn}
                            isHandTrackingEnabled={isHandTrackingEnabled}
                            showSettings={showSettings}
                            onTogglePower={togglePower}
                            onToggleMute={toggleMute}
                            onToggleVideo={toggleVideo}
                            onToggleSettings={() => setShowSettings(!showSettings)}
                            onToggleHand={() => setIsHandTrackingEnabled(!isHandTrackingEnabled)}
                            onToggleCad={() => setShowCadWindow(!showCadWindow)}
                            showCadWindow={showCadWindow}
                            onToggleBrowser={() => setShowBrowserWindow(!showBrowserWindow)}
                            showBrowserWindow={showBrowserWindow}
                            activeDragElement={activeDragElement}
                            position={elementPositions.tools}
                            onMouseDown={(e) => handleMouseDown(e, 'tools')}
                        />
                    </div>
                </>
            )}

            <ConfirmationPopup
                request={confirmationRequest}
                onConfirm={handleConfirmTool}
                onDeny={handleDenyTool}
            />

            <CommandPalette
                open={commandPaletteOpen}
                onClose={() => setCommandPaletteOpen(false)}
                actions={commandActions}
            />
        </div>
    );
}

export default App;
