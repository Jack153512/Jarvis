import React, { useMemo, useRef, useState } from 'react';
import { AlertTriangle, Bell, Activity, Sliders, Wifi, WifiOff, Mic, MicOff, Video, VideoOff, X, CheckCheck } from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';

// ── Severity helpers ──────────────────────────────────────────────────────────
function deriveSeverity(msg) {
    const m = String(msg || '').toLowerCase();
    if (m.includes('error') || m.includes('fail') || m.includes('crash') || m.includes('offline') || m.includes('exit'))
        return 'high';
    if (m.includes('warn') || m.includes('retry') || m.includes('timeout') || m.includes('slow'))
        return 'medium';
    if (m.includes('ready') || m.includes('connected') || m.includes('success') || m.includes('online'))
        return 'low';
    return 'info';
}

const SEV_STYLES = {
    high:   { badge: 'bg-[rgba(248,81,73,0.15)] border-[rgba(248,81,73,0.4)] text-[#f85149]',   bar: 'border-l-[#f85149]',  dot: '#f85149'  },
    medium: { badge: 'bg-[rgba(210,153,34,0.15)] border-[rgba(210,153,34,0.4)] text-[#d29922]', bar: 'border-l-[#d29922]',  dot: '#d29922'  },
    low:    { badge: 'bg-[rgba(46,229,157,0.10)] border-[rgba(46,229,157,0.3)] text-[#2ee59d]', bar: 'border-l-[#2ee59d]',  dot: '#2ee59d'  },
    info:   { badge: 'bg-[rgba(92,245,255,0.08)] border-[rgba(92,245,255,0.2)] text-[#5cf5ff]', bar: 'border-l-[rgba(92,245,255,0.5)]', dot: 'rgba(92,245,255,0.6)' },
};

// ── AlertCard ─────────────────────────────────────────────────────────────────
function AlertCard({ entry, onDismiss }) {
    const msg = typeof entry === 'object' ? entry.msg : String(entry);
    const ts  = typeof entry === 'object' ? entry.timestamp : null;
    const sev = deriveSeverity(msg);
    const { badge, bar } = SEV_STYLES[sev];

    return (
        <motion.div
            layout
            initial={{ opacity: 0, x: 12, height: 0 }}
            animate={{ opacity: 1, x: 0, height: 'auto' }}
            exit={{ opacity: 0, x: 12, height: 0 }}
            transition={{ duration: 0.16 }}
            className={`rounded-lg bg-[rgba(255,255,255,0.025)] border border-[rgba(255,255,255,0.07)] border-l-2 ${bar} overflow-hidden`}
        >
            <div className="px-2.5 py-2 flex items-start justify-between gap-2">
                <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5 mb-1 flex-wrap">
                        <span className={`text-[8px] font-mono tracking-widest px-1.5 py-0.5 rounded border ${badge}`}>
                            {sev.toUpperCase()}
                        </span>
                        {ts && (
                            <span className="text-[9px] font-mono text-[var(--text-3)] tabular-nums">{ts}</span>
                        )}
                    </div>
                    <p className="text-[11px] text-[var(--text-2)] leading-snug">{msg}</p>
                </div>
                <button
                    onClick={onDismiss}
                    title="Dismiss"
                    className="shrink-0 w-5 h-5 flex items-center justify-center rounded text-white/20 hover:text-white/60 transition-colors mt-0.5"
                >
                    <X size={10} />
                </button>
            </div>
        </motion.div>
    );
}

// ── SparkLine ─────────────────────────────────────────────────────────────────
function SparkLine({ values, color = '#5cf5ff', max = 60 }) {
    const W = 64, H = 20;
    if (!values || values.length < 2) return <div style={{ width: W, height: H }} />;
    const pts = values
        .map((v, i) => {
            const x = (i / (values.length - 1)) * W;
            const y = H - (Math.min(Math.max(v, 0), max) / max) * (H - 2) - 1;
            return `${x.toFixed(1)},${y.toFixed(1)}`;
        })
        .join(' ');
    return (
        <svg width={W} height={H} className="shrink-0" aria-hidden="true">
            <polyline
                points={pts}
                fill="none"
                stroke={color}
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                opacity="0.75"
            />
        </svg>
    );
}

// ── TelemetryRow ──────────────────────────────────────────────────────────────
function TelemetryRow({ label, value, sparkValues, color = 'var(--text-1)', max }) {
    return (
        <div className="flex items-center justify-between gap-3">
            <div className="min-w-0 flex-1">
                <div className="text-[9px] text-hud text-[var(--text-3)] mb-0.5">{label}</div>
                <span
                    className="text-[13px] font-mono font-semibold tabular-nums"
                    style={{ color }}
                >
                    {value}
                </span>
            </div>
            {sparkValues && <SparkLine values={sparkValues} color={color} max={max} />}
        </div>
    );
}

// ── SectionHeader ─────────────────────────────────────────────────────────────
function SectionHeader({ icon, title, count, onClearAll }) {
    return (
        <div className="flex items-center justify-between pb-1.5 border-b border-[rgba(255,255,255,0.06)]">
            <div className="flex items-center gap-1.5">
                <span className="text-[var(--accent)] opacity-60">{icon}</span>
                <span className="text-[10px] text-hud text-[var(--text-3)]">{title}</span>
            </div>
            <div className="flex items-center gap-2">
                {count != null && count > 0 && (
                    <span className="text-[9px] font-mono bg-[rgba(248,81,73,0.15)] text-[#f85149] border border-[rgba(248,81,73,0.3)] rounded-full px-1.5 py-0.5 tabular-nums">
                        {count}
                    </span>
                )}
                {onClearAll && count > 0 && (
                    <button
                        onClick={onClearAll}
                        title="Dismiss all"
                        className="text-[9px] font-mono text-[var(--text-3)] hover:text-[var(--text-2)] transition-colors flex items-center gap-0.5"
                    >
                        <CheckCheck size={9} /> ALL
                    </button>
                )}
            </div>
        </div>
    );
}

// ── QuickControl ──────────────────────────────────────────────────────────────
function QuickControl({ icon, label, active, onClick, activeColor = '#5cf5ff' }) {
    return (
        <button
            type="button"
            onClick={onClick}
            title={label}
            className={`flex flex-col items-center gap-1 py-2 rounded-lg border transition-all ${
                active
                    ? 'border-[rgba(92,245,255,0.25)] bg-[rgba(92,245,255,0.07)]'
                    : 'border-[rgba(255,255,255,0.07)] bg-transparent hover:border-[rgba(255,255,255,0.14)] hover:bg-[rgba(255,255,255,0.03)]'
            }`}
            style={{ color: active ? activeColor : 'rgba(255,255,255,0.28)' }}
        >
            {icon}
            <span className="text-[8px] font-mono tracking-wider leading-none">{label}</span>
        </button>
    );
}

// ── AlertPanel (main export) ──────────────────────────────────────────────────
function AlertPanel({
    status,
    socketConnected,
    isConnected,
    isMuted,
    isVideoOn,
    isHandTrackingEnabled,
    recentLogs,
    fps,
    onTogglePower,
    onToggleMute,
    onToggleVideo,
}) {
    const [dismissed, setDismissed] = useState(new Set());

    // FPS history ring-buffer
    const fpsHistRef    = useRef(new Array(20).fill(0));
    const evpmHistRef   = useRef(new Array(20).fill(0));
    const lastCountRef  = useRef(0);
    const lastTickRef   = useRef(Date.now());

    const fpsNum = typeof fps === 'number' ? fps : 0;

    // Update FPS history without triggering re-renders
    const lastFps = fpsHistRef.current[fpsHistRef.current.length - 1];
    if (lastFps !== fpsNum) {
        fpsHistRef.current = [...fpsHistRef.current.slice(-19), fpsNum];
    }

    // Approx events-per-minute — sampled every 5 s
    const logCount = (recentLogs || []).length;
    const nowMs = Date.now();
    if (nowMs - lastTickRef.current > 5000) {
        const diff    = Math.max(0, logCount - lastCountRef.current);
        const evPerMin = Math.round((diff / ((nowMs - lastTickRef.current) / 1000)) * 60);
        evpmHistRef.current = [...evpmHistRef.current.slice(-19), evPerMin];
        lastCountRef.current = logCount;
        lastTickRef.current  = nowMs;
    }

    const fpsColor    = fpsNum >= 55 ? '#2ee59d' : fpsNum >= 30 ? '#ffcc66' : '#ff5a7a';
    const latestEvpm  = evpmHistRef.current[evpmHistRef.current.length - 1] ?? 0;

    const modelState = useMemo(() => {
        const s = String(status || '');
        if (s === 'Model Connected') return { label: 'READY',      color: '#2ee59d' };
        if (s === 'Connected')       return { label: 'ONLINE',     color: '#2ee59d' };
        if (s.includes('Connecting')) return { label: 'CONNECTING', color: '#ffcc66' };
        return                              { label: 'OFFLINE',    color: '#ff5a7a' };
    }, [status]);

    const alerts = useMemo(
        () =>
            (recentLogs || [])
                .slice(-30)
                .reverse()
                .filter(e => !dismissed.has(typeof e === 'object' ? (e.id ?? e.msg) : String(e))),
        [recentLogs, dismissed]
    );

    const dismissEntry = (entry) => {
        const key = typeof entry === 'object' ? (entry.id ?? entry.msg) : String(entry);
        setDismissed(prev => new Set([...prev, key]));
    };

    const dismissAll = () =>
        setDismissed(
            new Set((recentLogs || []).map(e => (typeof e === 'object' ? (e.id ?? e.msg) : String(e))))
        );

    return (
        <div className="w-[260px] shrink-0 h-full border-l border-[var(--border-1)] bg-[rgba(7,12,21,0.7)] backdrop-blur-md flex flex-col overflow-hidden">

            {/* ── ALERTS ── */}
            <div className="flex-1 min-h-0 overflow-y-auto p-3 flex flex-col gap-2">
                <SectionHeader
                    icon={<Bell size={11} />}
                    title="ALERTS"
                    count={alerts.length}
                    onClearAll={dismissAll}
                />
                <AnimatePresence>
                    {alerts.length === 0 ? (
                        <motion.span
                            key="empty"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="text-[11px] font-mono text-[var(--text-3)] italic py-1"
                        >
                            No active alerts
                        </motion.span>
                    ) : (
                        alerts.map((entry, i) => (
                            <AlertCard
                                key={typeof entry === 'object' ? (entry.id ?? i) : i}
                                entry={entry}
                                onDismiss={() => dismissEntry(entry)}
                            />
                        ))
                    )}
                </AnimatePresence>
            </div>

            <div className="border-t border-[rgba(255,255,255,0.06)]" />

            {/* ── TELEMETRY ── */}
            <div className="p-3 flex flex-col gap-3 shrink-0">
                <SectionHeader icon={<Activity size={11} />} title="TELEMETRY" />

                <TelemetryRow
                    label="BACKEND"
                    value={socketConnected ? 'ONLINE' : 'OFFLINE'}
                    color={socketConnected ? '#2ee59d' : '#ff5a7a'}
                />
                <TelemetryRow
                    label="MODEL"
                    value={modelState.label}
                    color={modelState.color}
                />
                {isVideoOn && (
                    <TelemetryRow
                        label="CAMERA FPS"
                        value={fpsNum > 0 ? String(Math.round(fpsNum)) : '—'}
                        sparkValues={fpsHistRef.current}
                        color={fpsColor}
                        max={60}
                    />
                )}
                <TelemetryRow
                    label="EVENTS / MIN"
                    value={String(latestEvpm)}
                    sparkValues={evpmHistRef.current}
                    color="rgba(92,245,255,0.7)"
                    max={20}
                />
            </div>

            <div className="border-t border-[rgba(255,255,255,0.06)]" />

            {/* ── CONTROLS ── */}
            <div className="p-3 flex flex-col gap-2 shrink-0">
                <SectionHeader icon={<Sliders size={11} />} title="QUICK CONTROLS" />

                <div className="grid grid-cols-3 gap-1.5">
                    <QuickControl
                        icon={isConnected ? <Wifi size={13} /> : <WifiOff size={13} />}
                        label="SESSION"
                        active={isConnected}
                        onClick={onTogglePower}
                        activeColor="#2ee59d"
                    />
                    <QuickControl
                        icon={isMuted ? <MicOff size={13} /> : <Mic size={13} />}
                        label="MIC"
                        active={!isMuted}
                        onClick={onToggleMute}
                    />
                    <QuickControl
                        icon={isVideoOn ? <Video size={13} /> : <VideoOff size={13} />}
                        label="CAM"
                        active={isVideoOn}
                        onClick={onToggleVideo}
                    />
                </div>
            </div>
        </div>
    );
}

export default React.memo(AlertPanel);
