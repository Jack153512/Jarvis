import React, { useMemo, useRef } from 'react';

/**
 * Arc-ring chip — replaces the plain pill chip with an SVG progress arc.
 * Full arc = ok, ¾ = warn, ¼ = err, empty track = neutral.
 * No animation loops; only the LiveDot pulses.
 */
function RingChip({ tone = 'neutral', live = false, children }) {
    const r = 7;
    const circ = 2 * Math.PI * r;

    const { color, fill } = useMemo(() => {
        if (tone === 'ok')   return { color: '#2ee59d', fill: circ };
        if (tone === 'warn') return { color: '#ffcc66', fill: circ * 0.72 };
        if (tone === 'err')  return { color: '#ff5a7a', fill: circ * 0.28 };
        return { color: '#484f58', fill: circ * 0.15 };
    }, [tone, circ]);

    const borderClass = useMemo(() => {
        if (tone === 'ok')   return 'border-[rgba(46,229,157,0.22)]';
        if (tone === 'warn') return 'border-[rgba(255,204,102,0.22)]';
        if (tone === 'err')  return 'border-[rgba(255,90,122,0.22)]';
        return 'border-[var(--border-1)]';
    }, [tone]);

    return (
        <div className={`inline-flex items-center gap-1.5 text-[10px] tracking-widest px-2 py-0.5 rounded-full border bg-[rgba(0,0,0,0.12)] ${borderClass}`}
             style={{ color }}>
            <svg width="16" height="16" viewBox="0 0 16 16" className="shrink-0" aria-hidden="true">
                {/* Track */}
                <circle cx="8" cy="8" r={r} fill="none" stroke={`${color}22`} strokeWidth="1.5" />
                {/* Progress arc */}
                <circle
                    cx="8" cy="8" r={r}
                    fill="none"
                    stroke={`${color}bb`}
                    strokeWidth="1.5"
                    strokeDasharray={`${fill} ${circ}`}
                    strokeLinecap="round"
                    transform="rotate(-90 8 8)"
                />
                {/* Live dot in center */}
                {live && (
                    <circle cx="8" cy="8" r="1.8" fill={color} opacity="0.9" />
                )}
            </svg>
            {children}
        </div>
    );
}

function PanelCard({ title, children }) {
    return (
        <div className="rounded-[var(--radius-panel)] border border-[var(--border-1)] bg-[rgba(15,22,32,0.55)] backdrop-blur-md overflow-hidden">
            <div className="px-3 py-2 border-b border-[rgba(255,255,255,0.06)] flex items-center justify-between">
                <div className="text-[10px] text-hud text-[var(--text-3)]">{title}</div>
            </div>
            <div className="p-3">{children}</div>
        </div>
    );
}

function StatusPanel({
    status,
    socketConnected,
    isConnected,
    isMuted,
    isVideoOn,
    isHandTrackingEnabled,
    currentProject,
    recentLogs,
    fps,
}) {
    const backendTone = socketConnected ? 'ok' : 'err';
    const sessionTone = isConnected ? 'ok' : 'warn';
    const micTone = isMuted ? 'neutral' : 'ok';
    const camTone = isVideoOn ? 'ok' : 'neutral';
    const handTone = isHandTrackingEnabled ? 'ok' : 'neutral';

    const fpsNum = typeof fps === 'number' ? fps : 0;
    const fpsColor =
        fpsNum >= 55 ? 'text-[#2ee59d]' :
        fpsNum >= 30 ? 'text-[#ffcc66]' :
        'text-[#ff5a7a]';

    // FPS history ring-buffer — persists across renders without triggering re-renders.
    const fpsHistoryRef = useRef(new Array(20).fill(0));
    const last = fpsHistoryRef.current[fpsHistoryRef.current.length - 1];
    if (last !== fpsNum) {
        fpsHistoryRef.current = [...fpsHistoryRef.current.slice(-19), fpsNum];
    }

    // Derive a human-readable model state from `status`
    const modelState = useMemo(() => {
        const s = String(status || '');
        if (s === 'Model Connected') return { label: 'MODEL READY', tone: 'ok' };
        if (s === 'Connected')       return { label: 'CONNECTED', tone: 'ok' };
        if (s === 'Connecting...')   return { label: 'CONNECTING', tone: 'warn' };
        if (s.includes('Offline') || s === 'Disconnected') return { label: 'OFFLINE', tone: 'err' };
        return { label: s.toUpperCase().slice(0, 20), tone: 'neutral' };
    }, [status]);

    return (
        <div className="w-[280px] shrink-0 h-full border-l border-[var(--border-1)] bg-[rgba(10,15,24,0.55)] backdrop-blur-md p-3 flex flex-col gap-3 overflow-y-auto">

            {/* Section header + backend pill */}
            <div className="flex items-center justify-between">
                <span className="text-hud text-[11px] text-[var(--accent)]">SYSTEM</span>
                <RingChip tone={backendTone} live={socketConnected}>
                    {socketConnected ? 'ONLINE' : 'OFFLINE'}
                </RingChip>
            </div>

            {/* Status chips — compact grid */}
            <div className="grid grid-cols-2 gap-1.5">
                <RingChip tone={modelState.tone}>{modelState.label}</RingChip>
                <RingChip tone={sessionTone} live={isConnected}>
                    {isConnected ? 'SESSION ON' : 'SESSION OFF'}
                </RingChip>
                <RingChip tone={micTone} live={!isMuted}>
                    {isMuted ? 'MIC MUTED' : 'MIC LIVE'}
                </RingChip>
                <RingChip tone={camTone} live={isVideoOn}>
                    {isVideoOn ? 'CAM ON' : 'CAM OFF'}
                </RingChip>
                {isHandTrackingEnabled && (
                    <RingChip tone={handTone} live>HAND ON</RingChip>
                )}
            </div>

            {/* Project */}
            <PanelCard title="PROJECT">
                <div className="text-[13px] text-hud text-[var(--text-1)] truncate">
                    {String(currentProject || 'default').toUpperCase()}
                </div>
            </PanelCard>

            {/* Camera FPS — only show when camera is active */}
            {isVideoOn && (
                <PanelCard title="CAMERA">
                    <div className="flex items-center justify-between mb-1.5">
                        <span className="text-[10px] text-hud text-[var(--text-3)]">RENDER FPS</span>
                        <span className={`text-[13px] font-mono font-bold tabular-nums ${fpsColor}`}>
                            {fpsNum > 0 ? fpsNum.toFixed(0) : '—'}
                        </span>
                    </div>
                    <div className="flex items-end gap-[2px] h-6">
                        {fpsHistoryRef.current.map((v, i) => {
                            const h = Math.max(2, Math.round((Math.min(v, 60) / 60) * 22));
                            const col = v >= 55 ? '#2ee59d' : v >= 30 ? '#ffcc66' : '#ff5a7a';
                            return (
                                <div key={i} style={{ height: h, width: 3, background: col, opacity: 0.65, borderRadius: 1, flexShrink: 0 }} />
                            );
                        })}
                    </div>
                </PanelCard>
            )}

            {/* Activity feed */}
            <PanelCard title="ACTIVITY">
                <div className="space-y-1.5 max-h-[220px] overflow-y-auto">
                    {(recentLogs || []).length > 0 ? (
                        (recentLogs || []).slice(-8).reverse().map((entry, idx) => {
                            const msg = entry && typeof entry === 'object' ? entry.msg : entry;
                            const ts  = entry && typeof entry === 'object' ? entry.timestamp : null;
                            return (
                                <div key={idx} className="flex gap-2 items-start border-l-2 border-[rgba(92,245,255,0.15)] pl-2">
                                    {ts && (
                                        <span className="text-[9px] font-mono text-[var(--text-3)] shrink-0 mt-0.5 tabular-nums">{ts}</span>
                                    )}
                                    <span className="text-[11px] font-mono text-[var(--text-2)] leading-relaxed">{msg}</span>
                                </div>
                            );
                        })
                    ) : (
                        <span className="text-[11px] font-mono text-[var(--text-3)] italic">No recent activity</span>
                    )}
                </div>
            </PanelCard>

            <div className="flex-1" />

            <div className="text-[10px] text-hud text-[var(--text-3)] tracking-widest opacity-50">
                ⌃K — COMMAND PALETTE
            </div>
        </div>
    );
}

export default React.memo(StatusPanel);
