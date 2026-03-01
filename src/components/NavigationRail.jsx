import React from 'react';
import { Box, Globe, Hand, ImageIcon, Mic, MicOff, Power, Settings, Video, VideoOff, Command, MessageSquare } from 'lucide-react';

/**
 * 3-segment arc ring — mirrors the Iron Man HUD's concentric status rings.
 * Each segment is 100° with 20° gaps. Pure SVG, no animation loops.
 */
function ArcRing({ active }) {
    const r = 12;
    const cx = 16;
    const cy = 16;
    const circ = 2 * Math.PI * r;
    // 100° of the circle per segment
    const segLen = (100 / 360) * circ;
    const gapLen = circ - segLen;
    const color = active ? 'rgba(92,245,255,0.75)' : 'rgba(255,90,122,0.3)';

    return (
        <svg width="32" height="32" viewBox="0 0 32 32" aria-hidden="true">
            {/* Track ring */}
            <circle cx={cx} cy={cy} r={r} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="1.5" />
            {/* 3 segments, each rotated 120° */}
            {[0, 120, 240].map((deg) => (
                <circle
                    key={deg}
                    cx={cx}
                    cy={cy}
                    r={r}
                    fill="none"
                    stroke={color}
                    strokeWidth="1.5"
                    strokeDasharray={`${segLen} ${gapLen}`}
                    strokeLinecap="round"
                    transform={`rotate(${-90 + deg} ${cx} ${cy})`}
                />
            ))}
            {/* Center dot */}
            <circle cx={cx} cy={cy} r="2.5" fill={active ? 'rgba(92,245,255,0.6)' : 'rgba(255,90,122,0.25)'} />
        </svg>
    );
}

/** Calibration-ruler separator — tick marks instead of a plain line */
function TickSeparator() {
    return (
        <div className="flex items-center gap-[2px] w-10 my-1">
            <div className="h-px flex-1 bg-[rgba(255,255,255,0.05)]" />
            <div className="h-[5px] w-px bg-[rgba(92,245,255,0.2)]" />
            <div className="h-px flex-1 bg-[rgba(255,255,255,0.05)]" />
            <div className="h-[5px] w-px bg-[rgba(92,245,255,0.2)]" />
            <div className="h-px flex-1 bg-[rgba(255,255,255,0.05)]" />
        </div>
    );
}

function RailButton({ active, disabled, label, shortLabel, onClick, children }) {
    return (
        <div className="relative group active-press flex flex-col items-center gap-0.5">
            {/* Orbit ring */}
            {active && (
                <div className="absolute top-0 inset-x-0 h-11 rounded-xl border border-cyan-400/20 animate-orbit pointer-events-none" />
            )}
            {/* Outer arc-pulse glow ring */}
            {active && (
                <div className="absolute -inset-x-[3px] top-[-3px] h-[50px] rounded-[14px] pointer-events-none arc-pulse" />
            )}
            <button
                type="button"
                disabled={disabled}
                onClick={onClick}
                title={label}
                aria-label={label}
                className={`w-11 h-11 rounded-xl border transition-all flex items-center justify-center relative z-10 ${disabled
                    ? 'border-white/5 text-white/20 cursor-not-allowed'
                    : active
                        ? 'border-cyan-400/50 bg-cyan-400/10 text-cyan-400 shadow-[0_0_15px_rgba(34,211,238,0.2)]'
                        : 'border-white/10 text-white/40 hover:border-white/20 hover:text-white hover:bg-white/5'
                }`}
            >
                {children}
            </button>
            {/* Micro-label — always visible for discoverability */}
            {shortLabel && (
                <span className={`text-[8px] tracking-wider font-mono leading-none select-none ${
                    disabled ? 'text-white/15' : active ? 'text-cyan-400/70' : 'text-white/25 group-hover:text-white/50'
                } transition-colors`}>
                    {shortLabel}
                </span>
            )}
        </div>
    );
}

function NavigationRail({
    isConnected,
    isMuted,
    isVideoOn,
    isHandTrackingEnabled,
    showCadWindow,
    showBrowserWindow,
    showImageWindow,
    showSettings,
    showConversations,
    onTogglePower,
    onToggleMute,
    onToggleVideo,
    onToggleHand,
    onToggleCad,
    onToggleBrowser,
    onToggleImage,
    onToggleSettings,
    onOpenCommandPalette,
    onToggleConversations,
}) {
    return (
        <div className="w-[72px] shrink-0 h-full border-r border-[var(--border-1)] bg-[rgba(15,22,32,0.55)] backdrop-blur-md flex flex-col items-center py-4 gap-2 overflow-y-auto">
            {/* Power / connection ring */}
            <div className="flex flex-col items-center gap-0.5">
                <button
                    type="button"
                    onClick={onTogglePower}
                    title={isConnected ? 'Disconnect' : 'Connect'}
                    aria-label={isConnected ? 'Disconnect' : 'Connect'}
                    className="w-11 h-11 flex items-center justify-center rounded-xl border border-[var(--border-1)] bg-[rgba(0,0,0,0.20)] active-press hover:border-cyan-400/20 transition-colors"
                >
                    <ArcRing active={isConnected} />
                </button>
                <span className={`text-[8px] tracking-wider font-mono leading-none select-none ${isConnected ? 'text-cyan-400/70' : 'text-white/25'}`}>
                    {isConnected ? 'ON' : 'OFF'}
                </span>
            </div>

            <TickSeparator />

            <RailButton disabled={!isConnected} active={isConnected && !isMuted}
                label={isMuted ? 'Unmute Mic' : 'Mute Mic'} shortLabel="MIC" onClick={onToggleMute}>
                {isMuted ? <MicOff size={18} /> : <Mic size={18} />}
            </RailButton>

            <RailButton active={isVideoOn} label={isVideoOn ? 'Disable Camera' : 'Enable Camera'}
                shortLabel="CAM" onClick={onToggleVideo}>
                {isVideoOn ? <Video size={18} /> : <VideoOff size={18} />}
            </RailButton>

            <RailButton active={isHandTrackingEnabled}
                label={isHandTrackingEnabled ? 'Disable Hand Tracking' : 'Enable Hand Tracking'}
                shortLabel="HAND" onClick={onToggleHand}>
                <Hand size={18} />
            </RailButton>

            <TickSeparator />

            <RailButton active={showCadWindow} label="CAD Generator" shortLabel="CAD" onClick={onToggleCad}>
                <Box size={18} />
            </RailButton>

            <RailButton active={showBrowserWindow} label="Web Agent" shortLabel="WEB" onClick={onToggleBrowser}>
                <Globe size={18} />
            </RailButton>

            <RailButton active={showImageWindow} label="Image Creator" shortLabel="IMG" onClick={onToggleImage}>
                <ImageIcon size={18} />
            </RailButton>

            <div className="flex-1" />

            <RailButton active={showConversations} label="Conversations" shortLabel="HIST" onClick={onToggleConversations}>
                <MessageSquare size={18} />
            </RailButton>

            <RailButton label="Command Palette (Ctrl+K)" shortLabel="⌃K" onClick={onOpenCommandPalette}>
                <Command size={18} />
            </RailButton>

            <RailButton active={showSettings} label="Settings" shortLabel="SET" onClick={onToggleSettings}>
                <Settings size={18} />
            </RailButton>
        </div>
    );
}

export default React.memo(NavigationRail);
