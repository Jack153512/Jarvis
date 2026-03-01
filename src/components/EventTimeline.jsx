import React, { useEffect, useRef } from 'react';

function dotColor(msg) {
    const m = String(msg || '').toLowerCase();
    if (m.includes('error') || m.includes('fail') || m.includes('crash') || m.includes('offline') || m.includes('exit'))
        return '#f85149';
    if (m.includes('warn') || m.includes('retry') || m.includes('timeout'))
        return '#d29922';
    if (m.includes('ready') || m.includes('connected') || m.includes('success') || m.includes('online'))
        return '#2ee59d';
    return 'rgba(92,245,255,0.55)';
}

function EventTimeline({ events = [] }) {
    const trackRef = useRef(null);

    // Keep the timeline scrolled to the latest event
    useEffect(() => {
        if (trackRef.current) {
            trackRef.current.scrollLeft = trackRef.current.scrollWidth;
        }
    }, [events.length]);

    return (
        <div className="h-11 shrink-0 border-t border-[rgba(255,255,255,0.05)] bg-[rgba(5,9,17,0.88)] backdrop-blur-md flex items-stretch select-none">

            {/* Section label */}
            <div className="shrink-0 px-3 border-r border-[rgba(255,255,255,0.06)] flex items-center gap-2">
                <span className="text-[8px] font-mono text-[var(--text-3)] tracking-[0.2em] whitespace-nowrap">
                    EVENT LOG
                </span>
            </div>

            {/* Scrollable track */}
            <div
                ref={trackRef}
                className="flex-1 overflow-x-auto scrollbar-hide relative flex items-center px-4 gap-3"
                style={{ minWidth: 0 }}
            >
                {/* Baseline rule */}
                <div className="absolute left-4 right-4 top-1/2 h-px bg-[rgba(255,255,255,0.05)] pointer-events-none" />

                {events.length === 0 ? (
                    <span className="text-[9px] font-mono text-[var(--text-3)] tracking-widest z-10 relative">
                        NO EVENTS
                    </span>
                ) : (
                    events.slice(-80).map((entry, i) => {
                        const msg   = typeof entry === 'object' ? entry.msg       : String(entry);
                        const ts    = typeof entry === 'object' ? entry.timestamp : null;
                        const color = dotColor(msg);
                        return (
                            <div
                                key={typeof entry === 'object' ? (entry.id ?? i) : i}
                                className="shrink-0 relative group z-10 cursor-default flex flex-col items-center"
                            >
                                {/* Dot */}
                                <div
                                    className="w-2 h-2 rounded-full transition-transform duration-100 group-hover:scale-150"
                                    style={{ background: color, boxShadow: `0 0 5px ${color}` }}
                                />

                                {/* Tooltip — above the dot */}
                                <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-100 z-30 whitespace-nowrap">
                                    <div className="bg-[rgba(5,9,17,0.96)] border border-[rgba(255,255,255,0.10)] rounded-md px-2.5 py-1.5 shadow-2xl">
                                        {ts && (
                                            <div className="text-[8px] font-mono text-[var(--text-3)] tabular-nums mb-0.5">{ts}</div>
                                        )}
                                        <div
                                            className="text-[10px] font-mono max-w-[200px] truncate"
                                            style={{ color }}
                                        >
                                            {msg}
                                        </div>
                                    </div>
                                    {/* Pointer */}
                                    <div
                                        className="absolute top-full left-1/2 -translate-x-1/2 w-0 h-0"
                                        style={{ borderLeft: '4px solid transparent', borderRight: '4px solid transparent', borderTop: '4px solid rgba(255,255,255,0.10)' }}
                                    />
                                </div>
                            </div>
                        );
                    })
                )}
            </div>

            {/* Event count badge */}
            <div className="shrink-0 px-3 border-l border-[rgba(255,255,255,0.06)] flex items-center">
                <span className="text-[9px] font-mono text-[var(--text-3)] tabular-nums">
                    {events.length}
                </span>
            </div>
        </div>
    );
}

export default React.memo(EventTimeline);
