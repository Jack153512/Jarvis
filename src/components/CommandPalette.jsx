import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Command } from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';

function CommandPalette({ open, onClose, actions = [] }) {
    const [query, setQuery] = useState('');
    const [activeIndex, setActiveIndex] = useState(0);
    const inputRef = useRef(null);

    useEffect(() => {
        if (!open) return;
        setQuery('');
        setActiveIndex(0);
        const t = setTimeout(() => inputRef.current?.focus(), 0);
        return () => clearTimeout(t);
    }, [open]);

    const filtered = useMemo(() => {
        const q = query.trim().toLowerCase();
        const list = Array.isArray(actions) ? actions : [];
        if (!q) return list;
        return list.filter(a => {
            const hay = `${a.label || ''} ${a.keywords || ''}`.toLowerCase();
            return hay.includes(q);
        });
    }, [actions, query]);

    useEffect(() => {
        if (!open) return;
        const onKeyDown = (e) => {
            if (e.key === 'Escape') {
                e.preventDefault();
                onClose?.();
                return;
            }
            if (e.key === 'ArrowDown') {
                e.preventDefault();
                setActiveIndex(i => Math.min(i + 1, Math.max(filtered.length - 1, 0)));
                return;
            }
            if (e.key === 'ArrowUp') {
                e.preventDefault();
                setActiveIndex(i => Math.max(i - 1, 0));
                return;
            }
            if (e.key === 'Enter') {
                e.preventDefault();
                const picked = filtered[activeIndex];
                if (picked?.run) {
                    picked.run();
                    onClose?.();
                }
            }
        };
        window.addEventListener('keydown', onKeyDown, true);
        return () => window.removeEventListener('keydown', onKeyDown, true);
    }, [open, filtered, activeIndex, onClose]);

    return (
        <AnimatePresence>
            {open && (
        <motion.div
            key="command-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="fixed inset-0 z-[200] flex items-start justify-center pt-32"
        >
            <div className="absolute inset-0 bg-black/40 backdrop-blur-md" onMouseDown={onClose} />
            <motion.div
                initial={{ opacity: 0, scale: 0.97, y: -8 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.97, y: -8 }}
                transition={{ duration: 0.18 }}
                className="relative w-[min(640px,92vw)] frosted-glass rounded-2xl overflow-hidden"
            >
                <div className="px-6 py-4 border-b border-white/5 flex items-center gap-4">
                    <Command size={18} className="text-cyan-400/60" />
                    <input
                        ref={inputRef}
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Neural command input..."
                        className="flex-1 bg-transparent text-[var(--text-1)] placeholder:text-white/20 outline-none text-sm font-medium tracking-wide"
                    />
                    <div className="flex items-center gap-1.5 opacity-30">
                        <span className="text-[10px] font-mono border border-white/20 px-1.5 py-0.5 rounded">ESC</span>
                    </div>
                </div>
                <div className="max-h-[420px] overflow-y-auto scrollbar-hide p-2">
                    {filtered.length === 0 ? (
                        <div className="px-6 py-12 text-center">
                            <div className="text-xs terminal-text text-white/20">NO_MATCHES_FOUND</div>
                        </div>
                    ) : (
                        <div className="flex flex-col gap-1">
                            {filtered.map((a, idx) => (
                                <button
                                    key={a.id || `${a.label}-${idx}`}
                                    onClick={() => {
                                        a.run?.();
                                        onClose?.();
                                    }}
                                    onMouseEnter={() => setActiveIndex(idx)}
                                    className={`w-full text-left px-4 py-3 rounded-xl flex items-center justify-between gap-4 transition-all duration-200 ${idx === activeIndex ? 'bg-cyan-400/10 border-white/10' : 'bg-transparent hover:bg-white/5 border-transparent'} border`}
                                >
                                    <div className="flex items-center gap-3">
                                        <div className={`w-1.5 h-1.5 rounded-full transition-all duration-300 ${idx === activeIndex ? 'bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.8)]' : 'bg-white/10'}`} />
                                        <div>
                                            <div className={`text-sm transition-colors ${idx === activeIndex ? 'text-white' : 'text-white/60'}`}>{a.label}</div>
                                            {a.hint ? (
                                                <div className="text-[10px] text-white/30 font-mono mt-0.5 uppercase tracking-tighter">{a.hint}</div>
                                            ) : null}
                                        </div>
                                    </div>
                                    {a.shortcut ? (
                                        <div className={`text-[10px] border px-2 py-0.5 rounded-md font-mono transition-colors ${idx === activeIndex ? 'border-cyan-400/30 text-cyan-400/60' : 'border-white/5 text-white/20'}`}>
                                            {a.shortcut}
                                        </div>
                                    ) : null}
                                </button>
                            ))}
                        </div>
                    )}
                </div>
                <div className="px-4 py-2 bg-black/20 border-t border-white/5 flex items-center justify-between opacity-40">
                    <div className="terminal-text text-[9px]">SYSTEM_BRAIN_V2</div>
                    <div className="flex gap-4">
                        <div className="flex items-center gap-1">
                            <span className="text-[9px] font-mono border border-white/20 px-1 rounded">↑↓</span>
                            <span className="terminal-text text-[8px]">NAVIGATE</span>
                        </div>
                        <div className="flex items-center gap-1">
                            <span className="text-[9px] font-mono border border-white/20 px-1 rounded">↵</span>
                            <span className="terminal-text text-[8px]">EXECUTE</span>
                        </div>
                    </div>
                </div>
            </motion.div>
        </motion.div>
            )}
        </AnimatePresence>
    );
}

export default CommandPalette;
