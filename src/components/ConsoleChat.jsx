import React, { useEffect, useMemo, useRef } from 'react';
import { Send, Trash2 } from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';

function TypingIndicator() {
    return (
        <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 4 }}
            transition={{ duration: 0.15 }}
            className="flex items-center gap-1.5 px-3 py-2"
        >
            <span className="text-[10px] text-hud text-[var(--accent)] tracking-[0.2em]">PROCESSING</span>
            <span className="text-[10px] font-mono text-[var(--accent)] blink-cursor">_</span>
        </motion.div>
    );
}

function MessageBubble({ msg, showHeader }) {
    const sender = String(msg.sender || '').toLowerCase();
    const isUser = sender === 'you';
    const isSystem = sender === 'system';

    if (isSystem) {
        return (
            <div className="flex justify-center my-0.5">
                <div className="text-[10px] tracking-widest text-[var(--text-3)] bg-[rgba(255,255,255,0.02)] border border-[rgba(255,255,255,0.06)] rounded-full px-3 py-0.5">
                    {msg.text}
                </div>
            </div>
        );
    }

    return (
        <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'}`}>
            {/* Show sender label + timestamp only on role-change or explicit flag */}
            {showHeader && (
                <div className={`flex items-center gap-2 mb-1 px-1 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
                    <span className={`text-[10px] text-hud font-semibold ${isUser ? 'text-[rgba(255,255,255,0.5)]' : 'text-[rgba(92,245,255,0.7)]'}`}>
                        {String(msg.sender || '').toUpperCase()}
                    </span>
                    <span className="text-[9px] text-[var(--text-3)] font-mono tabular-nums">{msg.time}</span>
                </div>
            )}
            <div className={`max-w-[78%] rounded-xl px-4 py-2.5 ${
                isUser
                    ? 'bg-[rgba(255,255,255,0.06)] border border-[rgba(255,255,255,0.12)] rounded-br-sm'
                    : 'bg-[rgba(92,245,255,0.05)] border border-[rgba(92,245,255,0.14)] border-l-[3px] border-l-[rgba(92,245,255,0.45)] rounded-bl-sm'
            }`}>
                <div className={`leading-[1.65] whitespace-pre-wrap break-words ${
                    isUser
                        ? 'text-[13.5px] text-[var(--text-1)] text-right'
                        : 'text-[13.5px] text-[var(--text-1)]'
                }`}>
                    {msg.text}
                </div>
            </div>
        </div>
    );
}

function SessionDivider() {
    return (
        <div className="flex items-center gap-2 py-1">
            <div className="flex-1 h-px bg-[rgba(255,255,255,0.06)]" />
            <span className="text-[9px] tracking-[0.25em] text-[var(--text-3)] font-mono shrink-0">
                PREVIOUS SESSION
            </span>
            <div className="flex-1 h-px bg-[rgba(255,255,255,0.06)]" />
        </div>
    );
}

function ConsoleChat({ messages, inputValue, setInputValue, onSend, onKeyDown, isTyping, onClearHistory }) {
    const messagesEndRef = useRef(null);

    // Find where history ends and live messages begin
    const { visible, sessionBreakIdx } = useMemo(() => {
        const list = Array.isArray(messages) ? messages : [];
        const trimmed = list.slice(-150);
        // Find the last index that is still fromHistory
        let lastHistoryIdx = -1;
        for (let i = 0; i < trimmed.length; i++) {
            if (trimmed[i].fromHistory) lastHistoryIdx = i;
        }
        return { visible: trimmed, sessionBreakIdx: lastHistoryIdx };
    }, [messages]);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [visible.length, isTyping]);

    const isEmpty = visible.length === 0;
    const canSend = inputValue.trim().length > 0;

    return (
        <div className="h-full flex flex-col px-4 py-3">
            {/* Header row with clear button */}
            {!isEmpty && (
                <div className="flex items-center justify-end pb-2 shrink-0">
                    <button
                        type="button"
                        onClick={onClearHistory}
                        title="Clear history"
                        className="flex items-center gap-1 text-[9px] tracking-widest text-[var(--text-3)] hover:text-red-400/70 transition-colors font-mono"
                    >
                        <Trash2 size={10} />
                        CLEAR
                    </button>
                </div>
            )}

            {/* Message list — wider, more breathing room */}
            <div className="flex-1 min-h-0 overflow-y-auto scrollbar-hide">
                {isEmpty ? (
                    <div className="h-full flex flex-col items-center justify-center gap-3">
                        <motion.div
                            animate={{ opacity: [0.25, 0.55, 0.25] }}
                            transition={{ duration: 3, repeat: Infinity }}
                            className="text-[12px] tracking-[0.35em] text-[var(--text-3)] font-mono"
                        >
                            AWAITING INPUT
                        </motion.div>
                        <p className="text-[11px] text-[var(--text-3)] opacity-50 max-w-[260px] text-center leading-relaxed">
                            Type a message or press the mic button to speak.
                        </p>
                    </div>
                ) : (
                    <div className="space-y-1.5 py-2 max-w-3xl mx-auto w-full">
                        <AnimatePresence initial={false}>
                            {visible.map((msg, i) => {
                                const prev = i > 0 ? visible[i - 1] : null;
                                const senderChanged = !prev || prev.sender !== msg.sender;
                                const showHeader = senderChanged && msg.sender?.toLowerCase() !== 'system';
                                return (
                                    <React.Fragment key={msg.id ?? i}>
                                        {i === sessionBreakIdx + 1 && sessionBreakIdx >= 0 && (
                                            <SessionDivider />
                                        )}
                                        <motion.div
                                            initial={{ opacity: 0, y: 6 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            transition={{ duration: 0.15 }}
                                            className={senderChanged && i > 0 ? 'mt-4' : ''}
                                        >
                                            <MessageBubble msg={msg} showHeader={showHeader} />
                                        </motion.div>
                                    </React.Fragment>
                                );
                            })}
                        </AnimatePresence>

                        <AnimatePresence>
                            {isTyping && <TypingIndicator />}
                        </AnimatePresence>

                        <div ref={messagesEndRef} />
                    </div>
                )}
            </div>

            {/* Input area — full-width, prominent, centred max-width */}
            <div className="mt-3 shrink-0 max-w-3xl mx-auto w-full">
                <div className="flex items-center gap-2 rounded-xl bg-[rgba(8,14,24,0.80)] border border-[rgba(255,255,255,0.11)] shadow-[0_4px_24px_rgba(0,0,0,0.5)] px-3 py-2 focus-within:border-[rgba(92,245,255,0.32)] transition-colors">
                    <input
                        type="text"
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        onKeyDown={onKeyDown}
                        placeholder="Ask J.A.R.V.I.S anything…"
                        className="flex-1 bg-transparent px-1 py-1.5 text-[14px] text-[var(--text-1)] placeholder:text-[var(--text-3)] outline-none"
                    />
                    <button
                        type="button"
                        onClick={onSend}
                        disabled={!canSend}
                        title="Send (Enter)"
                        className={`h-8 w-8 rounded-lg border flex items-center justify-center transition-all shrink-0 ${
                            canSend
                                ? 'border-[rgba(92,245,255,0.4)] bg-[rgba(92,245,255,0.10)] text-[var(--accent)] hover:bg-[rgba(92,245,255,0.20)] active:scale-95'
                                : 'border-[rgba(255,255,255,0.05)] bg-transparent text-[var(--text-3)] cursor-not-allowed'
                        }`}
                    >
                        <Send size={14} />
                    </button>
                </div>
            </div>
        </div>
    );
}

export default React.memo(ConsoleChat);
