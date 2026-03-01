import React, { useState, useRef, useEffect } from 'react';
import { MessageSquarePlus, Trash2, Check, X, MessageSquare } from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';

function formatRelativeTime(isoString) {
    if (!isoString) return '';
    try {
        const date = new Date(isoString);
        const now = new Date();
        const diffMs = now - date;
        const diffMin = Math.floor(diffMs / 60000);
        const diffHr = Math.floor(diffMin / 60);
        const diffDay = Math.floor(diffHr / 24);
        if (diffMin < 1) return 'just now';
        if (diffMin < 60) return `${diffMin}m ago`;
        if (diffHr < 24) return `${diffHr}h ago`;
        if (diffDay === 1) return 'yesterday';
        if (diffDay < 7) return `${diffDay}d ago`;
        return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    } catch {
        return '';
    }
}

function ConversationItem({ conv, isActive, onLoad, onRename, onDelete }) {
    const [isEditing, setIsEditing] = useState(false);
    const [editTitle, setEditTitle] = useState(conv.title);
    const [isHovered, setIsHovered] = useState(false);
    const inputRef = useRef(null);

    useEffect(() => {
        if (isEditing) inputRef.current?.focus();
    }, [isEditing]);

    const commitRename = () => {
        const trimmed = editTitle.trim();
        if (trimmed && trimmed !== conv.title) onRename(conv.id, trimmed);
        setIsEditing(false);
    };

    const cancelRename = () => {
        setEditTitle(conv.title);
        setIsEditing(false);
    };

    return (
        <motion.div
            layout
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -8 }}
            transition={{ duration: 0.15 }}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
            className={`group relative rounded-lg px-2.5 py-2 cursor-pointer transition-all ${
                isActive
                    ? 'bg-[rgba(92,245,255,0.08)] border border-[rgba(92,245,255,0.18)]'
                    : 'border border-transparent hover:bg-[rgba(255,255,255,0.04)] hover:border-[rgba(255,255,255,0.07)]'
            }`}
            onClick={() => !isEditing && onLoad(conv.id)}
        >
            <div className="flex items-start gap-2 min-w-0">
                <MessageSquare
                    size={12}
                    className={`mt-0.5 shrink-0 ${isActive ? 'text-[rgba(92,245,255,0.7)]' : 'text-[var(--text-3)]'}`}
                />
                <div className="flex-1 min-w-0">
                    {isEditing ? (
                        <div className="flex items-center gap-1" onClick={e => e.stopPropagation()}>
                            <input
                                ref={inputRef}
                                value={editTitle}
                                onChange={e => setEditTitle(e.target.value)}
                                onKeyDown={e => {
                                    if (e.key === 'Enter') commitRename();
                                    if (e.key === 'Escape') cancelRename();
                                }}
                                className="flex-1 min-w-0 bg-[rgba(0,0,0,0.3)] border border-[rgba(92,245,255,0.3)] rounded px-1.5 py-0.5 text-[11px] text-[var(--text-1)] outline-none"
                            />
                            <button onClick={commitRename} className="text-[rgba(92,245,255,0.7)] hover:text-[rgba(92,245,255,1)] transition-colors">
                                <Check size={11} />
                            </button>
                            <button onClick={cancelRename} className="text-[var(--text-3)] hover:text-[var(--text-1)] transition-colors">
                                <X size={11} />
                            </button>
                        </div>
                    ) : (
                        <p
                            className={`text-[11px] font-medium leading-tight truncate ${
                                isActive ? 'text-[rgba(92,245,255,0.9)]' : 'text-[var(--text-1)]'
                            }`}
                            onDoubleClick={e => { e.stopPropagation(); setIsEditing(true); }}
                            title="Double-click to rename"
                        >
                            {conv.title}
                        </p>
                    )}
                    <p className="text-[9px] text-[var(--text-3)] mt-0.5 font-mono">
                        {formatRelativeTime(conv.updated_at)}
                    </p>
                </div>
            </div>

            {/* Action buttons — visible on hover, not while editing */}
            {!isEditing && isHovered && (
                <div
                    className="absolute right-1.5 top-1/2 -translate-y-1/2 flex items-center gap-0.5"
                    onClick={e => e.stopPropagation()}
                >
                    <button
                        onClick={() => setIsEditing(true)}
                        className="p-1 rounded text-[var(--text-3)] hover:text-[var(--text-1)] hover:bg-[rgba(255,255,255,0.06)] transition-all"
                        title="Rename"
                    >
                        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                        </svg>
                    </button>
                    <button
                        onClick={() => onDelete(conv.id)}
                        className="p-1 rounded text-[var(--text-3)] hover:text-red-400/70 hover:bg-[rgba(239,68,68,0.08)] transition-all"
                        title="Delete"
                    >
                        <Trash2 size={10} />
                    </button>
                </div>
            )}
        </motion.div>
    );
}

function DateGroup({ label, children }) {
    return (
        <div>
            <p className="px-2 pt-3 pb-1 text-[9px] tracking-[0.25em] text-[var(--text-3)] font-mono uppercase">
                {label}
            </p>
            {children}
        </div>
    );
}

function groupConversations(conversations) {
    const now = new Date();
    const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const yesterdayStart = new Date(todayStart - 86400000);
    const weekStart = new Date(todayStart - 6 * 86400000);

    const groups = { today: [], yesterday: [], thisWeek: [], older: [] };
    for (const c of conversations) {
        const d = new Date(c.updated_at || c.created_at);
        if (d >= todayStart) groups.today.push(c);
        else if (d >= yesterdayStart) groups.yesterday.push(c);
        else if (d >= weekStart) groups.thisWeek.push(c);
        else groups.older.push(c);
    }
    return groups;
}

function ConversationSidebar({ conversations, activeConversationId, onNew, onLoad, onRename, onDelete }) {
    const groups = groupConversations(Array.isArray(conversations) ? conversations : []);
    const hasAny = conversations.length > 0;

    return (
        <motion.div
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 240, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.22, ease: 'easeInOut' }}
            className="h-full shrink-0 border-r border-[var(--border-1)] bg-[rgba(10,16,24,0.72)] backdrop-blur-md flex flex-col overflow-hidden"
        >
            {/* Header */}
            <div className="px-3 pt-4 pb-2 flex items-center justify-between shrink-0">
                <span className="text-[9px] tracking-[0.3em] text-[var(--text-3)] font-mono uppercase">
                    Conversations
                </span>
                <button
                    onClick={onNew}
                    title="New Chat"
                    className="flex items-center gap-1.5 px-2 py-1 rounded-lg border border-[rgba(92,245,255,0.2)] bg-[rgba(92,245,255,0.06)] text-[rgba(92,245,255,0.7)] hover:bg-[rgba(92,245,255,0.12)] hover:text-[rgba(92,245,255,1)] transition-all text-[10px] font-mono tracking-widest"
                >
                    <MessageSquarePlus size={11} />
                    NEW
                </button>
            </div>

            <div className="h-px w-full bg-[rgba(255,255,255,0.05)] shrink-0" />

            {/* Conversation list */}
            <div className="flex-1 overflow-y-auto px-2 pb-3">
                {!hasAny ? (
                    <div className="flex items-center justify-center h-24">
                        <p className="text-[10px] text-[var(--text-3)] font-mono tracking-widest">
                            NO CHATS YET
                        </p>
                    </div>
                ) : (
                    <div>
                        {groups.today.length > 0 && (
                            <DateGroup label="Today">
                                <AnimatePresence initial={false}>
                                    {groups.today.map(c => (
                                        <ConversationItem
                                            key={c.id}
                                            conv={c}
                                            isActive={c.id === activeConversationId}
                                            onLoad={onLoad}
                                            onRename={onRename}
                                            onDelete={onDelete}
                                        />
                                    ))}
                                </AnimatePresence>
                            </DateGroup>
                        )}
                        {groups.yesterday.length > 0 && (
                            <DateGroup label="Yesterday">
                                <AnimatePresence initial={false}>
                                    {groups.yesterday.map(c => (
                                        <ConversationItem key={c.id} conv={c} isActive={c.id === activeConversationId} onLoad={onLoad} onRename={onRename} onDelete={onDelete} />
                                    ))}
                                </AnimatePresence>
                            </DateGroup>
                        )}
                        {groups.thisWeek.length > 0 && (
                            <DateGroup label="This Week">
                                <AnimatePresence initial={false}>
                                    {groups.thisWeek.map(c => (
                                        <ConversationItem key={c.id} conv={c} isActive={c.id === activeConversationId} onLoad={onLoad} onRename={onRename} onDelete={onDelete} />
                                    ))}
                                </AnimatePresence>
                            </DateGroup>
                        )}
                        {groups.older.length > 0 && (
                            <DateGroup label="Older">
                                <AnimatePresence initial={false}>
                                    {groups.older.map(c => (
                                        <ConversationItem key={c.id} conv={c} isActive={c.id === activeConversationId} onLoad={onLoad} onRename={onRename} onDelete={onDelete} />
                                    ))}
                                </AnimatePresence>
                            </DateGroup>
                        )}
                    </div>
                )}
            </div>
        </motion.div>
    );
}

export default React.memo(ConversationSidebar);
