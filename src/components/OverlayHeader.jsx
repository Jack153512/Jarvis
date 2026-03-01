import React from 'react';
import { X } from 'lucide-react';

function OverlayHeader({ icon, title, onClose, isDragging }) {
    return (
        <div
            data-drag-handle
            className={`h-9 shrink-0 flex items-center justify-between px-3 border-b border-[rgba(255,255,255,0.07)] cursor-grab active:cursor-grabbing transition-colors ${
                isDragging ? 'bg-[rgba(0,0,0,0.65)]' : 'bg-[rgba(0,0,0,0.45)]'
            }`}
        >
            <div className="flex items-center gap-2">
                <span className="text-[var(--accent)] opacity-60">{icon}</span>
                <span className="text-[11px] font-bold tracking-[0.22em] text-[var(--accent)] opacity-70">
                    {title}
                </span>
            </div>
            <button
                onClick={onClose}
                title="Close"
                className="w-6 h-6 flex items-center justify-center rounded text-white/30 hover:text-red-400 hover:bg-red-500/15 transition-colors"
            >
                <X size={13} />
            </button>
        </div>
    );
}

export default OverlayHeader;
