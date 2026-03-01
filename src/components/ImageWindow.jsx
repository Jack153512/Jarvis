import React, { useState, useEffect, useRef } from 'react';
import { Download, RefreshCw, ImageIcon, AlertTriangle, Sparkles } from 'lucide-react';

/* ─── Prompt overlay ─────────────────────────────────────────────────── */

function PromptOverlay({ isNew, onSubmit, onCancel, isLoading }) {
    const [prompt, setPrompt] = useState('');
    const [negative, setNegative] = useState('');
    const [size, setSize] = useState('512x512');
    const [steps, setSteps] = useState(20);
    const [showAdvanced, setShowAdvanced] = useState(false);

    const handleKey = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (prompt.trim()) onSubmit({ prompt: prompt.trim(), negative_prompt: negative.trim(), size, steps });
        }
        if (e.key === 'Escape' && !isNew) onCancel();
    };

    return (
        <div className="absolute inset-0 z-30 flex items-center justify-center bg-black/70 backdrop-blur-sm p-5">
            <div className="w-full max-w-sm rounded-xl border border-violet-500/30 bg-[rgba(8,6,20,0.95)] shadow-[0_0_40px_rgba(139,92,246,0.12)] p-5 space-y-4">
                {/* Header */}
                <div className="flex items-center gap-2">
                    <Sparkles size={13} className="text-violet-400/70" />
                    <span className="text-[11px] font-bold tracking-[0.2em] text-violet-400/80">
                        {isNew ? 'CREATE IMAGE' : 'REGENERATE'}
                    </span>
                </div>

                {/* Prompt */}
                <div className="space-y-1.5">
                    <label className="text-[10px] tracking-widest text-white/30 font-mono">PROMPT</label>
                    <textarea
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        onKeyDown={handleKey}
                        autoFocus
                        rows={3}
                        placeholder="Describe what you want to create…"
                        className="w-full resize-none rounded-lg border border-white/10 bg-white/5 px-3 py-2.5 text-sm text-white/90 placeholder:text-white/25 outline-none focus:border-violet-500/50 transition-colors leading-relaxed"
                    />
                </div>

                {/* Advanced toggle */}
                <button
                    type="button"
                    onClick={() => setShowAdvanced(v => !v)}
                    className="text-[10px] tracking-widest text-white/25 hover:text-white/50 transition-colors font-mono"
                >
                    {showAdvanced ? '▲ HIDE OPTIONS' : '▼ ADVANCED OPTIONS'}
                </button>

                {showAdvanced && (
                    <div className="space-y-3">
                        {/* Negative prompt */}
                        <div className="space-y-1.5">
                            <label className="text-[10px] tracking-widest text-white/30 font-mono">NEGATIVE PROMPT</label>
                            <textarea
                                value={negative}
                                onChange={(e) => setNegative(e.target.value)}
                                rows={2}
                                placeholder="What to avoid…"
                                className="w-full resize-none rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm text-white/90 placeholder:text-white/25 outline-none focus:border-violet-500/50 transition-colors leading-relaxed"
                            />
                        </div>
                        {/* Size + Steps */}
                        <div className="grid grid-cols-2 gap-3">
                            <div className="space-y-1.5">
                                <label className="text-[10px] tracking-widest text-white/30 font-mono">SIZE</label>
                                <select
                                    value={size}
                                    onChange={(e) => setSize(e.target.value)}
                                    className="w-full rounded-lg border border-white/10 bg-[rgba(0,0,0,0.4)] px-2.5 py-2 text-xs text-white/80 outline-none focus:border-violet-500/50 transition-colors"
                                >
                                    <option value="512x512">512 × 512</option>
                                    <option value="768x512">768 × 512</option>
                                    <option value="512x768">512 × 768</option>
                                    <option value="1024x1024">1024 × 1024</option>
                                    <option value="1024x576">1024 × 576</option>
                                </select>
                            </div>
                            <div className="space-y-1.5">
                                <label className="text-[10px] tracking-widest text-white/30 font-mono">STEPS</label>
                                <input
                                    type="number"
                                    value={steps}
                                    onChange={(e) => setSteps(Math.max(1, Math.min(150, Number(e.target.value))))}
                                    min={1}
                                    max={150}
                                    className="w-full rounded-lg border border-white/10 bg-[rgba(0,0,0,0.4)] px-2.5 py-2 text-xs text-white/80 outline-none focus:border-violet-500/50 transition-colors"
                                />
                            </div>
                        </div>
                    </div>
                )}

                {/* Actions */}
                <div className="flex justify-end gap-2 pt-1">
                    {!isNew && (
                        <button
                            onClick={onCancel}
                            className="px-3 py-1.5 text-xs text-white/40 hover:text-white/70 transition-colors rounded-lg hover:bg-white/5"
                        >
                            Cancel
                        </button>
                    )}
                    <button
                        onClick={() => prompt.trim() && onSubmit({ prompt: prompt.trim(), negative_prompt: negative.trim(), size, steps })}
                        disabled={!prompt.trim() || isLoading}
                        className="px-4 py-1.5 text-xs font-semibold rounded-lg bg-violet-500/20 border border-violet-500/40 text-violet-300 hover:bg-violet-500/30 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
                    >
                        {isLoading ? 'Generating…' : 'Generate'}
                    </button>
                </div>
            </div>
        </div>
    );
}

/* ─── Loading placeholder ────────────────────────────────────────────── */

function LoadingView({ elapsed }) {
    return (
        <div className="w-full h-full flex flex-col items-center justify-center gap-5 bg-[#09060f]">
            {/* Pulsing grid */}
            <div className="relative w-32 h-32">
                <div className="absolute inset-0 rounded-xl border border-violet-500/20 animate-pulse" />
                <div className="absolute inset-3 rounded-lg border border-violet-500/15 animate-pulse [animation-delay:150ms]" />
                <div className="absolute inset-6 rounded-md border border-violet-500/10 animate-pulse [animation-delay:300ms]" />
                <div className="absolute inset-0 flex items-center justify-center">
                    <Sparkles size={28} className="text-violet-400/40 animate-pulse" />
                </div>
            </div>
            <div className="space-y-1 text-center">
                <p className="text-[11px] font-bold tracking-[0.3em] text-violet-400/70 font-mono">RENDERING</p>
                <p className="text-[10px] text-white/25 font-mono">{elapsed}s elapsed</p>
            </div>
        </div>
    );
}

/* ─── Toolbar ────────────────────────────────────────────────────────── */

function Toolbar({ onRegenerate, onDownload }) {
    return (
        <div className="absolute top-2.5 left-3 z-10 flex items-center gap-1.5">
            <button
                onClick={onRegenerate}
                title="Regenerate"
                className="flex items-center gap-1.5 rounded-lg border border-white/[0.12] bg-black/40 backdrop-blur-sm px-2.5 py-1.5 text-[11px] font-semibold tracking-wide text-white/60 hover:text-violet-300 hover:border-violet-500/40 hover:bg-violet-500/10 transition-all"
            >
                <RefreshCw size={11} />
                REGENERATE
            </button>
            <button
                onClick={onDownload}
                title="Download PNG"
                className="flex items-center gap-1.5 rounded-lg border border-white/[0.12] bg-black/40 backdrop-blur-sm px-2.5 py-1.5 text-[11px] font-semibold tracking-wide text-white/60 hover:text-emerald-300 hover:border-emerald-500/40 hover:bg-emerald-500/10 transition-all"
            >
                <Download size={11} />
                DOWNLOAD
            </button>
        </div>
    );
}

/* ─── Main component ─────────────────────────────────────────────────── */

const ImageWindow = ({ data, socket }) => {
    const [isRegenerating, setIsRegenerating] = useState(false);
    const [elapsed, setElapsed] = useState(0);
    const startRef = useRef(null);

    const isLoading = data?.status === 'loading';
    const hasImage = data?.status === 'done' && Boolean(data?.image_b64);
    const hasError = data?.status === 'error';
    const showPrompt = isRegenerating || (!isLoading && !hasImage && !hasError);

    /* Elapsed timer while loading */
    useEffect(() => {
        if (isLoading) {
            startRef.current = Date.now();
            const id = setInterval(() => setElapsed(Math.floor((Date.now() - startRef.current) / 1000)), 250);
            return () => clearInterval(id);
        }
        if (!isLoading) setElapsed(0);
    }, [isLoading]);

    const handleGenerate = (opts) => {
        socket?.emit('generate_image', opts);
        setIsRegenerating(false);
    };

    const handleDownload = () => {
        if (!data?.image_b64) return;
        const fmt = data.format || 'png';
        const bytes = Uint8Array.from(atob(data.image_b64), (c) => c.charCodeAt(0));
        const blob = new Blob([bytes], { type: `image/${fmt}` });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `image.${fmt}`;
        a.click();
        URL.revokeObjectURL(url);
    };

    return (
        <div className="w-full h-full relative flex flex-col overflow-hidden bg-[#09060f]">

            {/* ── Toolbar (only when image visible) ── */}
            {hasImage && !showPrompt && (
                <Toolbar onRegenerate={() => setIsRegenerating(true)} onDownload={handleDownload} />
            )}

            {/* ── Status badge ── */}
            <div className="absolute bottom-3 right-3 z-10 text-[10px] font-mono tracking-widest text-violet-500/30 pointer-events-none">
                IMAGE · {data?.format?.toUpperCase() ?? (isLoading ? 'RENDERING' : 'READY')}
            </div>

            {/* ── Prompt overlay ── */}
            {showPrompt && (
                <PromptOverlay
                    isNew={!hasImage}
                    onSubmit={handleGenerate}
                    onCancel={() => setIsRegenerating(false)}
                    isLoading={isLoading}
                />
            )}

            {/* ── Loading view ── */}
            {isLoading && !showPrompt && <LoadingView elapsed={elapsed} />}

            {/* ── Error banner ── */}
            {hasError && !showPrompt && (
                <div className="flex-1 flex items-center justify-center p-6">
                    <div className="flex items-start gap-3 rounded-xl border border-red-500/25 bg-red-500/10 px-4 py-3 max-w-sm w-full">
                        <AlertTriangle size={14} className="text-red-400 mt-0.5 shrink-0" />
                        <div className="space-y-1 min-w-0 flex-1">
                            <p className="text-[11px] font-bold text-red-400/90 tracking-wide">GENERATION FAILED</p>
                            <p className="text-[11px] text-red-400/70 leading-relaxed break-words">
                                {data?.message || 'Backend did not return an error message — check the terminal for the full traceback.'}
                            </p>
                            {data?.traceback && (
                                <details className="mt-1">
                                    <summary className="text-[10px] text-red-400/50 hover:text-red-400/80 cursor-pointer font-mono tracking-wide">
                                        SHOW TRACEBACK
                                    </summary>
                                    <pre className="mt-1 text-[9px] text-red-400/60 leading-relaxed whitespace-pre-wrap break-all max-h-40 overflow-y-auto">
                                        {data.traceback}
                                    </pre>
                                </details>
                            )}
                            <button
                                onClick={() => { setIsRegenerating(true); }}
                                disabled={isLoading}
                                className="mt-2 text-[10px] tracking-widest text-red-400/60 hover:text-red-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors font-mono"
                            >
                                TRY AGAIN →
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* ── Image display ── */}
            {hasImage && !showPrompt && (
                <div className="flex-1 min-h-0 flex items-center justify-center p-3">
                    <img
                        src={`data:image/${data.format || 'png'};base64,${data.image_b64}`}
                        alt={data.prompt || 'Generated image'}
                        className="max-w-full max-h-full object-contain rounded-lg shadow-[0_0_30px_rgba(139,92,246,0.12)]"
                    />
                </div>
            )}

            {/* ── Prompt caption ── */}
            {hasImage && !showPrompt && data?.prompt && (
                <div className="shrink-0 px-3 pb-3 pt-1">
                    <p className="text-[10px] text-white/25 font-mono truncate" title={data.prompt}>
                        {data.prompt}
                    </p>
                </div>
            )}
        </div>
    );
};

export default ImageWindow;
