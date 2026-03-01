import React, { useEffect, useMemo, useState } from 'react';
import { Download, Image as ImageIcon, Sparkles, X } from 'lucide-react';

function slugify(s) {
    const t = String(s || '')
        .toLowerCase()
        .replace(/https?:\/\/\S+/g, '')
        .replace(/[^a-z0-9\s_-]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
    if (!t) return 'image';
    return t.split(' ').slice(0, 6).join('_').slice(0, 48) || 'image';
}

function formatTimestamp(d) {
    const pad = (n) => String(n).padStart(2, '0');
    const yyyy = d.getFullYear();
    const mm = pad(d.getMonth() + 1);
    const dd = pad(d.getDate());
    const hh = pad(d.getHours());
    const mi = pad(d.getMinutes());
    const ss = pad(d.getSeconds());
    return `${yyyy}${mm}${dd}_${hh}${mi}${ss}`;
}

const SIZE_OPTIONS = [
    { value: '1664*928', label: '1664×928 (16:9)' },
    { value: '1472*1104', label: '1472×1104 (4:3)' },
    { value: '1328*1328', label: '1328×1328 (1:1)' },
    { value: '1104*1472', label: '1104×1472 (3:4)' },
    { value: '928*1664', label: '928×1664 (9:16)' },
];

const PRESETS = {
    fast: { steps: 10, cfg: 6.5, size: '1328*1328' },
    balanced: { steps: 20, cfg: 7.0, size: '1664*928' },
    quality: { steps: 30, cfg: 7.0, size: '1472*1104' },
};

const ImageStudioWindow = ({ data, status, phase, phaseMs, progress, etaMs, benchmark, onClose, socket }) => {
    const [prompt, setPrompt] = useState('');
    const [negativePrompt, setNegativePrompt] = useState('');
    const [size, setSize] = useState('1664*928');
    const [seed, setSeed] = useState('');
    const [steps, setSteps] = useState(20);
    const [cfgScale, setCfgScale] = useState(7.0);
    const [promptExtend, setPromptExtend] = useState(true);
    const [watermark, setWatermark] = useState(false);
    const [isSending, setIsSending] = useState(false);
    const [error, setError] = useState('');

    useEffect(() => {
        if (status === 'generating') {
            setIsSending(true);
        } else {
            setIsSending(false);
        }
    }, [status]);

    const imgSrc = useMemo(() => {
        if (!data || !data.image_b64) return null;
        const fmt = String(data.format || 'png').toLowerCase();
        const mime = fmt === 'jpg' ? 'jpeg' : fmt;
        return `data:image/${mime};base64,${data.image_b64}`;
    }, [data]);

    const canDownload = Boolean(data && data.image_b64);

    const backendErrorText = useMemo(() => {
        const e = data?._error;
        if (!e || typeof e !== 'object') return '';
        const parts = [];
        const hs = e.http_status ? String(e.http_status) : '';
        const code = e.code ? String(e.code) : '';
        const rid = e.request_id ? String(e.request_id) : '';
        const msg = e.error ? String(e.error) : '';

        if (hs || code) parts.push(`${hs ? `HTTP ${hs}` : ''}${hs && code ? ' ' : ''}${code ? `(${code})` : ''}`.trim());
        if (msg) parts.push(msg);
        if (rid) parts.push(`request_id: ${rid}`);
        return parts.filter(Boolean).join('\n');
    }, [data]);

    const handleGenerate = () => {
        const p = String(prompt || '').trim();
        if (!p) return;

        setError('');

        if (!socket || !socket.connected) {
            setIsSending(false);
            setError('Backend offline (socket not connected).');
            return;
        }
        setIsSending(true);

        let seedVal = null;
        const s = String(seed || '').trim();
        if (s) {
            const n = Number(s);
            if (Number.isFinite(n)) seedVal = Math.max(0, Math.floor(n));
        }

        try {
            socket.emit('generate_image', {
                prompt: p,
                negative_prompt: String(negativePrompt || '').trim() || null,
                size,
                seed: seedVal,
                steps: Number.isFinite(Number(steps)) ? Math.max(1, Math.min(60, Math.floor(Number(steps)))) : 20,
                cfg_scale: Number.isFinite(Number(cfgScale)) ? Math.max(1, Math.min(20, Number(cfgScale))) : 7.0,
                prompt_extend: Boolean(promptExtend),
                watermark: Boolean(watermark),
            });
        } catch (e) {
            setIsSending(false);
            setError(`Failed to send request: ${e?.message || String(e)}`);
        }
    };

    const handleBenchmark = () => {
        setError('');
        if (!socket || !socket.connected) {
            setError('Backend offline (socket not connected).');
            return;
        }
        try {
            socket.emit('image_benchmark', {
                prompt: String(prompt || '').trim() || undefined,
                negative_prompt: String(negativePrompt || '').trim() || undefined,
                size,
            });
        } catch (e) {
            setError(`Failed to start benchmark: ${e?.message || String(e)}`);
        }
    };

    const phaseLine = useMemo(() => {
        const p = String(phase || '').trim();
        if (!p) return '';
        const ms = typeof phaseMs === 'number' && Number.isFinite(phaseMs) ? phaseMs : null;
        return ms != null ? `${p.toUpperCase()} · ${Math.round(ms)}ms` : p.toUpperCase();
    }, [phase, phaseMs]);

    const progressUi = useMemo(() => {
        const p = typeof progress === 'number' && Number.isFinite(progress) ? Math.max(0, Math.min(1, progress)) : null;
        const eta = typeof etaMs === 'number' && Number.isFinite(etaMs) ? Math.max(0, etaMs) : null;
        const pct = p != null ? Math.round(p * 100) : null;
        const etaS = eta != null ? Math.max(0, Math.round(eta / 1000)) : null;
        return { p, pct, etaS };
    }, [progress, etaMs]);

    const timingLines = useMemo(() => {
        const t = data?.timings;
        if (!t || typeof t !== 'object') return [];
        const out = [];
        if (typeof t.ping_ms === 'number') out.push(`PING: ${Math.round(t.ping_ms)}ms`);
        if (typeof t.inference_ms === 'number') out.push(`INFER: ${Math.round(t.inference_ms)}ms`);
        if (typeof t.total_ms === 'number') out.push(`TOTAL: ${Math.round(t.total_ms)}ms`);
        if (typeof t.backend_total_ms === 'number') out.push(`BACKEND: ${Math.round(t.backend_total_ms)}ms`);
        return out;
    }, [data]);

    const handleDownload = () => {
        if (!canDownload) return;

        try {
            const fmt = String(data.format || 'png').toLowerCase();
            const mime = fmt === 'jpg' ? 'image/jpeg' : fmt === 'webp' ? 'image/webp' : 'image/png';
            const bin = atob(data.image_b64);
            const bytes = new Uint8Array(bin.length);
            for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);

            const blob = new Blob([bytes], { type: mime });
            const url = URL.createObjectURL(blob);
            const ts = formatTimestamp(new Date());
            const name = `qwen_image_${ts}_${slugify(data.prompt || prompt)}.${fmt}`;

            const a = document.createElement('a');
            a.href = url;
            a.download = name;
            document.body.appendChild(a);
            a.click();
            a.remove();

            setTimeout(() => URL.revokeObjectURL(url), 1500);
        } catch (e) {
            console.error('Download failed:', e);
        }
    };

    return (
        <div className="w-full h-full relative bg-[#0B0F14] rounded-lg overflow-hidden border border-[rgba(92,245,255,0.18)] shadow-[0_0_30px_rgba(6,182,212,0.10)]">
            <div className="absolute inset-0 pointer-events-none opacity-10 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] mix-blend-overlay" />

            <div data-drag-handle className="h-9 bg-[rgba(15,22,32,0.72)] border-b border-[rgba(92,245,255,0.14)] backdrop-blur-md flex items-center justify-between px-3 cursor-grab active:cursor-grabbing">
                <div className="flex items-center gap-2 text-[11px] font-mono tracking-widest text-[rgba(92,245,255,0.75)]">
                    <ImageIcon size={14} />
                    <span>IMAGE_STUDIO</span>
                    <span className="text-[10px] text-[rgba(255,255,255,0.35)] tracking-[0.24em]">GENERATOR</span>
                </div>
                <button onClick={onClose} className="hover:bg-red-500/20 text-gray-400 hover:text-red-300 p-1.5 rounded transition-colors">
                    <X size={14} />
                </button>
            </div>

            <div className="h-[calc(100%-36px)] grid grid-cols-1 lg:grid-cols-2">
                <div className="relative bg-[radial-gradient(800px_circle_at_50%_35%,rgba(92,245,255,0.08),rgba(11,15,20,0.96)_55%,rgba(11,15,20,1)_100%)] border-b lg:border-b-0 lg:border-r border-[rgba(92,245,255,0.10)] flex items-center justify-center overflow-hidden">
                    {imgSrc ? (
                        <img
                            src={imgSrc}
                            alt="Generated"
                            className="max-w-full max-h-full object-contain"
                        />
                    ) : (
                        <div className="text-center px-6">
                            <div className="inline-flex items-center justify-center w-12 h-12 rounded-2xl border border-[rgba(92,245,255,0.20)] bg-[rgba(92,245,255,0.06)] mb-3">
                                <Sparkles size={20} className="text-[rgba(92,245,255,0.75)]" />
                            </div>
                            <div className="text-[12px] font-mono tracking-widest text-[rgba(255,255,255,0.45)]">NO IMAGE YET</div>
                            <div className="text-[11px] mt-2 text-[rgba(255,255,255,0.35)]">Generate an image to preview it here.</div>
                        </div>
                    )}

                    {(status === 'generating' || isSending) && (
                        <div className="absolute inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center">
                            <div className="px-4 py-2 rounded-xl border border-[rgba(92,245,255,0.22)] bg-[rgba(15,22,32,0.72)] text-[11px] font-mono tracking-widest text-[rgba(92,245,255,0.75)] animate-pulse">
                                GENERATING...
                            </div>
                        </div>
                    )}
                </div>

                <div className="p-4 lg:p-5 overflow-y-auto">
                    <div className="space-y-3">
                        {phaseLine && (
                            <div className="rounded-xl border border-[rgba(92,245,255,0.14)] bg-[rgba(92,245,255,0.06)] px-3 py-2 text-[10px] font-mono tracking-widest text-[rgba(92,245,255,0.70)]">
                                {phaseLine}
                            </div>
                        )}

                        {(status === 'generating' && progressUi.p != null) && (
                            <div className="rounded-xl border border-[rgba(255,255,255,0.10)] bg-[rgba(0,0,0,0.18)] px-3 py-2">
                                <div className="flex items-center justify-between text-[10px] font-mono tracking-widest text-[rgba(255,255,255,0.55)]">
                                    <span>PROGRESS</span>
                                    <span>
                                        {progressUi.pct != null ? `${progressUi.pct}%` : ''}
                                        {progressUi.etaS != null ? ` · ETA ${progressUi.etaS}s` : ''}
                                    </span>
                                </div>
                                <div className="mt-2 h-2 rounded-full bg-[rgba(255,255,255,0.06)] overflow-hidden">
                                    <div
                                        className="h-full bg-[rgba(92,245,255,0.55)]"
                                        style={{ width: `${progressUi.pct ?? 0}%` }}
                                    />
                                </div>
                            </div>
                        )}

                        {timingLines.length > 0 && (
                            <div className="text-[10px] font-mono tracking-widest text-[rgba(255,255,255,0.45)] whitespace-pre-wrap">
                                {timingLines.join(' · ')}
                            </div>
                        )}

                        <div>
                            <label className="block text-[10px] uppercase tracking-widest text-[rgba(92,245,255,0.55)] mb-1">Prompt</label>
                            <textarea
                                value={prompt}
                                onChange={(e) => setPrompt(e.target.value)}
                                placeholder="Describe the image you want..."
                                className="w-full h-24 resize-none rounded-xl bg-[rgba(0,0,0,0.25)] border border-[rgba(92,245,255,0.12)] focus:border-[rgba(92,245,255,0.35)] outline-none p-3 text-[12px] text-[rgba(255,255,255,0.85)]"
                                onKeyDown={(e) => {
                                    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                                        e.preventDefault();
                                        handleGenerate();
                                    }
                                }}
                            />
                        </div>

                        <div>
                            <label className="block text-[10px] uppercase tracking-widest text-[rgba(255,255,255,0.35)] mb-1">Negative prompt (optional)</label>
                            <textarea
                                value={negativePrompt}
                                onChange={(e) => setNegativePrompt(e.target.value)}
                                placeholder="What should be avoided..."
                                className="w-full h-16 resize-none rounded-xl bg-[rgba(0,0,0,0.20)] border border-[rgba(255,255,255,0.08)] focus:border-[rgba(92,245,255,0.28)] outline-none p-3 text-[12px] text-[rgba(255,255,255,0.70)]"
                            />
                        </div>

                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                            <div>
                                <label className="block text-[10px] uppercase tracking-widest text-[rgba(255,255,255,0.35)] mb-1">Size</label>
                                <select
                                    value={size}
                                    onChange={(e) => setSize(e.target.value)}
                                    className="w-full rounded-xl bg-[rgba(0,0,0,0.20)] border border-[rgba(255,255,255,0.08)] focus:border-[rgba(92,245,255,0.28)] outline-none p-2 text-[12px] text-[rgba(255,255,255,0.75)]"
                                >
                                    {SIZE_OPTIONS.map(opt => (
                                        <option key={opt.value} value={opt.value}>{opt.label}</option>
                                    ))}
                                </select>
                            </div>

                            <div>
                                <label className="block text-[10px] uppercase tracking-widest text-[rgba(255,255,255,0.35)] mb-1">Seed (optional)</label>
                                <input
                                    value={seed}
                                    onChange={(e) => setSeed(e.target.value)}
                                    placeholder="e.g. 42"
                                    className="w-full rounded-xl bg-[rgba(0,0,0,0.20)] border border-[rgba(255,255,255,0.08)] focus:border-[rgba(92,245,255,0.28)] outline-none p-2 text-[12px] text-[rgba(255,255,255,0.75)]"
                                />
                            </div>
                        </div>

                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                            <div>
                                <label className="block text-[10px] uppercase tracking-widest text-[rgba(255,255,255,0.35)] mb-1">Steps</label>
                                <input
                                    type="number"
                                    min={1}
                                    max={60}
                                    value={steps}
                                    onChange={(e) => setSteps(e.target.value)}
                                    className="w-full rounded-xl bg-[rgba(0,0,0,0.20)] border border-[rgba(255,255,255,0.08)] focus:border-[rgba(92,245,255,0.28)] outline-none p-2 text-[12px] text-[rgba(255,255,255,0.75)]"
                                />
                            </div>

                            <div>
                                <label className="block text-[10px] uppercase tracking-widest text-[rgba(255,255,255,0.35)] mb-1">CFG</label>
                                <input
                                    type="number"
                                    step={0.5}
                                    min={1}
                                    max={20}
                                    value={cfgScale}
                                    onChange={(e) => setCfgScale(e.target.value)}
                                    className="w-full rounded-xl bg-[rgba(0,0,0,0.20)] border border-[rgba(255,255,255,0.08)] focus:border-[rgba(92,245,255,0.28)] outline-none p-2 text-[12px] text-[rgba(255,255,255,0.75)]"
                                />
                            </div>
                        </div>

                        <div className="grid grid-cols-3 gap-2">
                            <button
                                type="button"
                                onClick={() => {
                                    setSteps(PRESETS.fast.steps);
                                    setCfgScale(PRESETS.fast.cfg);
                                    setSize(PRESETS.fast.size);
                                }}
                                className="rounded-xl px-3 py-2 border text-[11px] font-mono tracking-widest transition-colors border-[rgba(255,255,255,0.10)] bg-[rgba(0,0,0,0.20)] text-[rgba(255,255,255,0.65)] hover:bg-[rgba(255,255,255,0.06)]"
                            >
                                FAST
                            </button>
                            <button
                                type="button"
                                onClick={() => {
                                    setSteps(PRESETS.balanced.steps);
                                    setCfgScale(PRESETS.balanced.cfg);
                                    setSize(PRESETS.balanced.size);
                                }}
                                className="rounded-xl px-3 py-2 border text-[11px] font-mono tracking-widest transition-colors border-[rgba(92,245,255,0.22)] bg-[rgba(92,245,255,0.08)] text-[rgba(92,245,255,0.80)] hover:bg-[rgba(92,245,255,0.12)]"
                            >
                                BAL
                            </button>
                            <button
                                type="button"
                                onClick={() => {
                                    setSteps(PRESETS.quality.steps);
                                    setCfgScale(PRESETS.quality.cfg);
                                    setSize(PRESETS.quality.size);
                                }}
                                className="rounded-xl px-3 py-2 border text-[11px] font-mono tracking-widest transition-colors border-[rgba(255,255,255,0.10)] bg-[rgba(0,0,0,0.20)] text-[rgba(255,255,255,0.65)] hover:bg-[rgba(255,255,255,0.06)]"
                            >
                                HQ
                            </button>
                        </div>

                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                            <button
                                type="button"
                                onClick={() => setPromptExtend(v => !v)}
                                className={`rounded-xl border px-3 py-2 text-left text-[12px] transition-colors ${promptExtend
                                    ? 'border-[rgba(92,245,255,0.28)] bg-[rgba(92,245,255,0.08)] text-[rgba(92,245,255,0.85)]'
                                    : 'border-[rgba(255,255,255,0.10)] bg-[rgba(0,0,0,0.20)] text-[rgba(255,255,255,0.65)]'
                                }`}
                            >
                                <div className="text-[10px] uppercase tracking-widest opacity-80">Prompt Extend</div>
                                <div className="mt-0.5 opacity-80">{promptExtend ? 'On' : 'Off'}</div>
                            </button>

                            <button
                                type="button"
                                onClick={() => setWatermark(v => !v)}
                                className={`rounded-xl border px-3 py-2 text-left text-[12px] transition-colors ${watermark
                                    ? 'border-[rgba(92,245,255,0.28)] bg-[rgba(92,245,255,0.08)] text-[rgba(92,245,255,0.85)]'
                                    : 'border-[rgba(255,255,255,0.10)] bg-[rgba(0,0,0,0.20)] text-[rgba(255,255,255,0.65)]'
                                }`}
                            >
                                <div className="text-[10px] uppercase tracking-widest opacity-80">Watermark</div>
                                <div className="mt-0.5 opacity-80">{watermark ? 'On' : 'Off'}</div>
                            </button>
                        </div>

                        <div className="flex flex-col sm:flex-row gap-2 pt-1">
                            <button
                                type="button"
                                onClick={handleGenerate}
                                disabled={isSending || !String(prompt || '').trim()}
                                className={`flex-1 rounded-xl px-3 py-2 border text-[12px] font-mono tracking-widest transition-colors ${isSending
                                    ? 'border-[rgba(255,255,255,0.10)] bg-[rgba(255,255,255,0.03)] text-[rgba(255,255,255,0.35)]'
                                    : 'border-[rgba(92,245,255,0.35)] bg-[rgba(92,245,255,0.12)] text-[rgba(92,245,255,0.90)] hover:bg-[rgba(92,245,255,0.18)]'
                                }`}
                            >
                                {isSending ? 'WORKING…' : 'GENERATE'}
                            </button>

                            <button
                                type="button"
                                onClick={handleBenchmark}
                                className="rounded-xl px-3 py-2 border text-[12px] font-mono tracking-widest flex items-center justify-center gap-2 transition-colors border-[rgba(255,255,255,0.10)] bg-[rgba(0,0,0,0.20)] text-[rgba(255,255,255,0.65)] hover:bg-[rgba(255,255,255,0.06)]"
                            >
                                BENCH
                            </button>

                            <button
                                type="button"
                                onClick={handleDownload}
                                disabled={!canDownload}
                                className={`rounded-xl px-3 py-2 border text-[12px] font-mono tracking-widest flex items-center justify-center gap-2 transition-colors ${canDownload
                                    ? 'border-[rgba(255,255,255,0.14)] bg-[rgba(0,0,0,0.22)] text-[rgba(255,255,255,0.75)] hover:bg-[rgba(255,255,255,0.06)]'
                                    : 'border-[rgba(255,255,255,0.08)] bg-[rgba(255,255,255,0.02)] text-[rgba(255,255,255,0.30)]'
                                }`}
                            >
                                <Download size={14} />
                                DOWNLOAD
                            </button>
                        </div>

                        {(error || status === 'error') && (
                            <div className="mt-2 rounded-xl border border-[rgba(255,90,122,0.35)] bg-[rgba(255,90,122,0.08)] px-3 py-2 text-[11px] text-[rgba(255,255,255,0.75)]">
                                <div className="whitespace-pre-wrap">{error || backendErrorText || 'Image generation failed. Check backend logs.'}</div>
                            </div>
                        )}

                        {status === 'generating' && !error && (
                            <div className="mt-2 text-[10px] font-mono tracking-widest text-[rgba(92,245,255,0.55)]">
                                RUNNING…
                            </div>
                        )}

                        {data?.model && (
                            <div className="pt-2 text-[10px] font-mono text-[rgba(255,255,255,0.35)]">
                                MODEL: {String(data.model)}
                            </div>
                        )}

                        {benchmark && (
                            <div className="pt-2 text-[10px] font-mono text-[rgba(255,255,255,0.45)] whitespace-pre-wrap">
                                {(() => {
                                    try {
                                        if (typeof benchmark === 'string') return benchmark;
                                        const p = benchmark?.provider ? String(benchmark.provider) : 'provider';
                                        if (benchmark?.cold && benchmark?.warm) {
                                            const c = Math.round(Number(benchmark.cold.total_ms || 0));
                                            const w = Math.round(Number(benchmark.warm.total_ms || 0));
                                            return `BENCHMARK (${p})\nCOLD: ${c}ms\nWARM: ${w}ms`;
                                        }
                                        if (benchmark?.note) return String(benchmark.note);
                                        return JSON.stringify(benchmark, null, 2);
                                    } catch (e) {
                                        return '';
                                    }
                                })()}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ImageStudioWindow;
