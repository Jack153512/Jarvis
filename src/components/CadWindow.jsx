import React, { useMemo, useState, useEffect, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Center, Stage } from '@react-three/drei';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader';
import { Download, RefreshCw, RotateCcw, Layers, AlertTriangle } from 'lucide-react';

/* ─── 3-D helpers ────────────────────────────────────────────────────── */

const GeometryModel = ({ geometry }) => (
    <mesh geometry={geometry} castShadow receiveShadow>
        <meshStandardMaterial color="#06b6d4" roughness={0.25} metalness={0.75} />
    </mesh>
);

const LoadingCube = () => {
    const ref = useRef();
    useFrame((_, delta) => {
        ref.current.rotation.x += delta * 0.9;
        ref.current.rotation.y += delta * 0.9;
    });
    return (
        <mesh ref={ref}>
            <boxGeometry args={[10, 10, 10]} />
            <meshStandardMaterial wireframe color="#06b6d4" transparent opacity={0.45} />
        </mesh>
    );
};

/* ─── Stage label mapping ────────────────────────────────────────────── */

const STAGES = ['Starting', 'Sampling', 'Decoding', 'Exporting', 'Finalizing'];

function inferStage(thoughts) {
    const t = String(thoughts || '').toLowerCase();
    if (!t) return { label: 'Starting', idx: 0 };
    if (t.includes('sampling') || t.includes('latent')) return { label: 'Sampling', idx: 1 };
    if (t.includes('decod')) return { label: 'Decoding', idx: 2 };
    if (t.includes('export')) return { label: 'Exporting', idx: 3 };
    if (t.includes('success') || t.includes('completed') || t.includes('stl created')) return { label: 'Finalizing', idx: 4 };
    return { label: 'Processing', idx: 1 };
}

/* ─── Progress bar ───────────────────────────────────────────────────── */

function ProgressBar({ idx, label, elapsed }) {
    const pct = Math.min(100, Math.max(6, ((idx + 1) / STAGES.length) * 100));
    return (
        <div className="space-y-1">
            <div className="h-1 w-full rounded-full bg-white/10 overflow-hidden">
                <div
                    className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-emerald-400 transition-all duration-700"
                    style={{ width: `${pct}%` }}
                />
            </div>
            <div className="flex justify-between text-[10px] font-mono text-white/35">
                <span>{elapsed}s elapsed</span>
                <span className="text-cyan-400/70 tracking-widest">{label.toUpperCase()}</span>
            </div>
        </div>
    );
}

/* ─── Prompt overlay ─────────────────────────────────────────────────── */

function PromptOverlay({ isNew, onSubmit, onCancel, isLoading }) {
    const [value, setValue] = useState('');

    const handleKey = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (value.trim()) onSubmit(value.trim());
        }
        if (e.key === 'Escape' && !isNew) onCancel();
    };

    return (
        <div className="absolute inset-0 z-30 flex items-center justify-center bg-black/70 backdrop-blur-sm p-5">
            <div className="w-full max-w-sm rounded-xl border border-cyan-500/30 bg-[rgba(8,18,28,0.92)] shadow-[0_0_40px_rgba(6,182,212,0.12)] p-5 space-y-4">
                {/* Header */}
                <div className="flex items-center gap-2">
                    <Layers size={14} className="text-cyan-400/70" />
                    <span className="text-[11px] font-bold tracking-[0.2em] text-cyan-400/80">
                        {isNew ? 'NEW DESIGN' : 'REFINE DESIGN'}
                    </span>
                </div>

                {/* Input */}
                <textarea
                    value={value}
                    onChange={(e) => setValue(e.target.value)}
                    onKeyDown={handleKey}
                    autoFocus
                    rows={4}
                    placeholder={isNew
                        ? 'Describe what you want to create…'
                        : 'e.g. Make the walls thicker, add mounting holes…'
                    }
                    className="w-full resize-none rounded-lg border border-white/10 bg-white/5 px-3 py-2.5 text-sm text-white/90 placeholder:text-white/25 outline-none focus:border-cyan-500/50 transition-colors leading-relaxed"
                />

                {/* Actions */}
                <div className="flex justify-end gap-2">
                    {!isNew && (
                        <button
                            onClick={onCancel}
                            className="px-3 py-1.5 text-xs text-white/40 hover:text-white/70 transition-colors rounded-lg hover:bg-white/5"
                        >
                            Cancel
                        </button>
                    )}
                    <button
                        onClick={() => value.trim() && onSubmit(value.trim())}
                        disabled={!value.trim() || isLoading}
                        className="px-4 py-1.5 text-xs font-semibold rounded-lg bg-cyan-500/20 border border-cyan-500/40 text-cyan-300 hover:bg-cyan-500/30 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
                    >
                        {isLoading ? 'Working…' : (isNew ? 'Generate' : 'Update')}
                    </button>
                </div>
            </div>
        </div>
    );
}

/* ─── Thoughts / processing panel ────────────────────────────────────── */

function ThoughtsPanel({ thoughts, retryInfo, stage, elapsed }) {
    const endRef = useRef(null);
    useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [thoughts]);

    return (
        <div className="absolute inset-y-0 right-0 w-[42%] z-20 flex flex-col border-l border-white/[0.07] bg-[rgba(0,0,0,0.6)] backdrop-blur-md">
            {/* Panel header */}
            <div className="shrink-0 px-3 py-2.5 border-b border-white/[0.07] space-y-2">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                        <span className="text-[10px] font-bold tracking-[0.2em] text-emerald-400/80">
                            DESIGNER
                        </span>
                    </div>
                    {retryInfo?.attempt > 1 && (
                        <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
                            retryInfo.error
                                ? 'bg-amber-500/15 text-amber-400 border border-amber-500/30'
                                : 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/25'
                        }`}>
                            {retryInfo.attempt}/{retryInfo.maxAttempts ?? 3}
                        </span>
                    )}
                </div>
                <ProgressBar idx={stage.idx} label={stage.label} elapsed={elapsed} />
            </div>

            {/* Error banner */}
            {retryInfo?.error && (
                <div className="shrink-0 mx-3 mt-2 flex items-start gap-2 rounded-lg border border-red-500/25 bg-red-500/10 px-3 py-2">
                    <AlertTriangle size={12} className="text-red-400 mt-0.5 shrink-0" />
                    <p className="text-[11px] text-red-400/90 leading-relaxed">{retryInfo.error}</p>
                </div>
            )}

            {/* Scrolling thoughts */}
            <div className="flex-1 min-h-0 overflow-y-auto px-3 py-2">
                <pre className="text-[11px] font-mono text-emerald-400/70 whitespace-pre-wrap leading-relaxed">
                    {thoughts}
                </pre>
                <div ref={endRef} />
            </div>
        </div>
    );
}

/* ─── Toolbar ────────────────────────────────────────────────────────── */

function Toolbar({ hasModel, onIterate, onDownload }) {
    return (
        <div className="absolute top-2.5 left-3 z-10 flex items-center gap-1.5">
            <button
                onClick={onIterate}
                title="Refine design"
                className="flex items-center gap-1.5 rounded-lg border border-white/[0.12] bg-black/40 backdrop-blur-sm px-2.5 py-1.5 text-[11px] font-semibold tracking-wide text-white/60 hover:text-cyan-300 hover:border-cyan-500/40 hover:bg-cyan-500/10 transition-all"
            >
                <RefreshCw size={11} />
                ITERATE
            </button>

            {hasModel && (
                <button
                    onClick={onDownload}
                    title="Download STL"
                    className="flex items-center gap-1.5 rounded-lg border border-white/[0.12] bg-black/40 backdrop-blur-sm px-2.5 py-1.5 text-[11px] font-semibold tracking-wide text-white/60 hover:text-emerald-300 hover:border-emerald-500/40 hover:bg-emerald-500/10 transition-all"
                >
                    <Download size={11} />
                    DOWNLOAD
                </button>
            )}
        </div>
    );
}

/* ─── Main component ─────────────────────────────────────────────────── */

const CadWindow = ({ data, thoughts, retryInfo = {}, onClose, socket }) => {
    const [isIterating, setIsIterating] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [processingStartedAt, setProcessingStartedAt] = useState(null);
    const [elapsed, setElapsed] = useState(0);

    /* Track loading state */
    useEffect(() => {
        const loading = data?.format === 'loading';
        if (loading && !isProcessing) {
            setIsProcessing(true);
            setProcessingStartedAt(Date.now());
        }
        if (data?.format === 'stl' && isProcessing) {
            setIsProcessing(false);
            setProcessingStartedAt(null);
            setElapsed(0);
        }
    }, [data, isProcessing]);

    /* Elapsed timer */
    useEffect(() => {
        if (!isProcessing || !processingStartedAt) return;
        const id = setInterval(() => setElapsed(Math.floor((Date.now() - processingStartedAt) / 1000)), 250);
        return () => clearInterval(id);
    }, [isProcessing, processingStartedAt]);

    /* Parse STL geometry */
    const geometry = useMemo(() => {
        if (data?.format !== 'stl' || !data?.data) return null;
        try {
            const bytes = Uint8Array.from(atob(data.data), (c) => c.charCodeAt(0));
            const loader = new STLLoader();
            const geom = loader.parse(bytes.buffer);
            geom.rotateX(-Math.PI / 2);
            geom.center();
            return geom;
        } catch (e) {
            console.error('[CadWindow] STL parse failed:', e);
            return null;
        }
    }, [data]);

    const stage = useMemo(() => inferStage(thoughts), [thoughts]);
    const isLoading = isProcessing || data?.format === 'loading';
    const hasModel = data?.format === 'stl' && Boolean(data?.data);
    const showPrompt = isIterating || (!isLoading && !hasModel);

    /* Emit generate / iterate */
    const handleGenerate = (text) => {
        setIsProcessing(true);
        setProcessingStartedAt(Date.now());
        socket?.emit('generate_cad', { prompt: text });
        setIsIterating(false);
    };

    const handleIterate = (text) => {
        setIsProcessing(true);
        setProcessingStartedAt(Date.now());
        socket?.emit('iterate_cad', { prompt: text });
        setIsIterating(false);
    };

    /* Download STL */
    const handleDownload = () => {
        if (!data?.data) return;
        const bytes = Uint8Array.from(atob(data.data), (c) => c.charCodeAt(0));
        const blob = new Blob([bytes], { type: 'model/stl' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'model.stl';
        a.click();
        URL.revokeObjectURL(url);
    };

    return (
        <div className="w-full h-full relative flex overflow-hidden bg-[#080e14]">

            {/* ── Toolbar ── */}
            {!showPrompt && (
                <Toolbar
                    hasModel={hasModel}
                    onIterate={() => setIsIterating(true)}
                    onDownload={handleDownload}
                />
            )}

            {/* ── Reset button (bottom-left) ── */}
            {hasModel && !showPrompt && (
                <button
                    onClick={() => setIsIterating(true)}
                    title="New design"
                    className="absolute bottom-3 left-3 z-10 flex items-center gap-1.5 rounded-lg border border-white/[0.08] bg-black/40 backdrop-blur-sm px-2 py-1.5 text-[10px] tracking-widest text-white/30 hover:text-white/60 hover:border-white/20 transition-all"
                >
                    <RotateCcw size={10} />
                    NEW
                </button>
            )}

            {/* ── Status badge ── */}
            <div className="absolute bottom-3 right-3 z-10 text-[10px] font-mono tracking-widest text-cyan-500/30 pointer-events-none">
                CAD · {data?.format?.toUpperCase() ?? 'READY'}
            </div>

            {/* ── Prompt overlay ── */}
            {showPrompt && (
                <PromptOverlay
                    isNew={!data || data.format === 'loading' ? !hasModel : false}
                    onSubmit={hasModel ? handleIterate : handleGenerate}
                    onCancel={() => setIsIterating(false)}
                    isLoading={isLoading}
                />
            )}

            {/* ── 3-D viewport ── */}
            <div className={`flex-1 min-w-0 ${isLoading ? 'w-[58%]' : 'w-full'}`}>
                <Canvas shadows camera={{ position: [40, 30, 40], fov: 42 }}>
                    <color attach="background" args={['#080e14']} />
                    <Stage environment="city" intensity={0.45}>
                        {isLoading ? <LoadingCube /> : (geometry && <Center><GeometryModel geometry={geometry} /></Center>)}
                    </Stage>
                    <OrbitControls autoRotate={!isIterating && !isLoading} autoRotateSpeed={0.8} makeDefault />
                </Canvas>
            </div>

            {/* ── Thoughts panel ── */}
            {isLoading && (
                <ThoughtsPanel
                    thoughts={thoughts}
                    retryInfo={retryInfo}
                    stage={stage}
                    elapsed={elapsed}
                />
            )}
        </div>
    );
};

export default CadWindow;
