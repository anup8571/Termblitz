import React, { useState } from 'react';
import {
  Terminal,
  Shield,
  Zap,
  Layers,
  Sparkles,
  Smartphone,
  Cpu,
  CheckCircle2,
  Copy,
  Check,
  Code2,
  Flame,
  ArrowRight,
  Sliders,
  Grid
} from 'lucide-react';

type UpscaleOption = {
  id: 'off' | 'fsr' | 'neural' | 'combined' | 'multipass';
  label: string;
  badge: string;
  shortBadge: string;
  pipelineDesc: string;
  latencyMs: string;
  psnrGain: string;
  passes: string[];
};

const UPSCALE_OPTIONS: UpscaleOption[] = [
  {
    id: 'off',
    label: 'OFF',
    badge: 'OFF',
    shortBadge: 'OFF',
    pipelineDesc: 'Standard center-weighted 3x3 subpixel box downsampler only',
    latencyMs: '0.2 ms',
    psnrGain: 'Baseline',
    passes: ['Subpixel Box (3x3)']
  },
  {
    id: 'fsr',
    label: 'FSR 1.0 (EASU+RCAS)',
    badge: 'FSR',
    shortBadge: 'FSR',
    pipelineDesc: '12-tap elliptical Lanczos2 Edge-Adaptive Spatial Upsampling + 5-tap RCAS contrast sharpening',
    latencyMs: '1.4 ms',
    psnrGain: '+2.4 dB',
    passes: ['12-Tap EASU', '5-Tap RCAS (0.85)']
  },
  {
    id: 'neural',
    label: 'NEURAL (FSRCNN-ESPCN)',
    badge: 'NEURAL',
    shortBadge: 'NRL',
    pipelineDesc: 'BT.601 YCbCr split -> 4-layer Hybrid FSRCNN-ESPCN 2x PixelShuffle on Y + Bilinear Cb/Cr',
    latencyMs: '2.1 ms',
    psnrGain: '+3.6 dB',
    passes: ['YCbCr Split', 'Conv3x3(1→8)+PReLU', 'Conv1x1(8→4)', 'ResConv3x3(4→4)', 'ESPCN PixelShuffle 2x']
  },
  {
    id: 'combined',
    label: 'COMBINED (NEURAL+RCAS)',
    badge: 'COMBINED',
    shortBadge: 'CMB',
    pipelineDesc: 'Hybrid FSRCNN-ESPCN 2x sub-pixel neural reconstruction followed by RCAS adaptive sharpening',
    latencyMs: '2.6 ms',
    psnrGain: '+4.2 dB',
    passes: ['YCbCr Split', 'Hybrid FSRCNN-ESPCN 2x', '5-Tap RCAS (0.80)']
  },
  {
    id: 'multipass',
    label: 'MULTI-PASS',
    badge: 'MULTIPASS',
    shortBadge: 'MLP',
    pipelineDesc: '3-stage cascade: EASU directional pre-scale -> FSRCNN-ESPCN 2x detail synthesis -> RCAS lobe reinforcement',
    latencyMs: '3.5 ms',
    psnrGain: '+4.8 dB',
    passes: ['Stage 1: EASU 12-Tap', 'Stage 2: FSRCNN-ESPCN 2x', 'Stage 3: RCAS (0.85)']
  }
];

export default function App() {
  const [activeTab, setActiveTab] = useState<'overview' | 'architecture' | 'controls' | 'install'>('overview');
  const [copiedCmd, setCopiedCmd] = useState<string | null>(null);
  const [simulatedMode, setSimulatedMode] = useState<'hybrid' | 'raw' | 'native'>('hybrid');
  const [adblockEnabled, setAdblockEnabled] = useState<boolean>(true);
  const [upscaleIdx, setUpscaleIdx] = useState<number>(3); // Default highlight on COMBINED or FSR; user can cycle all 5
  const [ditherEnabled, setDitherEnabled] = useState<boolean>(true);

  const currentUpscale = UPSCALE_OPTIONS[upscaleIdx];

  const cycleUpscaleMode = () => {
    setUpscaleIdx((prev) => (prev + 1) % UPSCALE_OPTIONS.length);
  };

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCmd(id);
    setTimeout(() => setCopiedCmd(null), 2000);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Banner / Navigation */}
      <header className="border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center font-bold text-black shadow-lg shadow-orange-500/20">
              <Zap className="w-5 h-5 fill-current" />
            </div>
            <div>
              <span className="font-mono font-bold tracking-tight text-lg text-white">TERMBLITZ</span>
              <span className="ml-2 text-xs px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 font-mono border border-emerald-500/20">
                v0.3.0-neural-sr
              </span>
            </div>
          </div>

          <nav className="flex items-center gap-1 sm:gap-2">
            {[
              { id: 'overview', label: 'Overview', icon: Sparkles },
              { id: 'architecture', label: 'Architecture', icon: Layers },
              { id: 'controls', label: 'Controls & Gestures', icon: Smartphone },
              { id: 'install', label: 'Termux Install', icon: Terminal },
            ].map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-orange-500 text-white shadow-sm shadow-orange-500/30'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span className="hidden sm:inline">{tab.label}</span>
                </button>
              );
            })}
          </nav>
        </div>
      </header>

      {/* Interactive Super-Resolution & Dithering Companion Toolbar */}
      <div className="bg-slate-900/90 border-b border-slate-800/90">
        <div className="max-w-6xl mx-auto px-4 py-3 flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-mono uppercase tracking-wider text-slate-400 flex items-center gap-1.5 mr-1">
              <Cpu className="w-3.5 h-3.5 text-orange-400" />
              <span>Upscale &amp; Neural SR Mode:</span>
            </span>

            {/* Cycle Button */}
            <button
              onClick={cycleUpscaleMode}
              title="Click to cycle Super-Resolution mode (Hotkey: u / Alt+u)"
              className="px-3 py-1.5 rounded-md bg-orange-500/15 hover:bg-orange-500/25 border border-orange-500/40 text-orange-300 font-mono text-xs font-bold flex items-center gap-2 transition-all cursor-pointer"
            >
              <Sliders className="w-3.5 h-3.5 text-orange-400" />
              <span>[u] {currentUpscale.label}</span>
            </button>

            {/* Direct Mode Selector Pills */}
            <div className="hidden lg:flex items-center bg-slate-950/90 p-0.5 rounded-md border border-slate-800">
              {UPSCALE_OPTIONS.map((opt, idx) => (
                <button
                  key={opt.id}
                  onClick={() => setUpscaleIdx(idx)}
                  className={`px-2 py-1 rounded text-[11px] font-mono transition-all cursor-pointer ${
                    upscaleIdx === idx
                      ? 'bg-orange-500 text-black font-bold'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>

            {/* Dither Toggle Button */}
            <button
              onClick={() => setDitherEnabled((prev) => !prev)}
              title="Toggle 4x4 Ordered Bayer & Serpentine Floyd-Steinberg Dithering (Hotkey: d / Alt+d)"
              className={`px-3 py-1.5 rounded-md font-mono text-xs font-bold flex items-center gap-1.5 border transition-all cursor-pointer ${
                ditherEnabled
                  ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-300 hover:bg-emerald-500/25'
                  : 'bg-slate-800/70 border-slate-700 text-slate-400 hover:text-slate-200'
              }`}
            >
              <Grid className="w-3.5 h-3.5" />
              <span>[d] DITHER: {ditherEnabled ? 'ON' : 'OFF'}</span>
            </button>
          </div>

          {/* Live Pipeline Telemetry Readout */}
          <div className="flex items-center gap-3 text-xs font-mono text-slate-400">
            <span className="px-2 py-0.5 rounded bg-slate-950 border border-slate-800 text-amber-300">
              [SR: {currentUpscale.badge} | DITHER: {ditherEnabled ? 'ON' : 'OFF'}]
            </span>
            <span className="hidden sm:inline text-slate-500">•</span>
            <span className="hidden sm:inline text-emerald-400 font-semibold">{currentUpscale.psnrGain}</span>
            <span className="hidden sm:inline text-slate-500">•</span>
            <span className="hidden sm:inline text-slate-300">{currentUpscale.latencyMs}/frame</span>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <main className="flex-1 max-w-6xl mx-auto w-full px-4 py-8">
        {/* Hero Section */}
        <section className="mb-10 text-center sm:text-left lg:flex lg:items-center lg:justify-between gap-8 pb-10 border-b border-slate-800">
          <div className="max-w-2xl">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-orange-500/10 border border-orange-500/20 text-orange-400 text-xs font-mono mb-4">
              <Flame className="w-3.5 h-3.5 fill-current" />
              <span>Pure-Rust Neural Super-Resolution + AMD FSR 1.0 + Bayer/Sixel Dithering</span>
            </div>
            <h1 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-white mb-4 leading-tight">
              Visual Web Browsing, <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-orange-400 via-amber-300 to-yellow-400">
                Inside Your Terminal.
              </span>
            </h1>
            <p className="text-slate-400 text-base sm:text-lg leading-relaxed mb-6">
              Termblitz fuses a lightweight pure-Rust HTML parser (&lt;1MB RAM) with a Headless Chromium
              CDP hybrid rendering engine, a zero-dependency YCbCr Hybrid FSRCNN-ESPCN + AMD FSR 1.0 (EASU+RCAS) super-resolution pipeline, and 4x4 Ordered Bayer / Serpentine Floyd-Steinberg dithering.
            </p>
            <div className="flex flex-wrap items-center gap-3">
              <button
                onClick={() => setActiveTab('install')}
                className="px-5 py-2.5 rounded-lg bg-orange-500 hover:bg-orange-600 text-white font-medium text-sm flex items-center gap-2 shadow-lg shadow-orange-500/25 transition-all cursor-pointer"
              >
                <Terminal className="w-4 h-4" />
                <span>Build in Termux</span>
                <ArrowRight className="w-4 h-4" />
              </button>
              <button
                onClick={() => setActiveTab('architecture')}
                className="px-5 py-2.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-200 border border-slate-700 font-medium text-sm flex items-center gap-2 transition-all cursor-pointer"
              >
                <Code2 className="w-4 h-4" />
                <span>View Neural SR &amp; Rust Spec</span>
              </button>
            </div>
          </div>

          {/* Interactive Terminal Simulator Box */}
          <div className="mt-8 lg:mt-0 w-full lg:w-[430px] bg-black rounded-xl border border-slate-800 p-3 shadow-2xl shadow-orange-500/5">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-2 mb-2 px-1">
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-rose-500/80"></span>
                <span className="w-2.5 h-2.5 rounded-full bg-amber-500/80"></span>
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-500/80"></span>
              </div>
              <span className="text-[11px] font-mono text-slate-400">
                termblitz --upscale={currentUpscale.id} {!ditherEnabled ? '--no-dither' : ''}
              </span>
            </div>

            {/* Simulated Status Bar (Row 0) */}
            <div className="bg-slate-900 rounded p-1.5 font-mono text-[11px] flex items-center justify-between text-slate-200 mb-2 border border-slate-800">
              <button
                onClick={() => {
                  if (simulatedMode === 'hybrid') setSimulatedMode('raw');
                  else if (simulatedMode === 'raw') setSimulatedMode('native');
                  else setSimulatedMode('hybrid');
                }}
                className={`px-1.5 py-0.5 rounded text-[10px] font-bold uppercase transition-all cursor-pointer ${
                  simulatedMode === 'hybrid'
                    ? 'bg-amber-400 text-black'
                    : simulatedMode === 'raw'
                    ? 'bg-purple-600 text-white'
                    : 'bg-emerald-500 text-black'
                }`}
              >
                {simulatedMode} 100%
              </button>

              <span className="text-slate-300 text-[10px] truncate mx-1.5">google.com</span>

              <div className="flex items-center gap-1 text-[10px] text-slate-300 shrink-0">
                <button
                  onClick={cycleUpscaleMode}
                  className="px-1 py-0.5 rounded bg-slate-950 border border-slate-700 text-orange-300 hover:border-orange-500/50 cursor-pointer"
                  title="Click to cycle UpscaleMode"
                >
                  [SR: {currentUpscale.badge} | DITHER: {ditherEnabled ? 'ON' : 'OFF'}]
                </button>
                <button
                  onClick={() => setAdblockEnabled(!adblockEnabled)}
                  className={`px-1 rounded cursor-pointer ${adblockEnabled ? 'text-amber-400 font-bold' : 'text-slate-600'}`}
                >
                  {adblockEnabled ? 'SHIELD' : 'NO-AD'}
                </button>
              </div>
            </div>

            {/* Simulated Viewport Output */}
            <div
              className={`h-60 rounded border border-slate-900 p-2.5 font-mono text-xs flex flex-col justify-between overflow-hidden transition-all ${
                ditherEnabled
                  ? 'bg-gradient-to-b from-slate-950 via-slate-900/95 to-slate-950'
                  : 'bg-slate-950'
              }`}
            >
              <div className="space-y-2 text-center pt-1">
                {/* Simulated Google Logo with SR edge clarity indicator */}
                <div
                  className={`text-base font-bold tracking-wider transition-all ${
                    currentUpscale.id === 'off'
                      ? 'opacity-75 blur-[0.3px]'
                      : currentUpscale.id === 'multipass' || currentUpscale.id === 'combined'
                      ? 'drop-shadow-[0_0_6px_rgba(251,146,60,0.25)] scale-105'
                      : ''
                  }`}
                >
                  <span className="text-blue-400">G</span>
                  <span className="text-red-400">o</span>
                  <span className="text-yellow-400">o</span>
                  <span className="text-blue-400">g</span>
                  <span className="text-green-400">l</span>
                  <span className="text-red-400">e</span>
                </div>

                <div
                  className={`mx-auto w-11/12 bg-slate-900 rounded-full py-1 px-3 text-[11px] text-slate-300 flex items-center justify-between ${
                    currentUpscale.id !== 'off'
                      ? 'border border-slate-600 shadow-inner'
                      : 'border border-slate-800/80'
                  }`}
                >
                  <span>Ask Google or type URL</span>
                  <span className="text-orange-400 font-bold">↵</span>
                </div>

                <div className="text-[10px] text-slate-400 pt-0.5">
                  Google offered in:{' '}
                  <span className="text-blue-400 hover:underline cursor-pointer">हिन्दी</span>{' '}
                  <span className="text-blue-400 hover:underline cursor-pointer">বাংলা</span>{' '}
                  <span className="text-blue-400 hover:underline cursor-pointer">తెలుగు</span>{' '}
                  <span className="text-blue-400 hover:underline cursor-pointer">मराठी</span>
                </div>

                {/* Active SR Pipeline Stage Pills inside simulator */}
                <div className="pt-2 px-1">
                  <div className="text-[10px] text-slate-400 mb-1 flex items-center justify-between">
                    <span>Active Pipeline Stages:</span>
                    <span className="text-emerald-400">
                      {ditherEnabled ? '4x4 Bayer + Sixel FS' : 'Un-dithered Quant'}
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-1 justify-center">
                    {currentUpscale.passes.map((stage, i) => (
                      <span
                        key={i}
                        className="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-[10px] text-amber-300"
                      >
                        {stage}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              <div className="border-t border-slate-900 pt-1.5 text-[10px] text-slate-400 flex justify-between items-center px-1">
                <span>[m] Mode</span>
                <span>[u] SR: {currentUpscale.shortBadge}</span>
                <span>[d] Dither: {ditherEnabled ? 'ON' : 'OFF'}</span>
                <span>[o /] URL</span>
              </div>
            </div>
          </div>
        </section>

        {/* Feature Cards Grid */}
        <section className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <div className="p-5 rounded-xl bg-slate-900/70 border border-slate-800 hover:border-slate-700 transition-all">
            <div className="w-10 h-10 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center justify-center mb-3">
              <Cpu className="w-5 h-5" />
            </div>
            <h3 className="text-lg font-bold text-white mb-2">YCbCr Hybrid FSRCNN-ESPCN + FSR 1.0</h3>
            <p className="text-slate-400 text-sm leading-relaxed">
              100% pure-Rust super-resolution (`src/upscale.rs`) combining a 4-layer neural sub-pixel conv network on luminance (`Y`) with AMD FidelityFX EASU 12-tap + RCAS 5-tap sharpening.
            </p>
          </div>

          <div className="p-5 rounded-xl bg-slate-900/70 border border-slate-800 hover:border-slate-700 transition-all">
            <div className="w-10 h-10 rounded-lg bg-orange-500/10 text-orange-400 border border-orange-500/20 flex items-center justify-center mb-3">
              <Grid className="w-5 h-5" />
            </div>
            <h3 className="text-lg font-bold text-white mb-2">Bayer &amp; Serpentine Floyd-Steinberg</h3>
            <p className="text-slate-400 text-sm leading-relaxed">
              Eliminates TrueColor half-block (`▀`) banding via deterministic 4x4 Ordered Bayer dithering and sharpens 16-color Sixel graphics using bit-shifted serpentine error diffusion.
            </p>
          </div>

          <div className="p-5 rounded-xl bg-slate-900/70 border border-slate-800 hover:border-slate-700 transition-all">
            <div className="w-10 h-10 rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/20 flex items-center justify-center mb-3">
              <Smartphone className="w-5 h-5" />
            </div>
            <h3 className="text-lg font-bold text-white mb-2">Indic Scripts &amp; Brave Ad Shield</h3>
            <p className="text-slate-400 text-sm leading-relaxed">
              Full Unicode combining-mark clustering (`Cell.comb: [char; 3]`) for Devanagari, Bengali, Telugu, Tamil &amp; Malayalam, paired with network + cosmetic ad blocking.
            </p>
          </div>
        </section>

        {/* Tab 1: Overview & Highlights */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            <div className="p-6 rounded-xl bg-slate-900/50 border border-slate-800">
              <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-amber-400" />
                <span>Live Super-Resolution Pipeline Inspector ({currentUpscale.label})</span>
              </h2>
              <p className="text-sm text-slate-300 mb-4">{currentUpscale.pipelineDesc}</p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-slate-300">
                <div className="flex items-start gap-3">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
                  <div>
                    <strong className="text-white block mb-0.5">Fast YCbCr Split &amp; Bilinear Chroma</strong>
                    Runs heavy neural convolutions exclusively on the normalized luminance (`Y`) channel while bilinearly upscaling `Cb`/`Cr` for a 3x compute speedup.
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
                  <div>
                    <strong className="text-white block mb-0.5">AMD FSR 1.0 EASU + RCAS</strong>
                    12-tap elliptical Lanczos2 directional kernel with anti-ringing bounding-box clamping and 5-tap contrast-adaptive sharpening.
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
                  <div>
                    <strong className="text-white block mb-0.5">4x4 Ordered Bayer Half-Block Dithering</strong>
                    Applies a deterministic `[-3..+4]` integer offset per subpixel while keeping pure white/black backgrounds run-length batchable.
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
                  <div>
                    <strong className="text-white block mb-0.5">Serpentine Floyd-Steinberg Sixel Quantizer</strong>
                    Diffuses 16-color quantization error with bit-shift integer weights (`*7&gt;&gt;4`, `*3&gt;&gt;4`, `*5&gt;&gt;4`, `*1&gt;&gt;4`) clamped to `-32..=32`.
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Tab 2: Architecture */}
        {activeTab === 'architecture' && (
          <div className="space-y-6">
            <div className="p-6 rounded-xl bg-slate-900/50 border border-slate-800">
              <h2 className="text-xl font-bold text-white mb-3 flex items-center gap-2">
                <Layers className="w-5 h-5 text-orange-400" />
                <span>Rust Codebase Structure</span>
              </h2>
              <div className="font-mono text-xs bg-black/60 p-4 rounded-lg border border-slate-800 space-y-2 text-slate-300">
                <div>├── <strong className="text-orange-400">Cargo.toml</strong> : zero-bloat pure Rust dependencies (crossterm, image, libc)</div>
                <div>├── <strong className="text-orange-400">src/upscale.rs</strong> : YCbCr FSRCNN-ESPCN 2x neural network + AMD FSR 1.0 (EASU + RCAS) + UpscaleMode dispatcher</div>
                <div>├── <strong className="text-orange-400">src/renderer.rs</strong> : DiffRenderer, 4x4 Bayer half-block dithering, Sixel serpentine Floyd-Steinberg, Indic grapheme clustering</div>
                <div>├── <strong className="text-orange-400">src/engine.rs</strong> : Native HTML parser + Headless Chromium CDP worker &amp; ancestor background color resolver</div>
                <div>├── <strong className="text-orange-400">src/input.rs</strong> : Hotkeys (u/Alt+u SR cycle, d/Alt+d dither toggle, o/ omnibox) &amp; SGR 1006 touch gestures</div>
                <div>└── <strong className="text-orange-400">src/main.rs</strong> : CLI parser (--upscale=..., --no-dither), status bar [SR | DITHER] badge, frame loop</div>
              </div>
            </div>
          </div>
        )}

        {/* Tab 3: Controls */}
        {activeTab === 'controls' && (
          <div className="space-y-6">
            <div className="p-6 rounded-xl bg-slate-900/50 border border-slate-800">
              <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                <Smartphone className="w-5 h-5 text-blue-400" />
                <span>Controls &amp; Gesture Cheat Sheet</span>
              </h2>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm text-slate-300">
                  <thead className="text-xs uppercase bg-slate-800/60 text-slate-400 font-mono">
                    <tr>
                      <th className="py-2.5 px-4 rounded-l">Key / Gesture</th>
                      <th className="py-2.5 px-4">Action</th>
                      <th className="py-2.5 px-4 rounded-r">Description</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800 font-mono text-xs">
                    <tr>
                      <td className="py-2.5 px-4 text-orange-400 font-bold">u / Alt+u</td>
                      <td className="py-2.5 px-4 text-white">Cycle Super-Resolution</td>
                      <td className="py-2.5 px-4 text-slate-400">Cycles Off → FSR 1.0 → Neural 2x → Combined → Multi-Pass</td>
                    </tr>
                    <tr>
                      <td className="py-2.5 px-4 text-orange-400 font-bold">d / Alt+d</td>
                      <td className="py-2.5 px-4 text-white">Toggle Dithering</td>
                      <td className="py-2.5 px-4 text-slate-400">Toggles 4x4 Ordered Bayer (ANSI) &amp; Serpentine Floyd-Steinberg (Sixel)</td>
                    </tr>
                    <tr>
                      <td className="py-2.5 px-4 text-orange-400 font-bold">m / Tap Badge</td>
                      <td className="py-2.5 px-4 text-white">Mode Toggle</td>
                      <td className="py-2.5 px-4 text-slate-400">Cycles Native → Hybrid → Raw</td>
                    </tr>
                    <tr>
                      <td className="py-2.5 px-4 text-orange-400 font-bold">o / / / Tap Bar</td>
                      <td className="py-2.5 px-4 text-white">Focus Omnibox</td>
                      <td className="py-2.5 px-4 text-slate-400">Enter search query or URL</td>
                    </tr>
                    <tr>
                      <td className="py-2.5 px-4 text-orange-400 font-bold">a</td>
                      <td className="py-2.5 px-4 text-white">Toggle Adblock</td>
                      <td className="py-2.5 px-4 text-slate-400">Toggle Brave network &amp; cosmetic shield</td>
                    </tr>
                    <tr>
                      <td className="py-2.5 px-4 text-orange-400 font-bold">Swipe Right / [</td>
                      <td className="py-2.5 px-4 text-white">History Back</td>
                      <td className="py-2.5 px-4 text-slate-400">Navigate to previous page</td>
                    </tr>
                    <tr>
                      <td className="py-2.5 px-4 text-orange-400 font-bold">Swipe Left / ]</td>
                      <td className="py-2.5 px-4 text-white">History Forward</td>
                      <td className="py-2.5 px-4 text-slate-400">Navigate to next page</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Tab 4: Termux Install */}
        {activeTab === 'install' && (
          <div className="space-y-6">
            <div className="p-6 rounded-xl bg-slate-900/50 border border-slate-800">
              <h2 className="text-xl font-bold text-white mb-2 flex items-center gap-2">
                <Terminal className="w-5 h-5 text-emerald-400" />
                <span>Build &amp; Run in Android Termux</span>
              </h2>
              <p className="text-slate-400 text-sm mb-6">
                Termblitz compiles to a single standalone binary with zero external neural weight files.
              </p>

              <div className="space-y-4 font-mono text-xs">
                <div>
                  <div className="text-slate-400 text-[11px] mb-1.5 font-sans font-medium">1. Build the Optimized Release Binary:</div>
                  <div className="relative bg-black p-3.5 rounded-lg border border-slate-800 text-emerald-400">
                    <code>cargo build --release</code>
                    <button
                      onClick={() => copyToClipboard('cargo build --release', 'cmd1')}
                      className="absolute right-2.5 top-2.5 p-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 cursor-pointer"
                    >
                      {copiedCmd === 'cmd1' ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                    </button>
                  </div>
                </div>

                <div>
                  <div className="text-slate-400 text-[11px] mb-1.5 font-sans font-medium">2. Launch with Combined Neural + FSR Super-Resolution:</div>
                  <div className="relative bg-black p-3.5 rounded-lg border border-slate-800 text-emerald-400">
                    <code>./target/release/termblitz --upscale=combined https://www.google.com</code>
                    <button
                      onClick={() => copyToClipboard('./target/release/termblitz --upscale=combined https://www.google.com', 'cmd2')}
                      className="absolute right-2.5 top-2.5 p-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 cursor-pointer"
                    >
                      {copiedCmd === 'cmd2' ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 py-6 text-center text-xs text-slate-500 font-mono">
        Termblitz • Pure Rust Visual Web Browser for Android Termux • MIT License
      </footer>
    </div>
  );
}
