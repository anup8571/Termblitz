import { useState } from 'react';
import {
  Terminal,
  ShieldCheck,
  Smartphone,
  Monitor,
  GitBranch,
  Github,
  Globe,
  Sliders,
  Copy,
  Check,
  Layers,
  Touchpad,
  Zap,
} from 'lucide-react';

export default function App() {
  const [copiedCmd, setCopiedCmd] = useState<string | null>(null);
  const [adBlockEnabled, setAdBlockEnabled] = useState(true);
  const [lastGesture, setLastGesture] = useState<string>('None (Touch or drag below)');
  const [touchStart, setTouchStart] = useState<{ x: number; y: number } | null>(null);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCmd(text);
    setTimeout(() => setCopiedCmd(null), 2000);
  };

  const handleTouchStart = (e: React.TouchEvent | React.MouseEvent) => {
    const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
    const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
    setTouchStart({ x: clientX, y: clientY });
  };

  const handleTouchEnd = (e: React.TouchEvent | React.MouseEvent) => {
    if (!touchStart) return;
    const clientX = 'changedTouches' in e ? e.changedTouches[0].clientX : e.clientX;
    const clientY = 'changedTouches' in e ? e.changedTouches[0].clientY : e.clientY;

    const diffX = clientX - touchStart.x;
    const diffY = clientY - touchStart.y;
    const threshold = 30;

    if (Math.abs(diffX) > Math.abs(diffY)) {
      if (diffX > threshold) setLastGesture('👉 Swipe Right (Next Tab / Forward)');
      else if (diffX < -threshold) setLastGesture('👈 Swipe Left (Back / History)');
      else setLastGesture('👆 Tap / Click');
    } else {
      if (diffY > threshold) setLastGesture('👇 Swipe Down (Refresh / Page Scroll Down)');
      else if (diffY < -threshold) setLastGesture('☝️ Swipe Up (Address Bar / Toggle Menu)');
      else setLastGesture('👆 Tap / Click');
    }
    setTouchStart(null);
  };

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 font-sans selection:bg-emerald-500/30 selection:text-emerald-300">
      {/* Top Navigation Bar */}
      <header className="border-b border-neutral-800/80 bg-neutral-900/60 backdrop-blur-md sticky top-0 z-50 px-4 lg:px-8 py-3.5 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="h-9 w-9 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
            <Terminal className="h-5 w-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-bold text-base tracking-tight text-white">TermBrowser</span>
              <span className="px-2 py-0.5 text-xs font-mono bg-emerald-950/80 border border-emerald-800/50 text-emerald-400 rounded-full">
                v0.1.0-alpha
              </span>
            </div>
            <p className="text-xs text-neutral-400 font-mono">Visual Terminal Web Engine</p>
          </div>
        </div>

        <div className="flex items-center space-x-3 text-xs">
          <div className="flex items-center space-x-1.5 px-3 py-1.5 rounded-md bg-neutral-800/50 border border-neutral-700/60 text-neutral-300">
            <div className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="font-mono">Ready for GitHub Link</span>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="max-w-5xl mx-auto px-4 lg:px-8 py-10 space-y-10">
        {/* Hero Section */}
        <div className="relative overflow-hidden rounded-2xl border border-neutral-800 bg-gradient-to-b from-neutral-900/90 to-neutral-950/90 p-8 shadow-2xl">
          <div className="absolute -top-24 -right-24 h-64 w-64 rounded-full bg-emerald-500/10 blur-3xl pointer-events-none" />
          <div className="space-y-4 max-w-2xl relative">
            <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-mono">
              <Zap className="h-3.5 w-3.5" />
              <span>Hello World Baseline Active</span>
            </div>
            <h1 className="text-3xl lg:text-4xl font-extrabold tracking-tight text-white leading-tight">
              Hello, World! <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 via-teal-300 to-cyan-400">
                TermBrowser Workspace Initialized
              </span>
            </h1>
            <p className="text-neutral-300 text-sm lg:text-base leading-relaxed">
              Your clean foundation is ready. You can now connect this repository to GitHub, commit your code,
              and integrate your existing prototype files (visual rendering engine, touch/gesture handlers, and ad-blocking shield).
            </p>
          </div>
        </div>

        {/* Feature Grid & Prototype Targets */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {/* Gesture Control Card */}
          <div className="rounded-xl border border-neutral-800/90 bg-neutral-900/50 p-5 space-y-3 hover:border-neutral-700 transition">
            <div className="h-9 w-9 rounded-lg bg-teal-500/10 border border-teal-500/30 flex items-center justify-center text-teal-400">
              <Touchpad className="h-5 w-5" />
            </div>
            <h2 className="text-base font-semibold text-white">Gesture Navigation</h2>
            <p className="text-xs text-neutral-400 leading-relaxed">
              Touch, swipe, and motion controls mapped for Android Termux touchscreens and trackpad gestures across desktop terminals.
            </p>
            <div className="pt-2 flex flex-wrap gap-1.5 text-[11px] font-mono text-neutral-400">
              <span className="bg-neutral-800 px-2 py-0.5 rounded">Swipe back</span>
              <span className="bg-neutral-800 px-2 py-0.5 rounded">Tab switch</span>
              <span className="bg-neutral-800 px-2 py-0.5 rounded">Zoom</span>
            </div>
          </div>

          {/* Ad Blocker Card */}
          <div className="rounded-xl border border-neutral-800/90 bg-neutral-900/50 p-5 space-y-3 hover:border-neutral-700 transition">
            <div className="h-9 w-9 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
              <ShieldCheck className="h-5 w-5" />
            </div>
            <h2 className="text-base font-semibold text-white">Brave-Style Ad Blocker</h2>
            <p className="text-xs text-neutral-400 leading-relaxed">
              Cosmetic filtering and network request blocker targeting tracking scripts, banner ads, and crypto-miners at the terminal/web level.
            </p>
            <div className="pt-2 flex items-center justify-between text-xs">
              <span className="text-neutral-400 font-mono">Shield Protection:</span>
              <button
                onClick={() => setAdBlockEnabled(!adBlockEnabled)}
                className={`px-2.5 py-1 rounded text-[11px] font-mono font-medium transition cursor-pointer ${
                  adBlockEnabled
                    ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                    : 'bg-neutral-800 text-neutral-400 border border-neutral-700'
                }`}
              >
                {adBlockEnabled ? 'Active (Blocking)' : 'Disabled'}
              </button>
            </div>
          </div>

          {/* Cross-Terminal Support */}
          <div className="rounded-xl border border-neutral-800/90 bg-neutral-900/50 p-5 space-y-3 hover:border-neutral-700 transition">
            <div className="h-9 w-9 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
              <Monitor className="h-5 w-5" />
            </div>
            <h2 className="text-base font-semibold text-white">Universal Terminals</h2>
            <p className="text-xs text-neutral-400 leading-relaxed">
              Engineered to run seamlessly across Android Termux, Linux, macOS, and Windows PowerShell/WSL environments.
            </p>
            <div className="pt-2 flex flex-wrap gap-1.5 text-[11px] font-mono text-neutral-400">
              <span className="bg-neutral-800 px-2 py-0.5 rounded">Termux</span>
              <span className="bg-neutral-800 px-2 py-0.5 rounded">Linux</span>
              <span className="bg-neutral-800 px-2 py-0.5 rounded">macOS</span>
              <span className="bg-neutral-800 px-2 py-0.5 rounded">Windows</span>
            </div>
          </div>
        </div>

        {/* Interactive Gesture Test Pad */}
        <div className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Touchpad className="h-4 w-4 text-emerald-400" />
              <h2 className="text-sm font-semibold text-white uppercase tracking-wider font-mono">
                Gesture Test Sandbox
              </h2>
            </div>
            <span className="text-xs font-mono text-neutral-400">Try swiping or dragging</span>
          </div>

          <div
            onMouseDown={handleTouchStart}
            onMouseUp={handleTouchEnd}
            onTouchStart={handleTouchStart}
            onTouchEnd={handleTouchEnd}
            className="h-32 w-full rounded-lg border-2 border-dashed border-neutral-700 hover:border-emerald-500/50 bg-neutral-950/60 flex flex-col items-center justify-center cursor-grab active:cursor-grabbing select-none transition"
          >
            <p className="text-xs text-neutral-400 font-mono mb-1">Interactive touch & swipe surface</p>
            <p className="text-sm font-semibold text-emerald-400 font-mono">{lastGesture}</p>
          </div>
        </div>

        {/* GitHub & Prototype Integration Guide */}
        <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-6 space-y-5">
          <div className="flex items-center justify-between border-b border-neutral-800 pb-4">
            <div className="flex items-center space-x-2.5">
              <Github className="h-5 w-5 text-neutral-200" />
              <h2 className="text-base font-bold text-white">How to Connect to GitHub & Push Your Prototype</h2>
            </div>
            <div className="flex items-center space-x-1.5 text-xs text-neutral-400 font-mono">
              <GitBranch className="h-3.5 w-3.5" />
              <span>main</span>
            </div>
          </div>

          <div className="space-y-4 text-xs font-mono">
            {/* Step 1 */}
            <div className="space-y-1.5">
              <div className="text-neutral-400 flex items-center justify-between">
                <span>1. Add your remote GitHub repository:</span>
                <button
                  onClick={() => copyToClipboard('git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git')}
                  className="flex items-center space-x-1 text-emerald-400 hover:text-emerald-300 transition"
                >
                  {copiedCmd?.includes('git remote add origin') ? (
                    <>
                      <Check className="h-3 w-3" />
                      <span>Copied!</span>
                    </>
                  ) : (
                    <>
                      <Copy className="h-3 w-3" />
                      <span>Copy</span>
                    </>
                  )}
                </button>
              </div>
              <div className="p-3 bg-neutral-950 rounded-lg border border-neutral-800 text-neutral-300 overflow-x-auto">
                git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
              </div>
            </div>

            {/* Step 2 */}
            <div className="space-y-1.5">
              <div className="text-neutral-400 flex items-center justify-between">
                <span>2. Stage and push this Hello World baseline:</span>
                <button
                  onClick={() => copyToClipboard('git add . && git commit -m "feat: hello world termbrowser" && git push -u origin main')}
                  className="flex items-center space-x-1 text-emerald-400 hover:text-emerald-300 transition"
                >
                  {copiedCmd?.includes('git push') ? (
                    <>
                      <Check className="h-3 w-3" />
                      <span>Copied!</span>
                    </>
                  ) : (
                    <>
                      <Copy className="h-3 w-3" />
                      <span>Copy</span>
                    </>
                  )}
                </button>
              </div>
              <div className="p-3 bg-neutral-950 rounded-lg border border-neutral-800 text-neutral-300 overflow-x-auto">
                git add . && git commit -m "feat: hello world termbrowser baseline" && git push -u origin main
              </div>
            </div>

            {/* Step 3 */}
            <div className="p-3.5 rounded-lg bg-emerald-950/30 border border-emerald-800/40 text-emerald-200/90 text-[12px] font-sans leading-relaxed">
              <span className="font-semibold text-emerald-300">Next Step: </span>
              Once you've connected GitHub or paste in your prototype code/files, let me know which files you have added (e.g., rendering engine, ad-block filter lists, Termux gesture listeners, or TUI modules). I will review, refine, and build out the full browser features for you!
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="pt-4 border-t border-neutral-800/60 flex flex-col sm:flex-row items-center justify-between text-xs text-neutral-400 space-y-2 sm:space-y-0">
          <div className="flex items-center space-x-2">
            <Smartphone className="h-3.5 w-3.5 text-neutral-400" />
            <span>Android Termux</span>
            <span className="text-neutral-600">•</span>
            <span>Linux</span>
            <span className="text-neutral-600">•</span>
            <span>macOS</span>
            <span className="text-neutral-600">•</span>
            <span>Windows</span>
          </div>
          <div>TermBrowser Engine Baseline</div>
        </footer>
      </main>
    </div>
  );
}
