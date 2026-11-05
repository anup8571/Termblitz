# TermBlitz Runtime Stabilization: Zero-Loop Scroll, Native Audio, YouTube Ad-Shield & Tear-Free Video Fast-Path

This plan resolves four critical runtime bottlenecks in **TermBlitz** when browsing media-rich and long-scroll websites (such as YouTube and Wikipedia) while preserving 100% of existing Super-Resolution (`EASU`, `FSRCNN-ESPCN`, `RCAS`), Ordered Bayer / Serpentine Floyd-Steinberg dithering, and Indic script clustering features.

---

## User Review & Critical Decisions

> [!IMPORTANT]
> The following architectural choices were confirmed during clarification and govern the implementation:

- **Confirmed Decision 1 — Audio Mute vs. Render Mode Keybinding**:
  - `m` (and `F2`) remains dedicated to cycling Render Mode (`Native` $\rightarrow$ `Hybrid` $\rightarrow$ `Raw`) so existing muscle memory and diagnostic notices remain intact.
  - `Alt+m`, `M` (`Shift+m`), and tapping/clicking the `🔊` / `🔇` badge in the top status bar will toggle page audio mute (`AppAction::ToggleMute`).
- **Confirmed Decision 2 — Active Video Super-Resolution Fast-Path**:
  - Whenever an actively playing `<video>` element (`rect.width > 40 && rect.height > 40`) is detected in the viewport, the renderer switches the full frame to the fast single-pass terminal subpixel `fsr_rcas` sharpener (bypassing heavy multi-layer CPU convolutions) and suppresses DOM text overlays inside the `<video>` bounding box (`video_rects`) so video playback runs at maximum FPS without tearing.

---

## 1. Overview & Core Concept

- **What It Does**: Eliminates runaway/looping scroll inertia on touch and mouse-wheel input, enables real browser audio output (including automatic background PulseAudio startup on Android Termux), injects an active YouTube video/feed ad-blocker and auto-skipper, and eliminates horizontal "strips falling" video tearing via DEC Synchronized Output (`\x1b[?2026h` / `\x1b[?2026l`) and a `<video>` bounding-box fast-path.
- **Target Audience / Persona**: Terminal power users and Android Termux mobile users browsing modern web apps, Wikipedia, and YouTube directly inside a terminal emulator.
- **Key Value**: Stops runaway scroll loops the instant a user's finger lifts, plays YouTube audio out of the box, skips pre-roll/mid-roll video ads automatically, and renders tear-free video frames at interactive frame rates.

---

## 2. User Experience & Visual Design

- **Key User Flows**:
  1. **Zero-Latency Coalesced Scrolling**: Swiping on a touchscreen or spinning a mouse wheel drains all queued terminal events per tick, coalesces scroll deltas into a single bounded step (`-12..=12`), resets drag latching on `Up` / `ScrollUp` / `ScrollDown`, and executes immediate `window.scrollBy(0, delta)` in Chromium so scrolling halts the exact millisecond input stops.
  2. **Termux & Desktop Audio Playback + Mute Control**: Launching TermBlitz on Android Termux automatically checks for and starts `pulseaudio --start --exit-idle-time=-1` with `PULSE_SERVER=127.0.0.1` when available, while Chromium spawns with `--autoplay-policy=no-user-gesture-required` and `AudioServiceOutOfProcess`. Users see `🔊` or `🔇` in the top status bar and can press `Alt+m`, `M`, or tap the icon to mute/unmute all active `<video>` and `<audio>` elements.
  3. **Automatic YouTube Ad-Shield & Skipper**: When Brave Shield (`SHIELD`) is active, YouTube feed ad slots are hidden via CSS + `MutationObserver`, and any video ad (`.ad-showing`, `.ad-interrupting`) is immediately muted, fast-forwarded (`currentTime = duration || 9999`), and skipped via programmatic button click.
  4. **Tear-Free Synchronized Video Playback**: Every terminal frame flush is wrapped in DEC Mode 2026 (`\x1b[?2026h` ... `\x1b[?2026l`) in a single atomic `write_stdout_bytes` call. When a `<video>` is playing, DOM text overlays inside the video rectangle are suppressed so stale captions or links never slice across the video canvas.

---

## 3. Key Product Decisions & Trade-Offs

- **Decision 1 — Synchronous `window.scrollBy` + Worker Channel Coalescing vs. `Input.dispatchMouseEvent(mouseWheel)`**:
  - *Chosen Approach*: Replace `Input.dispatchMouseEvent` (`mouseWheel`) with synchronous `window.scrollBy({ top: delta_px, left: 0, behavior: 'instant' })` and coalesce consecutive `CdpWorkerCommand::Scroll` messages in the worker loop.
  - *Why*: Chromium's compositor treats `mouseWheel` events as asynchronous smooth-scroll animations that queue up across dozens of frames and trigger inertial runaway loops on heavy pages like YouTube. Immediate `scrollBy` with channel coalescing guarantees 1:1 deterministic scroll positioning.
- **Decision 2 — Atomic DEC Synchronized Output (`?2026h` / `?2026l`)**:
  - *Chosen Approach*: Prepend `\x1b[?2026h` and append `\x1b[?2026l` to `DiffRenderer::write_buf` (as well as Kitty and Sixel frame buffers) before invoking `write_stdout_bytes`.
  - *Why*: Terminal emulators (Termux, Kitty, WezTerm, Ghostty, Foot, iTerm2) buffer all cursor moves and RGB cell updates between `?2026h` and `?2026l` and paint the completed frame at vblank, eliminating the "horizontal strips falling" artifact.

---

## 4. Technical Architecture & Data Strategy *(Technical Reference)*

### Architecture & Pipeline Diagram

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                     Input Event Drain & Coalescer                          │
│  • Drainscrossterm::event::poll(Duration::ZERO) queue each frame tick     │
│  • Coalesces Scroll deltas (clamped) & resets drag_start/last_drag_y       │
│  • Binds Alt+m, 'M', and status-bar 🔊/🔇 hit-zone -> AppAction::ToggleMute│
└────────────────────────────────────┬───────────────────────────────────────┘
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                  CDP Worker & YouTube Ad-Shield Engine                     │
│  • Termux PulseAudio bootstrap (pulseaudio --start --exit-idle-time=-1)    │
│  • Chromium flags: --autoplay-policy=no-user-gesture-required,             │
│    --enable-features=AudioServiceOutOfProcess, --alsa-output-device=default│
│  • Coalesced Scroll: window.scrollBy(0, coalesced_delta * 72)              │
│  • 250ms + MutationObserver YouTube Ad-Skipper (.ad-showing -> skip/ffwd)  │
│  • Extracts active <video> rects (w>40, h>40) -> META VIDEO:x,y,w,h        │
│  • Boosts Screencast (everyNthFrame: 1, quality: 65) during video playback │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│             Video-Box Fast-Path & DEC Synchronized DiffRenderer            │
│  • Active video playing -> full-frame fast single-pass fsr_rcas subpixel   │
│  • Hybrid Mode -> suppresses DOM text overlays overlapping video_rects     │
│  • Wraps frame buffer in \x1b[?2026h ... \x1b[?2026l single atomic write   │
└────────────────────────────────────────────────────────────────────────────┘
```

### Interactive Component & State Mapping
- **Input Drain & Scroll State**: `poll_action` reads the first event with `timeout` and loops over `event::poll(Duration::ZERO)` to drain queued mouse/touch events, summing scroll steps into a single clamped `AppAction::Scroll(total_delta)` and clearing `drag_start` / `last_drag_y` on release or wheel scroll.
- **Audio State (`audio_muted: bool`)**: Stored on `CdpEngine` and mirrored in the top status bar (`🔊` unmuted, `🔇` muted) with a dedicated clickable hit-zone `mute_hit_range` in `InputHandler`. Toggling sends `CdpWorkerCommand::SetMuted(bool)` which applies `Page.setMuted` / JS `document.querySelectorAll('video, audio').forEach(...)`.
- **Video Rectangles (`video_rects: Arc<Mutex<Vec<(i32, i32, i32, i32)>>>`)**: Populated from the `META` payload whenever visible playing `<video>` elements (`!v.paused && !v.ended && rect.width > 40 && rect.height > 40`) exist. Used by `main.rs` and `DiffRenderer` to suppress overlapping DOM text cells and engage the fast single-pass `fsr_rcas` rendering path.
