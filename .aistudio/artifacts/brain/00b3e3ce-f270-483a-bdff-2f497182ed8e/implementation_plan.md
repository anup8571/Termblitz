# TermBlitz Carbonyl-Grade Architectural Upgrade & Optimization Plan

This plan cross-references every finding in `CARBONYL_VS_TERMBLITZ_ANALYSIS.md` against the current TermBlitz Rust codebase and defines a pure-Rust architectural upgrade to deliver **60 FPS butter-smooth scrolling**, **2x2 Unicode quadrant subpixel clarity**, **compressed differential ANSI emission**, and **zero-jank Hybrid DOM text synchronization**—without sacrificing TermBlitz's standalone portability or any existing features (2x/4x/8x Super-Resolution, JPEG Bilateral Denoising, Scale2x Pixel-Art mode, Sixel/Kitty protocols, AdBlock, Audio Mute, Direct Input Typing, and Google Search Omnibox).

---

## User Review & Critical Decisions

> [!IMPORTANT]
> **Zero Feature Removal Guarantee**: All existing hotkeys, CLI flags, Super-Resolution modes (`u`), dithering (`d`), pixel-art Scale2x (`p`), audio mute (`a`), AdBlock (`b`), and Native/Hybrid/Raw modes (`m`) are preserved intact. The upgrades below eliminate the root bottlenecks identified in `CARBONYL_VS_TERMBLITZ_ANALYSIS.md`.

- **Decision 1 — Motion-Adaptive Dual Rendering Pipeline**: During active scrolling (`last_scroll_instant.elapsed() < 220ms`) or active video playback, automatically bypass heavy multi-layer neural convolutions and synchronous DOM `TreeWalker` reflows, switching to a **Carbonyl-style integer ALU 2x2 Quadrant / Half-Block fast blitter**. The moment scrolling settles (`>= 220ms`), run the full selected Super-Resolution (`2x`/`4x`/`8x` + RCAS) and fresh occlusion-checked DOM extraction pass.
- **Decision 2 — Predictive Scroll-Delta Overlay Sync (Instead of Blank Text During Scroll)**: Rather than hiding DOM text overlays during scrolling or blocking Chromium's main thread with a 140ms `TreeWalker` reflow lock, tag each extracted span with its capture-time `scroll_top` and `is_fixed` state. During active scroll motion, shift non-fixed spans in Rust by `-Δscroll_y` on every frame so text moves in 1:1 lockstep with the scrolling background at 0ms JS cost.
- **Decision 3 — Carbonyl 2x2 Quadrant Subpixel Engine (`▖▗▘▙▚▛▜▝▞▟▀▄▌▐█ `)**: Upgrade the terminal subpixel blitter to sample a **2x4 pixel grid per cell** using Carbonyl's integer luminance formula `(R*77 + G*150 + B*29) >> 8` and dynamic midpoint binarization (`mid = min + (max - min) / 2`), doubling horizontal resolution for vertical borders, icons, and curves while preserving clean half-blocks (`▀`/`▄`) and flat spaces (`' '`) where appropriate.

---

## 1. Cross-Referenced Root-Cause Diagnosis

| Bottleneck in `CARBONYL_VS_TERMBLITZ_ANALYSIS.md` | Current TermBlitz Code Location | Root-Cause Impact | Planned Pure-Rust Fix |
| :--- | :--- | :--- | :--- |
| **1. 140ms DOM `TreeWalker` Reflow Lock During Scroll** | `engine.rs` (`request_text_extraction` called immediately inside `coalesced_scroll_delta != 0` and every `140ms` during scroll) | Forces Chromium to run `getComputedStyle`, `getClientRects`, and `elementFromPoint` on 400+ nodes mid-scroll, freezing Skia screencast generation for 50–120ms. | **Scroll-Gated Extraction + Predictive Rust Scroll Shift**: Suspend `TreeWalker` during active scroll (`< 220ms`); shift existing non-sticky overlays by `-Δy` in Rust; run full `TreeWalker` once on scroll settle. |
| **2. Synchronous JS Scroll Evaluation Overhead** | `engine.rs` (`scroll_js` running a 100-node `querySelectorAll` + `getComputedStyle` loop on every scroll tick) | Evaluates a heavy DOM search script on V8's main thread every tick instead of letting Chromium's compositor thread handle wheel scroll. | **Native CDP `Input.dispatchMouseEvent` (`mouseWheel`) + Cached O(1) Scroll Offset Sync**: Dispatch hardware-accelerated CDP `mouseWheel` events and read back `scrollY` in O(1) without DOM tree scans. |
| **3. 1x2 Half-Block Horizontal Resolution Limit** | `renderer.rs` (`sample_subpixel_box` + `blit_pixel_buffer` emitting only `'▀'` and `' '`) | Only 1 horizontal sample per column; vertical 1px lines and diagonal edges become blocky or require heavy RCAS sharpening. | **Carbonyl 2x4 -> 2x2 Quadrant Binarizer (`binarize_quadrant`)**: Sample 4 quadrants (`TL, TR, BL, BR`) per cell using `(R*77 + G*150 + B*29) >> 8` and 16-state quadrant lookup table with UI border preservation. |
| **4. Uncompressed 24-Bit SGR Flood & Full `\x1b[0m` Resets** | `renderer.rs` (`build_synchronized_frame` lines 1370–1406) | Emits 38 bytes of TrueColor SGR per cell change and resets both `last_fg` and `last_bg` whenever `bold`/`underline` toggles, flooding the Termux PTY buffer with 120–200 KB/frame. | **Targeted SGR Attribute Diffing (`\x1b[22m`/`\x1b[24m`), Perceptual Color Tolerance During Motion, Relative Cursor Moves (`\x1b[nC`), and Optional Fast 256-Color Mode (`rgb_to_xterm256`)**. |
| **5. Main Loop `thread::sleep` Frame Pacing Drift** | `main.rs` (`std::thread::sleep(target_frame_time - elapsed)` at line 995) | Fixed sleep granularity misses asynchronous CDP screencast frame arrivals by 4–12ms. | **Event-Driven Saturating Deadline Pacer**: Wake immediately when input arrives or when `cdp_engine.get_frame_count()` increments. |

---

## 2. Step-by-Step Architectural Upgrade Plan

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                   UPGRADED TERMBLITZ 60-FPS RENDERING ARCHITECTURE                 │
│                                                                                    │
│  [Input Thread / Batcher]                                                          │
│     │ Coalesced Scroll Delta + Pointer Pos                                         │
│     ▼                                                                              │
│  [CDP Worker Thread (engine.rs)]                                                   │
│     ├─► 1. Native Compositor Wheel (`Input.dispatchMouseEvent: mouseWheel`)        │
│     │      + O(1) Cached Scroll Target Fallback & `scroll_y` Readback              │
│     ├─► 2. Latest-Frame Screencast Drain (Immediate `screencastFrameAck`,          │
│     │      discard stale queued frames, zero-copy `as_rgb8()` decode)              │
│     └─► 3. Scroll-Gated DOM Extraction:                                            │
│            • While Scrolling (<220ms): ZERO `TreeWalker` calls; shift non-fixed    │
│              `TextOverlaySpan.px_y` in lockstep with live `scroll_y` delta         │
│            • On Scroll Settle (>=220ms): Run full occlusion-checked `TreeWalker`   │
│                                                                                    │
│  [DiffRenderer (renderer.rs)]                                                      │
│     ├─► Motion Fast-Path: Direct 2x4 Integer ALU Quadrant Sampler                  │
│     ├─► Static Settle Pass: Full 2x/4x/8x Neural SR + Bilateral Denoise + RCAS     │
│     ├─► 2x2 Quadrant Binarizer (`▖▗▘▙▚▛▜▝▞▟▀▄▌▐█ `) + UI Border Accentuation       │
│     └─► Compressed Differential ANSI Writer:                                       │
│            • Specific SGR Off-Codes (`\x1b[22m`, `\x1b[24m`) — No `\x1b[0m` Wipe   │
│            • Perceptual Color Delta Gate + Fast xterm-256 Motion Quantizer         │
│            • Relative Cursor Jumps (`\x1b[nC`) + Atomic DEC 2026 Single Write      │
└────────────────────────────────────────────────────────────────────────────────────┘
```

### Pillar 1: Butter-Smooth 60FPS Scrolling & Dynamic Motion Fast-Path
1. **Native Compositor Wheel + O(1) Scroll Sync**:
   - Dispatch CDP `Input.dispatchMouseEvent` (`type: "mouseWheel"`, `deltaX: 0`, `deltaY: pixel_delta`) so Chromium's compositor thread scrolls layers natively at 60 FPS.
   - Replace the 100-element `querySelectorAll` loop on every scroll tick with a cached scroll target reference (`window.__termblitz_scroll_el`) that only scans for a scrollable container if the cached element cannot scroll in `dy`'s direction, and returns the updated `[scrollTop, scrollPct]` in O(1) time.
2. **Scroll-Gated `TreeWalker` Suppression**:
   - While `last_scroll_instant.elapsed() < Duration::from_millis(220)`, completely suppress `request_text_extraction` (`TreeWalker` + `getComputedStyle` + `elementFromPoint`) so Chromium's main thread never pauses compositor frame production during scrolling.
   - As soon as scrolling pauses for `220ms`, automatically fire `request_text_extraction` once to refresh newly revealed DOM nodes.
3. **Screencast Queue Draining & Motion Quality Scaling**:
   - In the WebSocket poll loop, acknowledge every `Page.screencastFrame` immediately (`Page.screencastFrameAck`) while retaining only the final `latest_screencast_b64` payload in the batch so stale intermediate frames are never decoded.
   - When `scroll_fast_path` is active, bypass multi-layer YCbCr convolutions and RCAS passes altogether, running only the single-pass integer ALU subpixel sampler, then triggering a crisp high-resolution settle render as soon as `scroll_fast_path` expires.

### Pillar 2: Carbonyl-Grade 2x2 Quadrant Subpixel & Color ALU Engine
1. **Fast Integer Luminance & Color Averaging ALU**:
   - Implement `luma_fast_u32(c: Color) -> u32` using `(c.r as u32 * 77 + c.g as u32 * 150 + c.b as u32 * 29) >> 8` and `avg_color(c1, c2)` using `((c1 + c2) >> 1)` integer bit-shifts.
2. **2x4 Subpixel Sampling -> 16-State Quadrant Binarizer (`binarize_quadrant`)**:
   - For each terminal cell `(col, row)`, sample a **2x2 quadrant grid** (`tl, tr, bl, br`), where each quadrant averages its vertical pixel pair from the source/upscaled buffer while preserving 1px light-gray UI borders (`#dfe1e5`).
   - Compute `min`, `max`, and `mid = min + (max - min) / 2` across `(tl, tr, bl, br)`:
     - If `max - min < 6` (flat region): emit `' '` with the averaged background color (maximizing run-length batching and eliminating Bayer noise on flat surfaces).
     - If left and right columns match (`tl == tr && bl == br`): emit clean horizontal half-blocks (`'▀'`) with exact top/bottom TrueColor values.
     - Otherwise: classify the 4-bit quadrant mask `((a > mid) << 3) | ((b > mid) << 2) | ((c > mid) << 1) | (d > mid)` into the 16 Unicode quadrant glyphs (`▖`, `▗`, `▄`, `▝`, `▐`, `▞`, `▟`, `▘`, `▚`, `▌`, `▙`, `▀`, `▜`, `▛`, `█`, `' '`) with cluster-averaged foreground and background colors.

### Pillar 3: Zero-Tearing Differential ANSI Emitter & SGR State Compression
1. **Eliminate Destructive `\x1b[0m` Resets**:
   - Currently, toggling `bold` or `underline` emits `\x1b[0...m`, which wipes the terminal's active foreground and background colors and forces 38 bytes of `\x1b[38;2;...m\x1b[48;2;...m` to be re-sent.
   - Replace `\x1b[0m` with targeted SGR attribute transitions: `\x1b[1m` (bold on), `\x1b[22m` (bold off), `\x1b[4m` (underline on), `\x1b[24m` (underline off), preserving active `last_fg` and `last_bg` across the entire frame.
2. **Perceptual SGR Color Delta Gate & Fast xterm-256 Option**:
   - Add `color_dist_sq(c1, c2)` and `rgb_to_xterm256(c: Color) -> u8`.
   - During active scroll/video fast-path (or when `TERMBLITZ_FAST_COLOR=1` is set), coalesce SGR color updates when adjacent cells differ by less than a sub-perceptual RGB threshold (`max_channel_diff <= 2`), cutting ANSI escape sequence byte volume by 35–55% without visible degradation.
3. **Minimal Cursor Movement Encoding**:
   - When the next dirty cell is on the **same row** and only `1..=4` columns ahead of `cursor_pos`, emit `\x1b[{n}C` (Cursor Forward, 4 bytes) instead of `\x1b[{row};{col}H` (7–10 bytes).

### Pillar 4: Zero-Jank Hybrid DOM Overlay Sync & Frame Pacing
1. **Capture-Time Scroll Offset Tagging & Predictive Shift**:
   - Extend `request_text_extraction` to include the exact `scrollTop` in the `META` header and a `fixed` flag (`1` if the element or an ancestor has `position: fixed` or `position: sticky`, `0` otherwise) on each extracted item.
   - Store `overlay_capture_scroll_y: AtomicI32` and `live_scroll_y: AtomicI32` in `CdpEngine`.
   - When rendering Hybrid overlays in `main.rs`, compute `scroll_diff = live_scroll_y - overlay_capture_scroll_y`. For any span where `!span.is_fixed`, adjust its vertical pixel coordinate by `effective_px_y = span.px_y - scroll_diff` before mapping to terminal rows!
   - This guarantees that as the user scrolls, DOM text overlays glide synchronously with the background bitmap at 60 FPS with zero lag and zero `TreeWalker` stalls.
2. **Saturating Deadline Frame Pacer**:
   - Upgrade the main event/render loop in `main.rs` so input polling waits only up to `deadline.saturating_duration_since(Instant::now())` and wakes early whenever a new CDP screencast frame arrives, Triggering an immediate high-resolution settle redraw the instant `scroll_fast_until` completes.

---

## 3. Verification & Diagnostic Test Suite Additions

We will add unit tests verifying every new subsystem so `cargo test --verbose` validates:
1. `test_carbonyl_quadrant_binarization`: Verifies all 16 quadrant bitmask states (`▖▗▘▙▚▛▜▝▞▟▀▄▌▐█ `), integer luminance `(R*77 + G*150 + B*29) >> 8`, and flat-cell space emission.
2. `test_sgr_delta_compression_and_xterm256`: Verifies that toggling `bold`/`underline` uses `\x1b[22m`/`\x1b[24m` without resetting `fg`/`bg`, and validates `rgb_to_xterm256` grayscale and 6x6x6 cube mapping.
3. `test_predictive_scroll_overlay_shift`: Verifies that non-fixed `TextOverlaySpan` items shift by `-Δscroll_y` during active scrolling while `is_fixed` sticky headers/footers remain anchored.
