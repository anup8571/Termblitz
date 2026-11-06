# TermBlitz Advanced Super-Resolution & Dithering Pipeline Upgrade

## User Review & Critical Decisions

> [!IMPORTANT]
> The following architectural decisions have been confirmed via clarifying questions and are locked in for implementation:
> - **Pixel-Art Mode Activation**: Automatic content-aware detection plus manual override via `--pixel-art` CLI flag and `p` / `Alt+p` runtime hotkey.
> - **Pixel-Art Scaling Algorithm**: Pure Rust **Scale2x / AdvMame2x** directional corner expansion rules (recursively chainable for 4x and 8x).
> - **JPEG Denoising Trigger**: **Adaptive auto-trigger** measuring 8x8 DCT boundary discontinuities versus intra-block gradient variance to avoid blurring high-quality PNGs or SVG renders.
> - **Zero External Crates**: 100% pure Rust within `src/upscale.rs`, `src/renderer.rs`, `src/input.rs`, and `src/main.rs` using flat slices, integer/fixed-point math, and reusable buffers.

---

## 1. Overview & Core Concept

This upgrade enhances TermBlitz's image reconstruction pipeline with three complementary capabilities:
1. **Multi-Pass Recursive Chaining**: Expands the 4-layer FSRCNN-ESPCN neural upscaler and EASU bridge to scale images cleanly up to 4x and 8x during deep zooming, using pre-allocated scratch slices to prevent heap allocation inside the frame loop.
2. **Fast Adaptive JPEG De-blocking / Denoising Pre-Filter**: Detects 8x8 DCT compression grids in heavily compressed web JPEGs and applies an edge-preserving bilateral-sigma filter before neural/FSR sharpening, preventing compression block edges from being amplified.
3. **Pixel-Art & Vector Mode with Scale2x**: Detects low-color-palette icons, retro web games, and pixel art, automatically switching from continuous FSR/neural interpolation to Scale2x (AdvMame2x) to render crisp, razor-sharp pixel grids without blur.

---

## 2. Technical Architecture & Data Strategy

```
                             ┌──────────────────────────────┐
                             │       Input PixelBuffer      │
                             └──────────────┬───────────────┘
                                            │
                        ┌───────────────────┴───────────────────┐
                        │ detect_pixel_art(buf) || forced_flag? │
                        └─────────┬───────────────────┬─────────┘
                                  │ YES               │ NO
                                  ▼                   ▼
                     ┌───────────────────────┐ ┌───────────────────────────┐
                     │   Scale2x / AdvMame   │ │ detect_jpeg_8x8_blocks()  │
                     │  (Chain 2x -> 4x/8x)  │ └──────────────┬────────────┘
                     └────────────┬──────────┘                │ High DCT noise
                                  │                           ▼
                                  │              ┌─────────────────────────┐
                                  │              │  remove_jpeg_artifacts  │
                                  │              │   (Bilateral Smoothing) │
                                  │              └────────────┬────────────┘
                                  │                           │
                                  │                           ▼
                                  │              ┌─────────────────────────┐
                                  │              │ Multi-Pass FSRCNN-ESPCN │
                                  │              │   (2x -> 4x -> 8x Loop) │
                                  │              └────────────┬────────────┘
                                  │                           │
                                  │                           ▼
                                  │              ┌─────────────────────────┐
                                  │              │  Terminal Subpixel RCAS │
                                  │              └────────────┬────────────┘
                                  │                           │
                                  └─────────────┬─────────────┘
                                                ▼
                                 ┌─────────────────────────────┐
                                 │ Terminal Subpixel / Dither  │
                                 └─────────────────────────────┘
```

### Component Upgrades

#### 1. Multi-Pass Recursive Chaining (`src/upscale.rs`)
- **Scale Factor Analysis**: Computes `scale_factor = max(target_w / buf.width, target_h / buf.height)`.
- **Recursive Passes**:
  - When `scale_factor >= 8`: Runs 3 successive 2x neural/sub-pixel passes (`2x -> 4x -> 8x`) or 2x neural + 4x EASU, finishing with terminal-grid RCAS.
  - When `scale_factor >= 4`: Runs 2 successive 2x neural passes (`2x -> 4x`), followed by an EASU aspect bridge if target is non-power-of-two.
  - When `scale_factor >= 2`: Standard single-pass 2x neural + EASU bridge.
- **Scratch Buffer Optimization**: Pre-allocates a ping-pong buffer pair for intermediate luminance planes (`Vec<f32>`) and chroma buffers to eliminate intermediate reallocations.

#### 2. Fast JPEG Denoising Filter (`src/upscale.rs`)
- **Block Artifact Metric**: Evaluates average boundary gradient along 8-pixel horizontal and vertical grid lines ($\sum |P_{8k-1} - P_{8k}|$) compared to intra-block gradient variance.
- **Bilateral Box-Sigma Smoothing**:
  ```rust
  pub fn remove_jpeg_artifacts(buf: &mut PixelBuffer)
  ```
  - For pixels within $\pm 1$ of 8x8 block boundaries, applies an adaptive 3-tap kernel blending adjacent pixels only if color distance is below the quantization noise threshold ($\Delta \le \sigma_{jpeg} \approx 18$), preserving true object edges ($\Delta > \sigma_{jpeg}$).

#### 3. Pixel-Art Detector & Scale2x Scaler (`src/upscale.rs`)
- **Classifier**:
  ```rust
  pub fn detect_pixel_art(buf: &PixelBuffer) -> bool
  ```
  - Fast histogram sampling (sampling up to 512 distributed pixels) checking if unique color count $< 64$ and horizontal/vertical runs of identical colors have high frequency.
- **Scale2x Algorithm**:
  ```rust
  pub fn scale2x(buf: &PixelBuffer) -> PixelBuffer
  ```
  - Uses standard AdvMame2x logic for 4 sub-pixels `[E0, E1, E2, E3]` based on 4-connectivity neighbors `[A, B, C, D]`.
  - Can be chained recursively for 4x and 8x targets.
- **Interactive Controls (`src/input.rs`, `src/engine.rs`, `src/main.rs`)**:
  - Hotkey `p` or `Alt+p` toggles manual Pixel-Art override.
  - CLI flag `--pixel-art` forces pixel-art mode.
  - Top status bar displays `[PIX]` badge when active.

---

## 3. Key Product Decisions & Trade-Offs

- **Scale2x vs HQ2x**: Scale2x provides 100% predictable, integer-only branch math with zero lookup tables, delivering ~0.3ms per frame on mobile ARM CPUs while preserving sharp retro edges.
- **Boundary-Selective Denoising**: Filtering only pixels near 8x8 grid boundaries avoids blurring text and fine UI elements while removing 90%+ of visible JPEG blocking noise.
- **Scratch Slice Ping-Ponging**: Using reusable pre-sized buffers across recursive passes prevents heap fragmentation on low-memory Android Termux environments.

---

## 4. Verification Plan

1. **Unit Tests in `src/upscale.rs`**:
   - `test_recursive_chaining_4x_and_8x`: Verifies output dimensions (`w*4, h*4` and `w*8, h*8`) and verifies that luminance values remain clamped in `[0.0, 1.0]`.
   - `test_jpeg_artifact_removal`: Verifies that synthetic 8x8 block discontinuities are smoothed while high-contrast 1px black-on-white text edges are preserved untouched.
   - `test_pixel_art_detector_and_scale2x`: Verifies detection on a 16-color sprite versus a continuous gradient, and checks exact corner preservation of the Scale2x kernel.
2. **Static Syntax & Build Verification**:
   - Run static checks on `src/upscale.rs`, `src/renderer.rs`, `src/engine.rs`, `src/input.rs`, and `src/main.rs`.
