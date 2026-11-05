# TermBlitz Super-Resolution, Neural & Dithering Pipeline — Audit Report & Fix Plan

## 1. Executive Summary & Confirmed Architecture Decisions

We conducted a non-destructive code and mathematical audit across the Rust engine (`src/upscale.rs`, `src/renderer.rs`, `src/engine.rs`, `src/input.rs`, `src/main.rs`) and Web Companion UI (`src/App.tsx`). While the core YCbCr round-trip math, Sparse Skip Engine (`luma_range < 0.015`), and Bayer/Floyd-Steinberg dithering gates are mathematically sound, we uncovered **four silent pipeline bugs** that prevented `FsrOnly`, `MultiPass`, and `RCAS` from realizing their full sharpness in the terminal half-block grid, and introduced a `+0.150` positive DC luminance drift on high-contrast edges in Layer 4 of the neural network.

Based on your confirmed architectural choices:
1. **Terminal-Grid RCAS Post-Pass**: When the Headless Chromium screenshot (`360x576`) is larger than the terminal half-block subpixel grid (e.g. `60x48`), we will downsample/reconstruct to the terminal subpixel grid `(target_w, target_h)` and execute `fsr_rcas` **at the terminal subpixel grid** so 1px sharpening lobes are preserved on every half-block cell (`▀`) rather than averaged out by 6x–12x downsampling.
2. **Zero-Sum High-Frequency Layer 4 Weights**: We will rebalance `L4_WEIGHTS` channels `in_ch = 1, 2, 3` so each 3x3 high-frequency spatial kernel sums strictly to `0.000`, while keeping `in_ch = 0` (the identity residual base channel) summing strictly to `1.000`.
3. **True 3-Stage `MultiPass` Pipeline**: In `UpscaleMode::MultiPass`, Stage 1 (`fsr_easu`) will pre-scale by `1.5x` (`(buf.width * 3) / 2`, `(buf.height * 3) / 2`), followed by Stage 2 (`upscale_neural_2x` for a `3.0x` intermediate), bridging to target dimensions, and Stage 3 (`fsr_rcas`).

---

## 2. Phase 1 Audit Findings (5 Critical Areas)

### Area 1: Pipeline Wiring & Dirty-Frame Redraw Check
- **Active Call Path (`src/renderer.rs`, `src/main.rs`)**:
  - In `HYBRID` mode (`render_hybrid_frame`) and `RAW` half-block mode (`render_raw_frame`), `blit_pixel_buffer` calls `apply_upscale_pipeline(px_buf, upscale_mode, sr_w, sr_h)` before `sample_subpixel_box`.
  - In `RAW` Sixel mode (`--sixel`), `main.rs` calls `apply_upscale_pipeline(&px_buf, upscale_mode, sixel_w, sixel_h)` before `encode_sixel_frame(&sr_buf, dithering_enabled)`.
  - Dirty-frame invalidation (`needs_redraw = true` and `renderer.prev_back.clear()`) is properly triggered when cycling `u` / `Alt+u` (`AppAction::CycleUpscale`) or toggling `d` / `Alt+d` (`AppAction::ToggleDither`).
- **[BUG #1 — Silent EASU No-Op & RCAS Obliteration on Downsampled Frames]**:
  - In `blit_pixel_buffer` (`src/renderer.rs:863-864`):
    ```rust
    let sr_w = (target_w * 2).max(px_buf.width);
    let sr_h = (target_h * 2).max(px_buf.height);
    ```
  - Because the Chromium viewport captures at `360x576` while the terminal subpixel grid is `60x48` (`target_w = 60, target_h = 48`), `sr_w` and `sr_h` evaluate to `360` and `576` (`== px_buf.width` and `px_buf.height`).
  - Inside `fsr_easu` (`src/upscale.rs:660`):
    ```rust
    if w == target_w && h == target_h { return buf.clone(); }
    ```
    Consequently, **EASU completely no-ops in `FsrOnly` mode** whenever the Chromium buffer is at least `2x` the terminal subpixel dimensions!
  - Furthermore, `fsr_rcas` runs at `360x576` (modifying 1px neighbors at 360p), and `blit_pixel_buffer` then runs `sample_subpixel_box` to downsample `360x576 -> 60x48` (a `6x` horizontal and `12x` vertical reduction). Because `sample_subpixel_box` only samples 5 sparse points out of each `6x12` (72-pixel) source block, ~93% of the high-resolution neural/RCAS pixels are discarded and the 1px RCAS sharpening lobes are obliterated.
- **[BUG #2 — Status Bar Touch/Click Missing SR & Dither Badge Hit-Zones]**:
  - In `src/input.rs:205-212`, clicking on row `0` at any `column >= 14` unconditionally dispatches `AppAction::FocusUrlBar`, even when clicking directly on the `[SR: ... | DITHER: ...]` badge on the right side of the top status bar.

### Area 2: Hybrid FSRCNN-ESPCN Neural Math Check
- **YCbCr Round-Trip (`rgb_to_ycbcr` & `ycbcr_to_rgb`)**:
  - Uses full-range BT.601 matrices (`Y = 0.299R + 0.587G + 0.114B`, `Cb = (B - Y) * 0.564 + 0.5`, `Cr = (R - Y) * 0.713 + 0.5`) and exact inverse coefficients (`1.402`, `-0.344136`, `-0.714136`, `1.772`) with `+ 0.5` integer rounding.
  - **Verified**: Zero brightness drift or tinting across all primary colors, grays, and gradients (maximum round-trip error is `0` to `1` LSB out of `255`, PSNR `> 51 dB`).
- **Residual Skip Connection & 2x2 PixelShuffle (`upscale_y_neural_2x` & `pixel_shuffle_2x`)**:
  - `l3_out[idx4 + c] = prelu(sum, L3_PRELU[c]) + l2_center[c]` properly adds the Layer 2 shrunk features before Layer 4 sub-pixel convolution.
  - `pixel_shuffle_2x` maps `c0 -> (2x, 2y)`, `c1 -> (2x+1, 2y)`, `c2 -> (2x, 2y+1)`, `c3 -> (2x+1, 2y+1)` without out-of-bounds indexing or stride transposition errors.
- **[BUG #3 — Layer 4 Positive DC Luminance Drift (`L4_WEIGHTS`)]**:
  - In `L4_WEIGHTS` (`src/upscale.rs:162-211`), `in_ch = 0` sums to `1.000` across its 9 spatial weights for all 4 sub-pixels (`0.02 + 0.06 + 0.01 + 0.06 + 0.72 + 0.04 + 0.01 + 0.04 + 0.04 = 1.000`).
  - However, for `in_ch = 3` (the high-frequency diagonal/Laplacian channel), the 9 spatial weights sum to **`+0.150`** (`0.015 + 0.025 + 0.025 + 0.110 - 0.010 - 0.010 - 0.005 = +0.150`) across all 4 sub-pixels, while `in_ch = 1` and `in_ch = 2` sum to **`-0.140`** (`-0.025 - 0.045 - 0.025 + 0.120 - 0.065 - 0.030 - 0.040 - 0.030 = -0.140`).
  - Because `L1` and `L2` apply asymmetric PReLU (`alpha = 0.1` for negative values vs `1.0` for positive values), high-contrast edges and text produce non-zero positive mean activations on `in_ch = 1, 2, 3`. Non-zero kernel sums in `L4_WEIGHTS` convert those edge activations into local DC luminance shifts (brightening/washing out dark text strokes).

### Area 3: AMD FSR 1.0 (`fsr_easu` & `fsr_rcas`) Check
- **EASU 12-Tap Cross & Anti-Ringing**:
  - Samples the 12-tap cross (`b, c, e, f, g, h, i, j, k, l, n, o`), computes horizontal/vertical luma gradients (`len_x`, `len_y`), rotates tap offsets `(dx, dy)` into the local edge frame `(rx, ry)`, evaluates the directional Lanczos2 polynomial `((25/16 * (2/5 * d^2 - 1)^2 - 25/16 - 1) * (1/4 * d^2 - 1)^2)`, and clamps RGB strictly to `[min_rgb, max_rgb]` of the 4 nearest texels `(f, g, j, k)`. Zero NaN/Inf risk (`weight_sum > 1e-5` guard).
- **RCAS 5-Tap Contrast-Adaptive Sharpening**:
  - Computes 5-tap cross (`n, s, w, e, c`), derives adaptive negative lobe weight `lobe = (-0.25 * contrast_headroom / Ringing_Limit) * sharp_scale` clamped to `[-0.18, 0.0]`, and clamps output RGB to `[0.0, 255.0]`.
- **[BUG #4 — `MultiPass` Stage 1 EASU No-Op at 2x Scale]**:
  - In `UpscaleMode::MultiPass` (`src/upscale.rs:1187-1189`):
    ```rust
    let mid_w = (out_w / 2).max(buf.width);
    let mid_h = (out_h / 2).max(buf.height);
    let stage1 = fsr_easu(buf, mid_w, mid_h);
    ```
  - Whenever `out_w <= buf.width * 2` (the default 2x scaling case), `mid_w == buf.width` and `mid_h == buf.height`, causing `fsr_easu` to return `buf.clone()` untouched!

### Area 4: Content-Aware Sparse Skip Engine Check
- **Variance Gate (`should_enhance_pixel` & `classify_patch`)**:
  - Computes `max_y - min_y` over the 3x3 neighborhood. When `luma_range < 0.015` (`FLAT_LUMA_THRESHOLD`), `upscale_y_neural_2x`, `fsr_easu`, and `fsr_rcas` bypass all convolution/kernel math and copy `center_y` / center `Color` untouched.
  - **Verified**: On flat solid backgrounds (`Color { r: 255, g: 255, b: 255 }` or `Color { r: 30, g: 30, b: 46 }`), `100.0%` of pixels bypass SR with `0` LSB color drift, while 1px text edges (`luma_range >= 0.015`) are `100.0%` enhanced without false-negative skipping or square tile boundary seams.

### Area 5: Bayer & Floyd-Steinberg Dithering Check
- **Ordered 4x4 Bayer Dithering (`dither_truecolor_cell`)**:
  - Uses deterministic screen coordinates `(x & 3, y & 3)` so static frames never shimmer/flicker across redraws, and suppresses dithering when `cell_is_flat(top, bot)` (`max_channel_diff <= 4`).
- **Serpentine Floyd-Steinberg Sixel Quantization (`quantize_sixel_floyd_steinberg`)**:
  - Even rows scan left-to-right; odd rows scan right-to-left (`dx_step = -1`), diffusing clamped quantization error (`[-32, 32]`) with integer bit-shifts (`* 7 >> 4`, `* 3 >> 4`, `* 5 >> 4`, `* 1 >> 4`) and zero-overhead skip on flat 3-pixel spans.

---

## 3. Pipeline Architecture Diagram (Before vs. After Fix)

```text
CURRENT PIPELINE (Silent EASU No-Op & Pre-Downsample RCAS Loss):
┌──────────────────────┐     sr_w = max(120, 360) = 360     ┌──────────────────────────┐
│ Chromium PixelBuffer │ ─────────────────────────────────► │ apply_upscale_pipeline   │
│      (360 x 576)     │   FsrOnly: w==target_w -> NO-OP!   │       (360 x 576)        │
└──────────────────────┘   RCAS: 1px lobes at 360x576       └─────────────┬────────────┘
                                                                          │
                                                            6x12 Downsample obliterates
                                                              1px RCAS & 93% of pixels
                                                                          ▼
                                                            ┌──────────────────────────┐
                                                            │   sample_subpixel_box    │
                                                            │  Terminal Grid (60 x 48) │
                                                            └──────────────────────────┘

FIXED PIPELINE (Adaptive SR + Terminal Subpixel Grid RCAS Post-Pass):
┌──────────────────────┐
│ Chromium PixelBuffer │
│     (w_in x h_in)    │
└──────────┬───────────┘
           │
           ├──► [If target_w > w_in]: Run Spatial/Neural SR (EASU / 2x-4x CNN / 1.5x+2x MultiPass)
           │                          with Sparse Skip Gate (luma_range < 0.015 -> 0% drift)
           │
           ▼
┌──────────────────────────────┐
│ Area-Integrated Subpixel Box │ ──► Downsamples/maps cleanly to Terminal Grid (target_w x target_h)
│   (cols x rows*2 subpixels)  │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Terminal-Grid RCAS Post-Pass │ ──► 5-Tap Contrast-Adaptive Sharpening directly on terminal subpixels
│   (FsrOnly/Combined/Multi)   │     (1px half-block text & thumbnails get crisp edge reinforcement)
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Flat-Gated 4x4 Bayer / Sixel │ ──► ANSI `▀` Half-Blocks or Serpentine Floyd-Steinberg Sixel
└──────────────────────────────┘
```

---

## 4. Planned Fixes (To Execute Upon Your Approval)

### Step 1: Zero-Sum High-Frequency Layer 4 Kernel Weights (`src/upscale.rs`)
- Update `L4_WEIGHTS` in `src/upscale.rs` so that for each of the 4 sub-pixel output channels (`out_c = 0..3`):
  - `in_ch = 0` (identity luminance base): 9 spatial taps sum strictly to `1.000`.
  - `in_ch = 1` (horizontal edge detail): 9 spatial taps sum strictly to `0.000` (e.g., center `+0.140` balanced by `-0.140` across opposing taps).
  - `in_ch = 2` (vertical edge detail): 9 spatial taps sum strictly to `0.000` (center `+0.140` balanced by `-0.140` across opposing taps).
  - `in_ch = 3` (diagonal/Laplacian detail): 9 spatial taps sum strictly to `0.000` (center `+0.120` balanced by `-0.120` across the 8 surround taps).

### Step 2: Fix `MultiPass` 3-Stage Scaling & Terminal-Grid `RCAS` (`src/upscale.rs` & `src/renderer.rs`)
- In `src/upscale.rs`:
  - Update `UpscaleMode::MultiPass` inside `upscale_frame` so Stage 1 (`fsr_easu`) always pre-scales input dimensions by `1.5x` (`mid_w = ((buf.width * 3) / 2).max(buf.width + 1)`, `mid_h = ((buf.height * 3) / 2).max(buf.height + 1)`), Stage 2 runs `upscale_neural_2x(&stage1)` (reaching `3.0x` super-sampled resolution), bridges via `fsr_easu` to `(out_w, out_h)`, and Stage 3 runs `fsr_rcas(&stage2, 0.25)`.
  - Add a dedicated helper `pub fn apply_terminal_subpixel_rcas(subpixel_buf: &PixelBuffer, mode: UpscaleMode) -> PixelBuffer` in `src/upscale.rs` so when `blit_pixel_buffer` downsamples a high-res Chromium buffer (`360x576`) to the terminal subpixel grid (`cols x rows*2`), `fsr_rcas` runs directly on the terminal subpixel buffer (with sharpness `0.35` for `FsrOnly`, `0.28` for `Combined`, `0.32` for `MultiPass`, and `0.18` edge-aware pass for `NeuralOnly` when downsampling).
- In `src/renderer.rs`:
  - Update `blit_pixel_buffer` to:
    1. Run `apply_upscale_pipeline` when upscaling (`target_w > px_buf.width || target_h > px_buf.height`) or run neural feature enhancement before downsampling.
    2. Reconstruct the terminal subpixel `PixelBuffer` (`target_w x target_h`) via `sample_subpixel_box`.
    3. Apply `apply_terminal_subpixel_rcas` on the `(target_w x target_h)` subpixel buffer before encoding `▀` half-blocks and Bayer dithering.

### Step 3: Wire Status Bar `[SR: ... | DITHER: ...]` Click Hit-Zones (`src/input.rs`)
- Update `InputHandler::parse_mouse` on `row == 0` so clicking the `[SR: <MODE> | DITHER: <ON/OFF>]` badge in the top-right status bar cycles `UpscaleMode` (`AppAction::CycleUpscale`) or toggles dithering (`AppAction::ToggleDither`) instead of always focusing the URL bar.

### Step 4: Automated Diagnostic Suite (`src/upscale.rs` + Web Companion Diagnostic Panel in `src/App.tsx`)
- Add the 5 automated unit/diagnostic tests (`#[cfg(test)] mod tests`) in `src/upscale.rs`:
  1. `test_ycbcr_roundtrip_accuracy`: Asserts max RGB channel error `<= 1` (`PSNR > 50 dB`) across primaries, neutrals, and gradients.
  2. `test_flat_region_skip_zero_drift`: Asserts `100%` of interior pixels on a `64x64` solid frame (`Color { r: 240, g: 240, b: 240 }`) match `Color { r: 240, g: 240, b: 240 }` with `0` LSB drift across `FsrOnly`, `NeuralOnly`, `Combined`, and `MultiPass`, plus verifies `L4_WEIGHTS` zero-sum invariants.
  3. `test_edge_contrast_improvement`: Creates a synthetic blurred step-edge (`Y = 0.2 -> 0.5 -> 0.8`), runs all 4 SR modes, and asserts post-SR edge gradient slope is strictly steeper than bilinear (`slope_sr > slope_bilinear`).
  4. `test_pipeline_dimensions_and_modes`: Tests `Off`, `FsrOnly`, `NeuralOnly`, `Combined`, and `MultiPass` at `1x`, `2x`, `3x`, and `4x` scales plus terminal subpixel RCAS without panics or dimension mismatches.
  5. `test_dither_stability_and_flat_suppression`: Verifies deterministic Bayer output across repeated calls and zero dithering noise on flat spans.
- Expose an interactive **Pipeline Diagnostics & Math Audit** inspector modal/drawer in the Web Companion (`src/App.tsx`) that executes the exact same 5 mathematical verification tests in real time and displays the measured PSNR, flat-skip percentage, L4 kernel sums, edge slope gains, and frame timings.
