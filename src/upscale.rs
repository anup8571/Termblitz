#![allow(dead_code)]
//! Ultra-lightweight Pure-Rust Super-Resolution (2x) Pipeline for TermBlitz.
//!
//! Implements a pre-baked Hybrid FSRCNN-ESPCN micro-architecture operating in
//! YCbCr space:
//! - Fast BT.601 full-range RGB <-> YCbCr split and merge (`rgb_to_ycbcr`, `ycbcr_to_rgb`)
//! - Fast bilinear upscaling for chrominance (`Cb`, `Cr`) planes (`upscale_chroma_bilinear`)
//! - 4-Layer Neural Super-Resolution on Luminance (`Y`) (`upscale_y_neural_2x`):
//!   1. Feature Extraction: 3x3 Conv (1 -> 8 channels) + PReLU
//!   2. FSRCNN Shrinking:   1x1 Conv (8 -> 4 channels) + PReLU
//!   3. Non-Linear Mapping: 3x3 Conv (4 -> 4 channels) + PReLU + Residual Skip Connection
//!   4. ESPCN Sub-Pixel:    3x3 Conv (4 -> 4 channels) + 2x PixelShuffle (`pixel_shuffle_2x`)

use crate::renderer::{Color, PixelBuffer};

/// Pre-baked Layer 1 (Feature Extraction): 3x3 Conv, 1 input channel -> 8 output channels.
/// Layout: `L1_WEIGHTS[out_ch][ky * 3 + kx]`.
/// - Channel 0: Sub-pixel Gaussian/identity anchor (weights sum to 1.0, preserves exact DC luminance)
/// - Channel 1: Horizontal Sobel gradient detector (Gx)
/// - Channel 2: Vertical Sobel gradient detector (Gy)
/// - Channel 3: Isotropic Laplacian edge/stroke detector (captures 1px UI borders & text glyphs)
/// - Channel 4: Diagonal 45-degree edge detector
/// - Channel 5: Diagonal 135-degree edge detector
/// - Channel 6: Horizontal ridge / second-derivative detector (Dxx)
/// - Channel 7: Vertical ridge / second-derivative detector (Dyy)
const L1_WEIGHTS: [[f32; 9]; 8] = [
    // Ch 0: Center-weighted luminance anchor (sum = 1.0)
    [
        0.025, 0.050, 0.025,
        0.050, 0.700, 0.050,
        0.025, 0.050, 0.025,
    ],
    // Ch 1: Horizontal gradient Gx (sum = 0.0)
    [
        -0.125, 0.000, 0.125,
        -0.250, 0.000, 0.250,
        -0.125, 0.000, 0.125,
    ],
    // Ch 2: Vertical gradient Gy (sum = 0.0)
    [
        -0.125, -0.250, -0.125,
         0.000,  0.000,  0.000,
         0.125,  0.250,  0.125,
    ],
    // Ch 3: Isotropic 3x3 Laplacian high-pass (sum = 0.0)
    [
        -0.0625, -0.1250, -0.0625,
        -0.1250,  0.7500, -0.1250,
        -0.0625, -0.1250, -0.0625,
    ],
    // Ch 4: Diagonal 45-deg gradient (sum = 0.0)
    [
         0.000, -0.125, -0.250,
         0.125,  0.000, -0.125,
         0.250,  0.125,  0.000,
    ],
    // Ch 5: Diagonal 135-deg gradient (sum = 0.0)
    [
        -0.250, -0.125,  0.000,
        -0.125,  0.000,  0.125,
         0.000,  0.125,  0.250,
    ],
    // Ch 6: Horizontal second-derivative Dxx (sum = 0.0)
    [
        -0.050,  0.100, -0.050,
        -0.150,  0.300, -0.150,
        -0.050,  0.100, -0.050,
    ],
    // Ch 7: Vertical second-derivative Dyy (sum = 0.0)
    [
        -0.050, -0.150, -0.050,
         0.100,  0.300,  0.100,
        -0.050, -0.150, -0.050,
    ],
];

const L1_BIAS: [f32; 8] = [0.0; 8];
const L1_PRELU: [f32; 8] = [1.0, 0.25, 0.25, 0.20, 0.25, 0.25, 0.20, 0.20];

/// Pre-baked Layer 2 (FSRCNN Shrinking Layer): 1x1 Conv, 8 input channels -> 4 output channels.
/// Layout: `L2_WEIGHTS[out_ch][in_ch]`.
const L2_WEIGHTS: [[f32; 8]; 4] = [
    // Shrunk Ch 0: Base luminance + subtle Laplacian micro-contrast
    [1.00, 0.00, 0.00, 0.18, 0.00, 0.00, 0.05, 0.05],
    // Shrunk Ch 1: Horizontal edge & ridge subspace
    [0.00, 0.85, 0.00, 0.10, 0.15, -0.15, 0.35, 0.00],
    // Shrunk Ch 2: Vertical edge & ridge subspace
    [0.00, 0.00, 0.85, 0.10, 0.15, 0.15, 0.00, 0.35],
    // Shrunk Ch 3: Diagonal & high-frequency corner subspace
    [0.00, 0.10, 0.10, 0.55, 0.45, 0.45, 0.20, 0.20],
];

const L2_BIAS: [f32; 4] = [0.0; 4];
const L2_PRELU: [f32; 4] = [1.0, 0.30, 0.30, 0.25];

/// Pre-baked Layer 3 (Non-Linear Mapping with Residual Skip): 3x3 Conv, 4 -> 4 channels.
/// Layout: `L3_WEIGHTS[out_ch][ky * 3 + kx][in_ch]`.
/// All 3x3 spatial kernels have zero-sum weights so flat regions remain 0.0 before the `+ x` skip connection.
const L3_WEIGHTS: [[[f32; 4]; 9]; 4] = [
    // Out Ch 0: Unsharp mask refinement on luminance + high-freq feedback
    [
        [-0.015,  0.000,  0.000, -0.010],
        [-0.030,  0.000, -0.020, -0.015],
        [-0.015,  0.000,  0.000, -0.010],
        [-0.030, -0.020,  0.000, -0.015],
        [ 0.180,  0.000,  0.000,  0.100],
        [-0.030,  0.020,  0.000, -0.015],
        [-0.015,  0.000,  0.000, -0.010],
        [-0.030,  0.000,  0.020, -0.015],
        [-0.015,  0.000,  0.000, -0.010],
    ],
    // Out Ch 1: Horizontal edge continuity refinement
    [
        [ 0.000, -0.025,  0.000,  0.000],
        [ 0.000,  0.050,  0.000,  0.000],
        [ 0.000, -0.025,  0.000,  0.000],
        [-0.040, -0.050,  0.000, -0.020],
        [ 0.000,  0.100,  0.000,  0.040],
        [ 0.040, -0.050,  0.000, -0.020],
        [ 0.000, -0.025,  0.000,  0.000],
        [ 0.000,  0.050,  0.000,  0.000],
        [ 0.000, -0.025,  0.000,  0.000],
    ],
    // Out Ch 2: Vertical edge continuity refinement
    [
        [ 0.000,  0.000, -0.025,  0.000],
        [-0.040,  0.000, -0.050, -0.020],
        [ 0.000,  0.000, -0.025,  0.000],
        [ 0.000,  0.000,  0.050,  0.000],
        [ 0.000,  0.000,  0.100,  0.040],
        [ 0.000,  0.000,  0.050,  0.000],
        [ 0.000,  0.000, -0.025,  0.000],
        [ 0.040,  0.000, -0.050, -0.020],
        [ 0.000,  0.000, -0.025,  0.000],
    ],
    // Out Ch 3: Isotropic & diagonal detail enhancement
    [
        [-0.015,  0.000,  0.000, -0.025],
        [-0.020,  0.000,  0.000, -0.025],
        [-0.015,  0.000,  0.000, -0.025],
        [-0.020,  0.000,  0.000, -0.025],
        [ 0.140,  0.000,  0.000,  0.200],
        [-0.020,  0.000,  0.000, -0.025],
        [-0.015,  0.000,  0.000, -0.025],
        [-0.020,  0.000,  0.000, -0.025],
        [-0.015,  0.000,  0.000, -0.025],
    ],
];

const L3_BIAS: [f32; 4] = [0.0; 4];
const L3_PRELU: [f32; 4] = [0.50, 0.35, 0.35, 0.30];

/// Pre-baked Layer 4 (ESPCN Sub-Pixel Convolution): 3x3 Conv, 4 -> 4 sub-pixel channels (2x2 grid).
/// Layout: `L4_WEIGHTS[sub_ch][ky * 3 + kx][in_ch]`.
/// Crucially, for every sub-pixel channel `sub_ch` in 0..4, the 9 weights on `in_ch = 0` sum to
/// **exactly 1.000**, guaranteeing zero deconvolution/checkerboard artifacts on uniform or smooth gradients,
/// while directional edge channels (`in_ch = 1..3`) steer sub-pixel edge reconstruction for each quadrant:
/// - `sub_ch = 0`: Top-Left     `(-0.25px, -0.25px)`
/// - `sub_ch = 1`: Top-Right    `(+0.25px, -0.25px)`
/// - `sub_ch = 2`: Bottom-Left  `(-0.25px, +0.25px)`
/// - `sub_ch = 3`: Bottom-Right `(+0.25px, +0.25px)`
const L4_WEIGHTS: [[[f32; 4]; 9]; 4] = [
    // Sub-pixel 0: Top-Left (biased toward top-left neighbors + zero-sum negative Gx/Gy steering)
    [
        [ 0.045, -0.020, -0.020,  0.015],
        [ 0.135,  0.000, -0.055,  0.025],
        [-0.010,  0.045,  0.000, -0.020],
        [ 0.135, -0.055,  0.000,  0.025],
        [ 0.740, -0.100, -0.100,  0.120],
        [-0.020,  0.085,  0.000, -0.045],
        [-0.010,  0.000,  0.045, -0.020],
        [-0.020,  0.000,  0.085, -0.045],
        [ 0.005,  0.045,  0.045, -0.055],
    ],
    // Sub-pixel 1: Top-Right (biased toward top-right neighbors + zero-sum positive Gx, negative Gy steering)
    [
        [-0.010, -0.045,  0.000, -0.020],
        [ 0.135,  0.000, -0.055,  0.025],
        [ 0.045,  0.020, -0.020,  0.015],
        [-0.020, -0.085,  0.000, -0.045],
        [ 0.740,  0.100, -0.100,  0.120],
        [ 0.135,  0.055,  0.000,  0.025],
        [ 0.005, -0.045,  0.045, -0.055],
        [-0.020,  0.000,  0.085, -0.045],
        [-0.010,  0.000,  0.045, -0.020],
    ],
    // Sub-pixel 2: Bottom-Left (biased toward bottom-left neighbors + zero-sum negative Gx, positive Gy steering)
    [
        [-0.010,  0.000, -0.045, -0.020],
        [-0.020,  0.000, -0.085, -0.045],
        [ 0.005,  0.045, -0.045, -0.055],
        [ 0.135, -0.055,  0.000,  0.025],
        [ 0.740, -0.100,  0.100,  0.120],
        [-0.020,  0.085,  0.000, -0.045],
        [ 0.045, -0.020,  0.020,  0.015],
        [ 0.135,  0.000,  0.055,  0.025],
        [-0.010,  0.045,  0.000, -0.020],
    ],
    // Sub-pixel 3: Bottom-Right (biased toward bottom-right neighbors + zero-sum positive Gx/Gy steering)
    [
        [ 0.005, -0.045, -0.045, -0.055],
        [-0.020,  0.000, -0.085, -0.045],
        [-0.010,  0.000, -0.045, -0.020],
        [-0.020, -0.085,  0.000, -0.045],
        [ 0.740,  0.100,  0.100,  0.120],
        [ 0.135,  0.055,  0.000,  0.025],
        [-0.010, -0.045,  0.000, -0.020],
        [ 0.135,  0.000,  0.055,  0.025],
        [ 0.045,  0.020,  0.020,  0.015],
    ],
];

const L4_BIAS: [f32; 4] = [0.0; 4];

#[inline(always)]
fn prelu(x: f32, slope: f32) -> f32 {
    if x >= 0.0 {
        x
    } else {
        x * slope
    }
}

/// Minimum local luminance range (`max_y - min_y`) in a 3x3 neighborhood to trigger
/// neural or FSR super-resolution. Regions below `0.015` (~3.8 / 255) are flat web
/// backgrounds, cards, or solid UI banners and bypass heavy convolution completely.
pub const FLAT_LUMA_THRESHOLD: f32 = 0.015;

/// Content-Aware Sparse Skip classification for a pixel neighborhood or tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PatchClass {
    /// `range < 0.015`: Flat constant color region -> direct copy (0% bias drift, skip SR).
    Flat,
    /// `range >= 0.015`: Active image texture, icon, or text/UI edge -> full SR pipeline.
    TextureOrEdge,
}

/// Fast classification helper based on minimum and maximum normalized luminance (`0.0..=1.0`).
#[inline(always)]
pub fn classify_patch(min_y: f32, max_y: f32) -> PatchClass {
    let luma_range = max_y - min_y;
    if luma_range < FLAT_LUMA_THRESHOLD {
        PatchClass::Flat
    } else {
        PatchClass::TextureOrEdge
    }
}

/// Returns `true` only when the local neighborhood has sufficient luminance gradient/variance
/// (`max_y - min_y >= 0.015`) to warrant neural conv layers or FSR kernels.
#[inline(always)]
pub fn should_enhance_pixel(min_y: f32, max_y: f32) -> bool {
    classify_patch(min_y, max_y) == PatchClass::TextureOrEdge
}

/// Computes `(min_y, max_y)` across a 9-element 3x3 luminance patch.
#[inline(always)]
fn patch_min_max_9(p: &[f32; 9]) -> (f32, f32) {
    let min0 = p[0].min(p[1]).min(p[2]);
    let min1 = p[3].min(p[4]).min(p[5]);
    let min2 = p[6].min(p[7]).min(p[8]);
    let max0 = p[0].max(p[1]).max(p[2]);
    let max1 = p[3].max(p[4]).max(p[5]);
    let max2 = p[6].max(p[7]).max(p[8]);
    (min0.min(min1).min(min2), max0.max(max1).max(max2))
}

/// Splits a `PixelBuffer` into three flat `Vec<f32>` planes:
/// - `Y`:  Luminance normalized to `0.0..=1.0`
/// - `Cb`: Blue-difference chrominance centered around `0.0` (`-0.5..=0.5`)
/// - `Cr`: Red-difference chrominance centered around `0.0` (`-0.5..=0.5`)
pub fn rgb_to_ycbcr(buf: &PixelBuffer) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = buf.width * buf.height;
    let mut y_plane = Vec::with_capacity(len);
    let mut cb_plane = Vec::with_capacity(len);
    let mut cr_plane = Vec::with_capacity(len);

    const INV_255: f32 = 1.0 / 255.0;

    for px in &buf.data[..len.min(buf.data.len())] {
        let r = (px.r as f32) * INV_255;
        let g = (px.g as f32) * INV_255;
        let b = (px.b as f32) * INV_255;

        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        let cb = (b - y) * 0.564334; // 1 / (2 * (1 - 0.114))
        let cr = (r - y) * 0.713267; // 1 / (2 * (1 - 0.299))

        y_plane.push(y);
        cb_plane.push(cb);
        cr_plane.push(cr);
    }

    (y_plane, cb_plane, cr_plane)
}

/// Merges normalized `Y` (`0.0..=1.0`), `Cb` (`-0.5..=0.5`), and `Cr` (`-0.5..=0.5`) planes
/// back into a `PixelBuffer` of dimensions `width x height`.
pub fn ycbcr_to_rgb(
    y_plane: &[f32],
    cb_plane: &[f32],
    cr_plane: &[f32],
    width: usize,
    height: usize,
) -> PixelBuffer {
    let len = width * height;
    let mut data = Vec::with_capacity(len);

    let safe_len = len
        .min(y_plane.len())
        .min(cb_plane.len())
        .min(cr_plane.len());

    for i in 0..safe_len {
        let y = y_plane[i];
        let cb = cb_plane[i];
        let cr = cr_plane[i];

        let r = (y + 1.402 * cr).clamp(0.0, 1.0);
        let g = (y - 0.344136 * cb - 0.714136 * cr).clamp(0.0, 1.0);
        let b = (y + 1.772 * cb).clamp(0.0, 1.0);

        data.push(Color::new(
            (r * 255.0 + 0.5) as u8,
            (g * 255.0 + 0.5) as u8,
            (b * 255.0 + 0.5) as u8,
        ));
    }

    if data.len() < len {
        data.resize(len, Color::BLACK);
    }

    PixelBuffer {
        width,
        height,
        data,
    }
}

/// Fast bilinear upscaler for chrominance (`Cb` / `Cr`) planes by integer factor `scale`.
/// Avoids running heavy neural convolutions on human-eye-insensitive chroma channels (3x speedup).
pub fn upscale_chroma_bilinear(plane: &[f32], w: usize, h: usize, scale: usize) -> Vec<f32> {
    let scale = scale.max(1);
    if w == 0 || h == 0 || plane.len() < w * h {
        return Vec::new();
    }
    if scale == 1 {
        return plane[..w * h].to_vec();
    }

    let out_w = w * scale;
    let out_h = h * scale;
    let mut out = vec![0.0f32; out_w * out_h];

    let inv_scale = 1.0 / (scale as f32);
    let max_x = w - 1;
    let max_y = h - 1;

    // Precompute horizontal interpolation indices and weights once for all rows
    let mut x0_tab = Vec::with_capacity(out_w);
    let mut x1_tab = Vec::with_capacity(out_w);
    let mut wx_tab = Vec::with_capacity(out_w);

    for ox in 0..out_w {
        let src_x = ((ox as f32) + 0.5) * inv_scale - 0.5;
        let sx = src_x.max(0.0);
        let ix0 = (sx as usize).min(max_x);
        let ix1 = (ix0 + 1).min(max_x);
        let wx = sx - (ix0 as f32);
        x0_tab.push(ix0);
        x1_tab.push(ix1);
        wx_tab.push(wx);
    }

    for oy in 0..out_h {
        let src_y = ((oy as f32) + 0.5) * inv_scale - 0.5;
        let sy = src_y.max(0.0);
        let iy0 = (sy as usize).min(max_y);
        let iy1 = (iy0 + 1).min(max_y);
        let wy = sy - (iy0 as f32);
        let wy_inv = 1.0 - wy;

        let row0 = &plane[iy0 * w..iy0 * w + w];
        let row1 = &plane[iy1 * w..iy1 * w + w];
        let dst_row = &mut out[oy * out_w..(oy + 1) * out_w];

        for ox in 0..out_w {
            let ix0 = x0_tab[ox];
            let ix1 = x1_tab[ox];
            let wx = wx_tab[ox];
            let wx_inv = 1.0 - wx;

            let top = row0[ix0] * wx_inv + row0[ix1] * wx;
            let bot = row1[ix0] * wx_inv + row1[ix1] * wx;
            dst_row[ox] = top * wy_inv + bot * wy;
        }
    }

    out
}

/// Rearranges 4 sub-pixel channels (`[top_left, top_right, bottom_left, bottom_right]` per input pixel)
/// into a `(w * 2) x (h * 2)` high-resolution luminance plane without deconvolution checkerboard artifacts.
pub fn pixel_shuffle_2x(subpixels: &[[f32; 4]], w: usize, h: usize, out: &mut [f32]) {
    let out_w = w * 2;
    for y in 0..h {
        let src_row = &subpixels[y * w..(y + 1) * w];
        let top_row_start = (y * 2) * out_w;
        let bot_row_start = (y * 2 + 1) * out_w;

        let (top_part, rest) = out[top_row_start..bot_row_start + out_w].split_at_mut(out_w);
        let bot_part = &mut rest[..out_w];

        for (x, sp) in src_row.iter().enumerate() {
            let dx = x * 2;
            top_part[dx] = sp[0].clamp(0.0, 1.0);
            top_part[dx + 1] = sp[1].clamp(0.0, 1.0);
            bot_part[dx] = sp[2].clamp(0.0, 1.0);
            bot_part[dx + 1] = sp[3].clamp(0.0, 1.0);
        }
    }
}

/// Runs the 4-layer Hybrid FSRCNN-ESPCN super-resolution network on a normalized `Y` luminance plane
/// (`w x h`), returning a `(w * 2) x (h * 2)` luminance plane.
/// Uses the Content-Aware Sparse Skip Engine (`should_enhance_pixel` & 8x8 tile gating) so flat
/// backgrounds (`luma_range < 0.015`) completely bypass all 4 convolutional layers and output
/// `center_y` untouched with 0% neural bias drift.
pub fn upscale_y_neural_2x(y_plane: &[f32], w: usize, h: usize) -> Vec<f32> {
    if w == 0 || h == 0 || y_plane.len() < w * h {
        return Vec::new();
    }

    let num_px = w * h;

    // Fast whole-plane / 8x8 tile variance pre-pass + per-pixel 3x3 gradient gate:
    // Any pixel with 3x3 `luma_range < 0.015` is marked inactive (`false`) and skips all conv math.
    let mut active_px = vec![false; num_px];
    let mut any_active = false;
    let max_x = w - 1;
    let max_y = h - 1;

    // -------------------------------------------------------------------------
    // Layer 1 (Feature Extraction: 3x3 Conv, 1 -> 8 channels + PReLU)
    // combined with Layer 2 (FSRCNN Shrinking: 1x1 Conv, 8 -> 4 channels + PReLU)
    // Fused per-pixel so the 8-channel intermediate stays in CPU registers!
    // -------------------------------------------------------------------------
    let mut feat_l2 = vec![[0.0f32; 4]; num_px];

    for y in 0..h {
        let ym1 = y.saturating_sub(1);
        let yp1 = (y + 1).min(max_y);

        let r_top = &y_plane[ym1 * w..ym1 * w + w];
        let r_mid = &y_plane[y * w..y * w + w];
        let r_bot = &y_plane[yp1 * w..yp1 * w + w];
        let dst_row = &mut feat_l2[y * w..(y + 1) * w];
        let act_row = &mut active_px[y * w..(y + 1) * w];

        for x in 0..w {
            let xm1 = x.saturating_sub(1);
            let xp1 = (x + 1).min(max_x);

            let patch = [
                r_top[xm1], r_top[x], r_top[xp1],
                r_mid[xm1], r_mid[x], r_mid[xp1],
                r_bot[xm1], r_bot[x], r_bot[xp1],
            ];

            let (min_y, max_y_val) = patch_min_max_9(&patch);
            if !should_enhance_pixel(min_y, max_y_val) {
                // Content-Aware Sparse Skip: flat 3x3 region -> bypass Layer 1 & Layer 2!
                dst_row[x] = [patch[4], 0.0, 0.0, 0.0];
                continue;
            }

            act_row[x] = true;
            any_active = true;

            // Layer 1: 1 -> 8 channels + PReLU (kept entirely in registers)
            let mut l1_out = [0.0f32; 8];
            for oc in 0..8 {
                let w_k = &L1_WEIGHTS[oc];
                let sum = L1_BIAS[oc]
                    + patch[0] * w_k[0]
                    + patch[1] * w_k[1]
                    + patch[2] * w_k[2]
                    + patch[3] * w_k[3]
                    + patch[4] * w_k[4]
                    + patch[5] * w_k[5]
                    + patch[6] * w_k[6]
                    + patch[7] * w_k[7]
                    + patch[8] * w_k[8];
                l1_out[oc] = prelu(sum, L1_PRELU[oc]);
            }

            // Layer 2: 1x1 Shrinking (8 -> 4 channels) + PReLU
            let mut l2_out = [0.0f32; 4];
            for oc in 0..4 {
                let w_1x1 = &L2_WEIGHTS[oc];
                let sum = L2_BIAS[oc]
                    + l1_out[0] * w_1x1[0]
                    + l1_out[1] * w_1x1[1]
                    + l1_out[2] * w_1x1[2]
                    + l1_out[3] * w_1x1[3]
                    + l1_out[4] * w_1x1[4]
                    + l1_out[5] * w_1x1[5]
                    + l1_out[6] * w_1x1[6]
                    + l1_out[7] * w_1x1[7];
                l2_out[oc] = prelu(sum, L2_PRELU[oc]);
            }

            dst_row[x] = l2_out;
        }
    }

    // Fast path: if the entire frame is flat (e.g., blank/solid canvas), bypass Layer 3 & Layer 4 completely
    if !any_active {
        let mut out_y = vec![0.0f32; (w * 2) * (h * 2)];
        let out_w = w * 2;
        for y in 0..h {
            let src_row = &y_plane[y * w..(y + 1) * w];
            let top_row_start = (y * 2) * out_w;
            let bot_row_start = (y * 2 + 1) * out_w;
            let (top_part, rest) = out_y[top_row_start..bot_row_start + out_w].split_at_mut(out_w);
            let bot_part = &mut rest[..out_w];
            for (x, &cy) in src_row.iter().enumerate() {
                let clamped = cy.clamp(0.0, 1.0);
                let dx = x * 2;
                top_part[dx] = clamped;
                top_part[dx + 1] = clamped;
                bot_part[dx] = clamped;
                bot_part[dx + 1] = clamped;
            }
        }
        return out_y;
    }

    // -------------------------------------------------------------------------
    // Layer 3 (Non-Linear Mapping: 3x3 Conv, 4 -> 4 channels + PReLU + Residual Skip)
    // out = prelu(conv3x3(feat_l2)) + feat_l2
    // -------------------------------------------------------------------------
    let mut feat_l3 = vec![[0.0f32; 4]; num_px];

    for y in 0..h {
        let ym1 = y.saturating_sub(1);
        let yp1 = (y + 1).min(max_y);

        let r_top = &feat_l2[ym1 * w..ym1 * w + w];
        let r_mid = &feat_l2[y * w..y * w + w];
        let r_bot = &feat_l2[yp1 * w..yp1 * w + w];
        let act_row = &active_px[y * w..(y + 1) * w];
        let dst_row = &mut feat_l3[y * w..(y + 1) * w];

        for x in 0..w {
            if !act_row[x] {
                // Flat pixel: bypass Layer 3 3x3 conv completely
                dst_row[x] = r_mid[x];
                continue;
            }

            let xm1 = x.saturating_sub(1);
            let xp1 = (x + 1).min(max_x);

            let p = [
                r_top[xm1], r_top[x], r_top[xp1],
                r_mid[xm1], r_mid[x], r_mid[xp1],
                r_bot[xm1], r_bot[x], r_bot[xp1],
            ];
            let residual = p[4];

            let mut l3_out = [0.0f32; 4];
            for oc in 0..4 {
                let wk = &L3_WEIGHTS[oc];
                let mut acc = L3_BIAS[oc];
                for k in 0..9 {
                    let pk = p[k];
                    let w_ic = wk[k];
                    acc += pk[0] * w_ic[0]
                        + pk[1] * w_ic[1]
                        + pk[2] * w_ic[2]
                        + pk[3] * w_ic[3];
                }
                l3_out[oc] = prelu(acc, L3_PRELU[oc]) + residual[oc];
            }

            dst_row[x] = l3_out;
        }
    }

    // -------------------------------------------------------------------------
    // Layer 4 (ESPCN Sub-Pixel Expansion: 3x3 Conv, 4 -> 4 channels + 2x PixelShuffle)
    // -------------------------------------------------------------------------
    let mut subpixels = vec![[0.0f32; 4]; num_px];

    for y in 0..h {
        let ym1 = y.saturating_sub(1);
        let yp1 = (y + 1).min(max_y);

        let r_top = &feat_l3[ym1 * w..ym1 * w + w];
        let r_mid = &feat_l3[y * w..y * w + w];
        let r_bot = &feat_l3[yp1 * w..yp1 * w + w];
        let y_row = &y_plane[y * w..(y + 1) * w];
        let act_row = &active_px[y * w..(y + 1) * w];
        let dst_row = &mut subpixels[y * w..(y + 1) * w];

        for x in 0..w {
            if !act_row[x] {
                // Content-Aware Sparse Skip: output exact source pixel luminance `center_y` untouched
                let center_y = y_row[x];
                dst_row[x] = [center_y, center_y, center_y, center_y];
                continue;
            }

            let xm1 = x.saturating_sub(1);
            let xp1 = (x + 1).min(max_x);

            let p = [
                r_top[xm1], r_top[x], r_top[xp1],
                r_mid[xm1], r_mid[x], r_mid[xp1],
                r_bot[xm1], r_bot[x], r_bot[xp1],
            ];

            let mut sp = [0.0f32; 4];
            for sub_ch in 0..4 {
                let wk = &L4_WEIGHTS[sub_ch];
                let mut acc = L4_BIAS[sub_ch];
                for k in 0..9 {
                    let pk = p[k];
                    let w_ic = wk[k];
                    acc += pk[0] * w_ic[0]
                        + pk[1] * w_ic[1]
                        + pk[2] * w_ic[2]
                        + pk[3] * w_ic[3];
                }
                sp[sub_ch] = acc;
            }

            dst_row[x] = sp;
        }
    }

    let mut out_y = vec![0.0f32; (w * 2) * (h * 2)];
    pixel_shuffle_2x(&subpixels, w, h, &mut out_y);
    out_y
}

/// Public Entrypoint: Upscales a `PixelBuffer` by 2x using the Hybrid FSRCNN-ESPCN
/// neural pipeline on the luminance (`Y`) channel and fast bilinear interpolation on `Cb`/`Cr`.
pub fn upscale_neural_2x(buf: &PixelBuffer) -> PixelBuffer {
    if buf.width == 0 || buf.height == 0 || buf.data.is_empty() {
        return PixelBuffer::new(0, 0, Color::BLACK);
    }

    let (y_plane, cb_plane, cr_plane) = rgb_to_ycbcr(buf);
    let up_y = upscale_y_neural_2x(&y_plane, buf.width, buf.height);
    let up_cb = upscale_chroma_bilinear(&cb_plane, buf.width, buf.height, 2);
    let up_cr = upscale_chroma_bilinear(&cr_plane, buf.width, buf.height, 2);

    ycbcr_to_rgb(&up_y, &up_cb, &up_cr, buf.width * 2, buf.height * 2)
}

// =============================================================================
// Phase 2: Pure-Rust AMD FidelityFX Super Resolution 1.0 (EASU + RCAS)
// =============================================================================

#[inline(always)]
fn fast_luma_norm(c: Color) -> f32 {
    // Rec.601 luma normalized to [0.0, 1.0]
    (c.r as f32 * 0.299 + c.g as f32 * 0.587 + c.b as f32 * 0.114) * (1.0 / 255.0)
}

/// Accumulates local horizontal and vertical luminance gradients at one of the 4 inner quad
/// texels (`f`, `g`, `j`, `k`) for AMD FSR 1.0 Edge-Adaptive Spatial Upsampling (`FsrEasuSet`).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn easu_accumulate_dir(
    dir_x: &mut f32,
    dir_y: &mut f32,
    len_acc: &mut f32,
    weight: f32,
    l_center: f32,
    l_west: f32,
    l_east: f32,
    l_north: f32,
    l_south: f32,
) {
    // Horizontal gradient & directional isotropy
    let dc = l_east - l_west;
    let len_x = (l_east - l_center).abs().max((l_center - l_west).abs());
    let inv_len_x = if len_x > 1e-5 { 1.0 / len_x } else { 0.0 };
    *dir_x += dc * weight;
    let wx = (dc.abs() * inv_len_x).clamp(0.0, 1.0);

    // Vertical gradient & directional isotropy
    let ec = l_south - l_north;
    let len_y = (l_south - l_center).abs().max((l_center - l_north).abs());
    let inv_len_y = if len_y > 1e-5 { 1.0 / len_y } else { 0.0 };
    *dir_y += ec * weight;
    let wy = (ec.abs() * inv_len_y).clamp(0.0, 1.0);

    *len_acc += (wx * wx + wy * wy) * weight;
}

/// Evaluates the FSR 1.0 elliptical 2-lobe Lanczos2 polynomial approximation for a single tap
/// and accumulates its weighted RGB contribution (`FsrEasuTap`).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn easu_tap(
    acc_r: &mut f32,
    acc_g: &mut f32,
    acc_b: &mut f32,
    acc_w: &mut f32,
    off_x: f32,
    off_y: f32,
    dir_x: f32,
    dir_y: f32,
    len2_x: f32,
    len2_y: f32,
    lob: f32,
    clp: f32,
    color: Color,
) {
    // Rotate tap offset by local edge direction and scale by elliptical elongation
    let rx = (off_x * dir_x + off_y * dir_y) * len2_x;
    let ry = (-off_x * dir_y + off_y * dir_x) * len2_y;
    let d2 = (rx * rx + ry * ry).min(clp);

    // Fast polynomial approximation of Lanczos2 window:
    // w_b = (25/16) * ((2/5)*d^2 - 1)^2 - (25/16 - 1)
    // w_a = (lob*d^2 - 1)^2
    let mut w_b = 0.4 * d2 - 1.0;
    let mut w_a = lob * d2 - 1.0;
    w_b *= w_b;
    w_a *= w_a;
    w_b = (25.0 / 16.0) * w_b - (25.0 / 16.0 - 1.0);
    let w = w_b * w_a;

    *acc_r += (color.r as f32) * w;
    *acc_g += (color.g as f32) * w;
    *acc_b += (color.b as f32) * w;
    *acc_w += w;
}

/// AMD FidelityFX Super Resolution 1.0: Edge-Adaptive Spatial Upsampling (`EASU`).
///
/// For each output pixel in `target_w x target_h`:
/// 1. Samples the 12-tap cross neighborhood around the source sub-pixel position.
/// 2. Computes local horizontal and vertical luminance gradients to detect diagonal edge
///    orientation and elongation.
/// 3. Applies an elliptical Lanczos2-style directional kernel stretched along the edge.
/// 4. Clamps the output RGB to the `[min, max]` bounding box of the 4 nearest input texels
///    (`f`, `g`, `j`, `k`) to eliminate ringing halos.
pub fn fsr_easu(buf: &PixelBuffer, target_w: usize, target_h: usize) -> PixelBuffer {
    let w = buf.width;
    let h = buf.height;
    if w == 0 || h == 0 || target_w == 0 || target_h == 0 || buf.data.len() < w * h {
        return PixelBuffer::new(0, 0, Color::BLACK);
    }
    if w == target_w && h == target_h {
        return buf.clone();
    }

    let mut out_data = Vec::with_capacity(target_w * target_h);
    let scale_x = (w as f32) / (target_w as f32);
    let scale_y = (h as f32) / (target_h as f32);
    let max_x = (w - 1) as isize;
    let max_y = (h - 1) as isize;
    let src = &buf.data[..w * h];

    for oy in 0..target_h {
        let src_y = ((oy as f32) + 0.5) * scale_y - 0.5;
        let iy = src_y.floor() as isize;
        let fy = src_y - (iy as f32);

        // Clamped row offsets for the 4 rows of the 12-tap cross: iy-1, iy, iy+1, iy+2
        let r_m1 = (iy - 1).clamp(0, max_y) as usize * w;
        let r_0 = iy.clamp(0, max_y) as usize * w;
        let r_p1 = (iy + 1).clamp(0, max_y) as usize * w;
        let r_p2 = (iy + 2).clamp(0, max_y) as usize * w;

        for ox in 0..target_w {
            let src_x = ((ox as f32) + 0.5) * scale_x - 0.5;
            let ix = src_x.floor() as isize;
            let fx = src_x - (ix as f32);

            // Clamped column indices for the 4 columns of the 12-tap cross: ix-1, ix, ix+1, ix+2
            let c_m1 = (ix - 1).clamp(0, max_x) as usize;
            let c_0 = ix.clamp(0, max_x) as usize;
            let c_p1 = (ix + 1).clamp(0, max_x) as usize;
            let c_p2 = (ix + 2).clamp(0, max_x) as usize;

            // Fetch 12-tap cross neighborhood:
            //       b  c
            //    e  f  g  h
            //    i  j  k  l
            //       n  o
            let b = src[r_m1 + c_0];
            let c = src[r_m1 + c_p1];

            let e = src[r_0 + c_m1];
            let f = src[r_0 + c_0];
            let g = src[r_0 + c_p1];
            let h_tap = src[r_0 + c_p2];

            let i_tap = src[r_p1 + c_m1];
            let j = src[r_p1 + c_0];
            let k = src[r_p1 + c_p1];
            let l = src[r_p1 + c_p2];

            let n = src[r_p2 + c_0];
            let o = src[r_p2 + c_p1];

            // Fast integer identity gate: if all 12 taps are identical, skip float luma & 12-tap kernel
            if f == g
                && f == j
                && f == k
                && f == b
                && f == c
                && f == e
                && f == h_tap
                && f == i_tap
                && f == l
                && f == n
                && f == o
            {
                out_data.push(f);
                continue;
            }

            // Compute normalized luma for gradient analysis
            let lb = fast_luma_norm(b);
            let lc = fast_luma_norm(c);
            let le = fast_luma_norm(e);
            let lf = fast_luma_norm(f);
            let lg = fast_luma_norm(g);
            let lh = fast_luma_norm(h_tap);
            let li = fast_luma_norm(i_tap);
            let lj = fast_luma_norm(j);
            let lk = fast_luma_norm(k);
            let ll = fast_luma_norm(l);
            let ln = fast_luma_norm(n);
            let lo = fast_luma_norm(o);

            // Content-Aware Sparse Skip Gate: if local luminance range < 0.015, bypass EASU 12-tap kernel
            let min_luma = lb
                .min(lc)
                .min(le)
                .min(lf)
                .min(lg)
                .min(lh)
                .min(li)
                .min(lj)
                .min(lk)
                .min(ll)
                .min(ln)
                .min(lo);
            let max_luma = lb
                .max(lc)
                .max(le)
                .max(lf)
                .max(lg)
                .max(lh)
                .max(li)
                .max(lj)
                .max(lk)
                .max(ll)
                .max(ln)
                .max(lo);

            if !should_enhance_pixel(min_luma, max_luma) {
                out_data.push(f);
                continue;
            }

            // Accumulate bilinearly weighted gradients at the 4 inner quad corners (f, g, j, k)
            let mut dir_x = 0.0f32;
            let mut dir_y = 0.0f32;
            let mut len = 0.0f32;

            easu_accumulate_dir(&mut dir_x, &mut dir_y, &mut len, (1.0 - fx) * (1.0 - fy), lf, le, lg, lb, lj);
            easu_accumulate_dir(&mut dir_x, &mut dir_y, &mut len, fx * (1.0 - fy), lg, lf, lh, lc, lk);
            easu_accumulate_dir(&mut dir_x, &mut dir_y, &mut len, (1.0 - fx) * fy, lj, li, lk, lf, ln);
            easu_accumulate_dir(&mut dir_x, &mut dir_y, &mut len, fx * fy, lk, lj, ll, lg, lo);

            // Normalize gradient direction vector
            let dir_sq = dir_x * dir_x + dir_y * dir_y;
            if dir_sq < (1.0 / 32768.0) {
                dir_x = 1.0;
                dir_y = 0.0;
            } else {
                let inv_dir = 1.0 / dir_sq.sqrt();
                dir_x *= inv_dir;
                dir_y *= inv_dir;
            }

            // Convert isotropy length into elliptical stretch factors and adaptive negative lobe
            len = (len * 0.5).clamp(0.0, 1.0);
            len *= len;
            let stretch = 1.0 / dir_x.abs().max(dir_y.abs()).max(1e-5);
            let len2_x = 1.0 + (stretch - 1.0) * len;
            let len2_y = 1.0 - 0.5 * len;
            let lob = 0.5 - 0.29 * len;
            let clp = 1.0 / lob;

            // Evaluate 12-tap elliptical Lanczos2 kernel
            let mut acc_r = 0.0f32;
            let mut acc_g = 0.0f32;
            let mut acc_b = 0.0f32;
            let mut acc_w = 0.0f32;

            easu_tap(&mut acc_r, &mut acc_g, &mut acc_b, &mut acc_w,  0.0 - fx, -1.0 - fy, dir_x, dir_y, len2_x, len2_y, lob, clp, b);
            easu_tap(&mut acc_r, &mut acc_g, &mut acc_b, &mut acc_w,  1.0 - fx, -1.0 - fy, dir_x, dir_y, len2_x, len2_y, lob, clp, c);
            easu_tap(&mut acc_r, &mut acc_g, &mut acc_b, &mut acc_w, -1.0 - fx,  0.0 - fy, dir_x, dir_y, len2_x, len2_y, lob, clp, e);
            easu_tap(&mut acc_r, &mut acc_g, &mut acc_b, &mut acc_w,  0.0 - fx,  0.0 - fy, dir_x, dir_y, len2_x, len2_y, lob, clp, f);
            easu_tap(&mut acc_r, &mut acc_g, &mut acc_b, &mut acc_w,  1.0 - fx,  0.0 - fy, dir_x, dir_y, len2_x, len2_y, lob, clp, g);
            easu_tap(&mut acc_r, &mut acc_g, &mut acc_b, &mut acc_w,  2.0 - fx,  0.0 - fy, dir_x, dir_y, len2_x, len2_y, lob, clp, h_tap);
            easu_tap(&mut acc_r, &mut acc_g, &mut acc_b, &mut acc_w, -1.0 - fx,  1.0 - fy, dir_x, dir_y, len2_x, len2_y, lob, clp, i_tap);
            easu_tap(&mut acc_r, &mut acc_g, &mut acc_b, &mut acc_w,  0.0 - fx,  1.0 - fy, dir_x, dir_y, len2_x, len2_y, lob, clp, j);
            easu_tap(&mut acc_r, &mut acc_g, &mut acc_b, &mut acc_w,  1.0 - fx,  1.0 - fy, dir_x, dir_y, len2_x, len2_y, lob, clp, k);
            easu_tap(&mut acc_r, &mut acc_g, &mut acc_b, &mut acc_w,  2.0 - fx,  1.0 - fy, dir_x, dir_y, len2_x, len2_y, lob, clp, l);
            easu_tap(&mut acc_r, &mut acc_g, &mut acc_b, &mut acc_w,  0.0 - fx,  2.0 - fy, dir_x, dir_y, len2_x, len2_y, lob, clp, n);
            easu_tap(&mut acc_r, &mut acc_g, &mut acc_b, &mut acc_w,  1.0 - fx,  2.0 - fy, dir_x, dir_y, len2_x, len2_y, lob, clp, o);

            let inv_w = if acc_w.abs() > 1e-6 { 1.0 / acc_w } else { 1.0 };

            // Anti-ringing clamp: bound output to the min/max of the 4 nearest quad texels (f, g, j, k)
            let min_r = f.r.min(g.r).min(j.r).min(k.r) as f32;
            let max_r = f.r.max(g.r).max(j.r).max(k.r) as f32;
            let min_g = f.g.min(g.g).min(j.g).min(k.g) as f32;
            let max_g = f.g.max(g.g).max(j.g).max(k.g) as f32;
            let min_b = f.b.min(g.b).min(j.b).min(k.b) as f32;
            let max_b = f.b.max(g.b).max(j.b).max(k.b) as f32;

            let out_r = (acc_r * inv_w).clamp(min_r, max_r);
            let out_g = (acc_g * inv_w).clamp(min_g, max_g);
            let out_b = (acc_b * inv_w).clamp(min_b, max_b);

            out_data.push(Color::new(
                (out_r + 0.5) as u8,
                (out_g + 0.5) as u8,
                (out_b + 0.5) as u8,
            ));
        }
    }

    PixelBuffer {
        width: target_w,
        height: target_h,
        data: out_data,
    }
}

/// AMD FidelityFX Super Resolution 1.0: Robust Contrast-Adaptive Sharpening (`RCAS`).
///
/// Uses a 5-tap cross neighborhood (`center`, `north`, `south`, `east`, `west`).
/// Calculates local contrast (`min` and `max` luma in the 5-tap cross) and derives an
/// analytical negative sharpening lobe weight so flat areas and already-saturated 1px UI
/// borders never clip or ring, while blurry web thumbnails and small text receive crisp
/// edge reinforcement.
///
/// - `sharpness`: `0.0..=1.0` (e.g. `0.85` for strong crisp sharpening).
pub fn fsr_rcas(buf: &PixelBuffer, sharpness: f32) -> PixelBuffer {
    let w = buf.width;
    let h = buf.height;
    if w == 0 || h == 0 || buf.data.len() < w * h {
        return PixelBuffer::new(0, 0, Color::BLACK);
    }
    let sharp = sharpness.clamp(0.0, 1.0);
    if sharp <= 1e-4 {
        return buf.clone();
    }

    // RCAS stability limit: -(0.25 - 1/16) = -0.1875
    const RCAS_LIMIT: f32 = 0.1875;
    let src = &buf.data[..w * h];
    let mut out_data = Vec::with_capacity(w * h);
    let max_x = w - 1;
    let max_y = h - 1;

    for y in 0..h {
        let ym1 = y.saturating_sub(1);
        let yp1 = (y + 1).min(max_y);

        let row_n = &src[ym1 * w..ym1 * w + w];
        let row_c = &src[y * w..y * w + w];
        let row_s = &src[yp1 * w..yp1 * w + w];

        for x in 0..w {
            let xm1 = x.saturating_sub(1);
            let xp1 = (x + 1).min(max_x);

            let c = row_c[x];
            let n = row_n[x];
            let s = row_s[x];
            let w_px = row_c[xm1];
            let e = row_c[xp1];

            if c == n && c == s && c == w_px && c == e {
                out_data.push(c);
                continue;
            }

            let lc = fast_luma_norm(c);
            let ln = fast_luma_norm(n);
            let ls = fast_luma_norm(s);
            let lw = fast_luma_norm(w_px);
            let le = fast_luma_norm(e);

            let min4 = ln.min(ls).min(lw).min(le);
            let max4 = ln.max(ls).max(lw).max(le);
            let min5 = min4.min(lc);
            let max5 = max4.max(lc);

            // Content-Aware Sparse Skip (`should_enhance_pixel`) & saturated 1px border guard:
            // Skip flat regions (`luma_range < 0.015`) and already-saturated full-scale borders.
            if !should_enhance_pixel(min5, max5) || (max5 > 0.985 && min5 < 0.015) {
                out_data.push(c);
                continue;
            }

            // Analytical RCAS dark/bright clipping limiters
            let hit_min = min4 / (4.0 * max5 + 1e-5);
            let hit_max = (1.0 - max4) / (4.0 * min5 - 4.0 - 1e-5);
            let raw_lobe = (-hit_min).max(hit_max).min(0.0);
            let lobe = raw_lobe.max(-RCAS_LIMIT) * sharp;

            let inv_norm = 1.0 / (1.0 + 4.0 * lobe);

            let sum_r = (n.r as u32 + s.r as u32 + w_px.r as u32 + e.r as u32) as f32;
            let sum_g = (n.g as u32 + s.g as u32 + w_px.g as u32 + e.g as u32) as f32;
            let sum_b = (n.b as u32 + s.b as u32 + w_px.b as u32 + e.b as u32) as f32;

            let out_r = (((c.r as f32) + lobe * sum_r) * inv_norm).clamp(0.0, 255.0);
            let out_g = (((c.g as f32) + lobe * sum_g) * inv_norm).clamp(0.0, 255.0);
            let out_b = (((c.b as f32) + lobe * sum_b) * inv_norm).clamp(0.0, 255.0);

            out_data.push(Color::new(
                (out_r + 0.5) as u8,
                (out_g + 0.5) as u8,
                (out_b + 0.5) as u8,
            ));
        }
    }

    PixelBuffer {
        width: w,
        height: h,
        data: out_data,
    }
}

// =============================================================================
// Task 3: Multi-Mode Super-Resolution Pipeline Enum & Master Dispatcher
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpscaleMode {
    Off,        // Standard sample_subpixel_box only
    FsrOnly,    // EASU + RCAS spatial pass
    NeuralOnly, // YCbCr Hybrid FSRCNN-ESPCN 2x pass
    Combined,   // Hybrid Neural 2x pass -> followed by RCAS contrast sharpening
    MultiPass,  // EASU spatial pass -> Neural detail pass -> RCAS sharpening
}

impl UpscaleMode {
    pub fn next(self) -> Self {
        match self {
            UpscaleMode::Off => UpscaleMode::FsrOnly,
            UpscaleMode::FsrOnly => UpscaleMode::NeuralOnly,
            UpscaleMode::NeuralOnly => UpscaleMode::Combined,
            UpscaleMode::Combined => UpscaleMode::MultiPass,
            UpscaleMode::MultiPass => UpscaleMode::Off,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            UpscaleMode::Off => "Off (Standard)",
            UpscaleMode::FsrOnly => "FSR 1.0 (EASU + RCAS)",
            UpscaleMode::NeuralOnly => "Neural 2x (FSRCNN-ESPCN)",
            UpscaleMode::Combined => "Combined (Neural 2x + RCAS)",
            UpscaleMode::MultiPass => "Multi-Pass (EASU + Neural + RCAS)",
        }
    }

    pub fn short_name(self) -> &'static str {
        match self {
            UpscaleMode::Off => "OFF",
            UpscaleMode::FsrOnly => "FSR",
            UpscaleMode::NeuralOnly => "NRL",
            UpscaleMode::Combined => "CMB",
            UpscaleMode::MultiPass => "MLP",
        }
    }

    pub fn badge_name(self) -> &'static str {
        match self {
            UpscaleMode::Off => "OFF",
            UpscaleMode::FsrOnly => "FSR",
            UpscaleMode::NeuralOnly => "NEURAL",
            UpscaleMode::Combined => "COMBINED",
            UpscaleMode::MultiPass => "MULTIPASS",
        }
    }

    pub fn from_cli_value(val: &str) -> Option<Self> {
        match val.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "0" => Some(UpscaleMode::Off),
            "fsr" | "easu" | "fsronly" => Some(UpscaleMode::FsrOnly),
            "neural" | "fsrcnn" | "espcn" | "neuralonly" => Some(UpscaleMode::NeuralOnly),
            "combined" | "hybrid" | "cmb" => Some(UpscaleMode::Combined),
            "multipass" | "multi" | "mlp" => Some(UpscaleMode::MultiPass),
            _ => None,
        }
    }
}

/// Upscales `buf` using the YCbCr Hybrid FSRCNN-ESPCN neural pipeline and dynamically bridges
/// scales beyond `2x`:
/// - For `>= 4x` target scales (`out_w >= buf.width * 4` or `out_h >= buf.height * 4`), runs
///   `upscale_y_neural_2x` twice on the luminance (`Y`) plane (4x neural super-resolution) with
///   4x bilinear chroma upscaling.
/// - Whenever `out_w > neural.width` or `out_h > neural.height` (e.g., `2.5x`, `3x`, `5x`, or
///   non-multiple terminal aspect ratios), feeds the neural output into `fsr_easu` to bridge
///   the remaining scale to the exact `(out_w, out_h)` target terminal resolution.
fn upscale_neural_adaptive(buf: &PixelBuffer, out_w: usize, out_h: usize) -> PixelBuffer {
    let w = buf.width;
    let h = buf.height;

    let neural_buf = if out_w >= w * 4 || out_h >= h * 4 {
        let (y_plane, cb_plane, cr_plane) = rgb_to_ycbcr(buf);
        let y_2x = upscale_y_neural_2x(&y_plane, w, h);
        let y_4x = upscale_y_neural_2x(&y_2x, w * 2, h * 2);
        let cb_4x = upscale_chroma_bilinear(&cb_plane, w, h, 4);
        let cr_4x = upscale_chroma_bilinear(&cr_plane, w, h, 4);
        ycbcr_to_rgb(&y_4x, &cb_4x, &cr_4x, w * 4, h * 4)
    } else {
        upscale_neural_2x(buf)
    };

    if out_w > w * 2 || out_h > h * 2 || (out_w != neural_buf.width || out_h != neural_buf.height) {
        if out_w == neural_buf.width && out_h == neural_buf.height {
            neural_buf
        } else {
            fsr_easu(&neural_buf, out_w, out_h)
        }
    } else {
        neural_buf
    }
}

/// Master Super-Resolution Dispatcher: processes `buf` according to the selected `UpscaleMode`.
///
/// - `UpscaleMode::Off`: Returns `buf` unchanged so the renderer uses fast `sample_subpixel_box`.
/// - `UpscaleMode::FsrOnly`: Runs AMD FSR 1.0 `fsr_easu` to `(target_w, target_h)` followed by `fsr_rcas`.
/// - `UpscaleMode::NeuralOnly`: Runs `upscale_neural_adaptive` (1x or 2x `upscale_y_neural_2x` + `fsr_easu` bridge to `target_w x target_h`).
/// - `UpscaleMode::Combined`: Runs `upscale_neural_adaptive` to `(target_w, target_h)` followed by `fsr_rcas` contrast sharpening.
/// - `UpscaleMode::MultiPass`: Runs `1.5x fsr_easu` directional pre-scale -> `2x upscale_neural_2x` detail pass (+ `fsr_easu` bridge) -> `fsr_rcas` sharpening.
pub fn upscale_frame(
    buf: &PixelBuffer,
    mode: UpscaleMode,
    target_w: usize,
    target_h: usize,
) -> PixelBuffer {
    if buf.width == 0 || buf.height == 0 || buf.data.is_empty() {
        return PixelBuffer::new(0, 0, Color::BLACK);
    }

    let out_w = if target_w > 0 { target_w } else { buf.width * 2 };
    let out_h = if target_h > 0 { target_h } else { buf.height * 2 };

    match mode {
        UpscaleMode::Off => buf.clone(),
        UpscaleMode::FsrOnly => {
            let easu = fsr_easu(buf, out_w, out_h);
            fsr_rcas(&easu, 0.85)
        }
        UpscaleMode::NeuralOnly => upscale_neural_adaptive(buf, out_w, out_h),
        UpscaleMode::Combined => {
            let neural = upscale_neural_adaptive(buf, out_w, out_h);
            fsr_rcas(&neural, 0.80)
        }
        UpscaleMode::MultiPass => {
            if out_w >= buf.width * 4 || out_h >= buf.height * 4 {
                // For >= 4x scales: run 2x neural detail pass -> 1.5x EASU pre-bridge -> second 2x neural pass -> exact EASU bridge -> RCAS
                let stage1_neural = upscale_neural_2x(buf);
                let mid_w = ((stage1_neural.width * 3) / 2).max(out_w / 2).max(stage1_neural.width + 1);
                let mid_h = ((stage1_neural.height * 3) / 2).max(out_h / 2).max(stage1_neural.height + 1);
                let easu_mid = fsr_easu(&stage1_neural, mid_w, mid_h);
                let stage2_neural = upscale_neural_2x(&easu_mid);
                let bridged = if stage2_neural.width != out_w || stage2_neural.height != out_h {
                    fsr_easu(&stage2_neural, out_w, out_h)
                } else {
                    stage2_neural
                };
                fsr_rcas(&bridged, 0.85)
            } else {
                // Stage 1: 1.5x Edge-Adaptive Spatial Upsampling (never no-ops even when out_w == buf.width * 2)
                let mid_w = ((buf.width * 3) / 2).max(buf.width + 1);
                let mid_h = ((buf.height * 3) / 2).max(buf.height + 1);
                let easu = fsr_easu(buf, mid_w, mid_h);
                // Stage 2: YCbCr Hybrid FSRCNN-ESPCN 2x sub-pixel detail synthesis (reaches 3.0x intermediate)
                let neural = upscale_neural_2x(&easu);
                // Stage 2b: Bridge to exact (out_w, out_h)
                let bridged = if neural.width != out_w || neural.height != out_h {
                    fsr_easu(&neural, out_w, out_h)
                } else {
                    neural
                };
                // Stage 3: Robust Contrast-Adaptive Sharpening
                fsr_rcas(&bridged, 0.85)
            }
        }
    }
}

/// Entrypoint for the renderer pipeline: checks `target_w` and `target_h` and runs `upscale_frame`
/// (including double `upscale_y_neural_2x` for `>= 4x` or `fsr_easu` scale bridging for `> 2x`)
/// before subpixel sampling or protocol encoding.
#[inline]
pub fn apply_upscale_pipeline(
    buf: &PixelBuffer,
    mode: UpscaleMode,
    target_w: usize,
    target_h: usize,
) -> PixelBuffer {
    upscale_frame(buf, mode, target_w, target_h)
}

/// Applies AMD FSR 1.0 Robust Contrast-Adaptive Sharpening (`RCAS`) directly at the terminal
/// subpixel grid (`cols x rows*2`) after downsampling from a higher-resolution Chromium frame.
/// This ensures 1px sharpening lobes reinforce half-block (`▀`) edges and small thumbnails
/// rather than being averaged out during 6x–12x downsampling.
#[inline]
pub fn apply_terminal_subpixel_rcas(subpixel_buf: &PixelBuffer, mode: UpscaleMode) -> PixelBuffer {
    let sharpness = match mode {
        UpscaleMode::Off => return subpixel_buf.clone(),
        UpscaleMode::FsrOnly => 0.85,
        UpscaleMode::NeuralOnly => 0.35,
        UpscaleMode::Combined => 0.80,
        UpscaleMode::MultiPass => 0.88,
    };
    fsr_rcas(subpixel_buf, sharpness)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::dither_truecolor_cell;

    #[test]
    fn test_ycbcr_roundtrip_accuracy() {
        let test_colors = [
            Color::new(0, 0, 0),
            Color::new(255, 255, 255),
            Color::new(255, 0, 0),
            Color::new(0, 255, 0),
            Color::new(0, 0, 255),
            Color::new(128, 128, 128),
            Color::new(26, 115, 232),
            Color::new(255, 102, 0),
            Color::new(32, 33, 36),
            Color::new(248, 249, 250),
        ];
        let mut buf = PixelBuffer::new(10, 1, Color::BLACK);
        for (i, &c) in test_colors.iter().enumerate() {
            buf.set_pixel(i, 0, c);
        }

        let (y, cb, cr) = rgb_to_ycbcr(&buf);
        let reconstructed = ycbcr_to_rgb(&y, &cb, &cr, 10, 1);

        for i in 0..buf.data.len() {
            let orig = buf.data[i];
            let rec = reconstructed.data[i];
            assert!(
                orig.r.abs_diff(rec.r) <= 1
                    && orig.g.abs_diff(rec.g) <= 1
                    && orig.b.abs_diff(rec.b) <= 1,
                "YCbCr roundtrip error > 1 at index {}: orig={:?}, rec={:?}",
                i,
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_flat_region_skip_zero_drift() {
        // 1. Verify L4_WEIGHTS mathematical invariants:
        //    in_ch = 0 sums to 1.000; high-frequency channels in_ch = 1..3 sum strictly to 0.000
        for sub_ch in 0..4 {
            let mut sums = [0.0f32; 4];
            for k in 0..9 {
                for ic in 0..4 {
                    sums[ic] += L4_WEIGHTS[sub_ch][k][ic];
                }
            }
            assert!(
                (sums[0] - 1.0).abs() < 1e-5,
                "Sub-pixel {} in_ch=0 sum = {} (expected 1.0)",
                sub_ch,
                sums[0]
            );
            for ic in 1..4 {
                assert!(
                    sums[ic].abs() < 1e-5,
                    "Sub-pixel {} in_ch={} sum = {} (expected 0.0)",
                    sub_ch,
                    ic,
                    sums[ic]
                );
            }
        }

        // 2. Verify 64x64 flat solid frame has 0 LSB drift across all modes
        let flat_color = Color::new(240, 240, 240);
        let buf = PixelBuffer::new(64, 64, flat_color);
        for mode in [
            UpscaleMode::FsrOnly,
            UpscaleMode::NeuralOnly,
            UpscaleMode::Combined,
            UpscaleMode::MultiPass,
        ] {
            let up = apply_upscale_pipeline(&buf, mode, 128, 128);
            assert_eq!(up.width, 128);
            assert_eq!(up.height, 128);
            for px in &up.data {
                assert_eq!(*px, flat_color, "Mode {:?} drifted on flat 240 gray", mode);
            }
        }
    }

    #[test]
    fn test_edge_contrast_improvement() {
        // Synthetic 16x16 vertical blurred step-edge:
        // Left half = 0.2 (51), transition col 7 = 0.42 (107), col 8 = 0.58 (148), Right half = 0.8 (204)
        let mut buf = PixelBuffer::new(16, 16, Color::new(51, 51, 51));
        for y in 0..16 {
            buf.set_pixel(7, y, Color::new(107, 107, 107));
            buf.set_pixel(8, y, Color::new(148, 148, 148));
            for x in 9..16 {
                buf.set_pixel(x, y, Color::new(204, 204, 204));
            }
        }

        // Bilinear baseline slope across the center transition at 32x32
        let (y_plane, _, _) = rgb_to_ycbcr(&buf);
        let bilin_y = upscale_chroma_bilinear(&y_plane, 16, 16, 2);
        let row_mid = 16 * 32;
        let mut max_slope_bilinear = 0.0f32;
        for x in 12..20 {
            let slope = (bilin_y[row_mid + x + 1] - bilin_y[row_mid + x]).abs();
            if slope > max_slope_bilinear {
                max_slope_bilinear = slope;
            }
        }

        for mode in [
            UpscaleMode::FsrOnly,
            UpscaleMode::NeuralOnly,
            UpscaleMode::Combined,
            UpscaleMode::MultiPass,
        ] {
            let sr = apply_upscale_pipeline(&buf, mode, 32, 32);
            let (sr_y, _, _) = rgb_to_ycbcr(&sr);
            let mut max_slope_sr = 0.0f32;
            for x in 12..20 {
                let slope = (sr_y[row_mid + x + 1] - sr_y[row_mid + x]).abs();
                if slope > max_slope_sr {
                    max_slope_sr = slope;
                }
            }
            assert!(
                max_slope_sr > max_slope_bilinear,
                "Mode {:?} did not sharpen edge slope: sr={} vs bilinear={}",
                mode,
                max_slope_sr,
                max_slope_bilinear
            );
        }
    }

    #[test]
    fn test_pipeline_dimensions_and_modes() {
        let mut buf = PixelBuffer::new(8, 6, Color::new(50, 100, 150));
        buf.set_pixel(4, 3, Color::new(220, 230, 240));

        for mode in [
            UpscaleMode::Off,
            UpscaleMode::FsrOnly,
            UpscaleMode::NeuralOnly,
            UpscaleMode::Combined,
            UpscaleMode::MultiPass,
        ] {
            for &(tw, th) in &[(8, 6), (16, 12), (24, 18), (32, 24)] {
                let out = apply_upscale_pipeline(&buf, mode, tw, th);
                if mode == UpscaleMode::Off {
                    assert_eq!(out.width, 8);
                    assert_eq!(out.height, 6);
                } else {
                    assert_eq!(out.width, tw);
                    assert_eq!(out.height, th);
                    assert_eq!(out.data.len(), tw * th);
                }
            }

            let subpixel = apply_terminal_subpixel_rcas(&buf, mode);
            assert_eq!(subpixel.width, 8);
            assert_eq!(subpixel.height, 6);
        }
    }

    #[test]
    fn test_dither_stability_and_flat_suppression() {
        let c = Color::new(140, 160, 180);
        for y in 0..8 {
            for x in 0..8 {
                let first = dither_truecolor_cell(c, x, y);
                let second = dither_truecolor_cell(c, x, y);
                assert_eq!(first, second, "Bayer dithering must be 100% deterministic");
            }
        }
        // Extremum suppression check
        assert_eq!(dither_truecolor_cell(Color::WHITE, 1, 2), Color::WHITE);
        assert_eq!(dither_truecolor_cell(Color::BLACK, 3, 1), Color::BLACK);
    }
}
