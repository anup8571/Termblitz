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

/// Exact BT.601 Fixed-Point Bit-Shift Integer RGB -> YCbCr conversion for the hot rendering loop:
/// - `Y  = (( 77 * R + 150 * G +  29 * B) >> 8)`
/// - `Cb = (((-43 * R -  85 * G + 128 * B) >> 8) + 128)`
/// - `Cr = (((128 * R - 107 * G -  21 * B) >> 8) + 128)`
#[inline(always)]
pub fn bt601_rgb_to_ycbcr_fixed(c: Color) -> (u8, u8, u8) {
    let r = c.r as i32;
    let g = c.g as i32;
    let b = c.b as i32;
    let y = ((77 * r + 150 * g + 29 * b) >> 8).clamp(0, 255) as u8;
    let cb = (((-43 * r - 85 * g + 128 * b) >> 8) + 128).clamp(0, 255) as u8;
    let cr = (((128 * r - 107 * g - 21 * b) >> 8) + 128).clamp(0, 255) as u8;
    (y, cb, cr)
}

/// Exact BT.601 Fixed-Point Bit-Shift Integer YCbCr -> RGB conversion:
/// Uses branchless `.clamp(0, 255)` and integer shifts so LLVM emits packed NEON/SIMD instructions.
#[inline(always)]
pub fn bt601_ycbcr_to_rgb_fixed(y: u8, cb: u8, cr: u8) -> Color {
    let y_i = y as i32;
    let cb_i = (cb as i32) - 128;
    let cr_i = (cr as i32) - 128;
    let r = (y_i + ((359 * cr_i + 128) >> 8)).clamp(0, 255) as u8;
    let g = (y_i - ((88 * cb_i + 183 * cr_i + 128) >> 8)).clamp(0, 255) as u8;
    let b = (y_i + ((454 * cb_i + 128) >> 8)).clamp(0, 255) as u8;
    Color::new(r, g, b)
}

/// Splits a `PixelBuffer` into three pre-allocated `Vec<f32>` planes (`y_plane`, `cb_plane`, `cr_plane`)
/// using Q16.16 fixed-point BT.601 integer coefficients (`77*R + 150*G + 29*B`, `-43*R - 85*G + 128*B`,
/// `128*R - 107*G - 21*B`) with zero heap allocations when capacities are already warm.
#[inline]
pub fn rgb_to_ycbcr_into(
    buf: &PixelBuffer,
    y_plane: &mut Vec<f32>,
    cb_plane: &mut Vec<f32>,
    cr_plane: &mut Vec<f32>,
) {
    let len = (buf.width * buf.height).min(buf.data.len());
    y_plane.clear();
    cb_plane.clear();
    cr_plane.clear();
    y_plane.reserve(len);
    cb_plane.reserve(len);
    cr_plane.reserve(len);

    // 1.0 / (255.0 * 256.0)
    const INV_65280: f32 = 1.0 / 65280.0;

    for px in &buf.data[..len] {
        let r = px.r as i32;
        let g = px.g as i32;
        let b = px.b as i32;

        // Exact BT.601 Q8fixed numerators before normalization
        let y_q8 = 77 * r + 150 * g + 29 * b;
        let cb_q8 = -43 * r - 85 * g + 128 * b;
        let cr_q8 = 128 * r - 107 * g - 21 * b;

        y_plane.push((y_q8 as f32) * INV_65280);
        cb_plane.push((cb_q8 as f32) * INV_65280);
        cr_plane.push((cr_q8 as f32) * INV_65280);
    }
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
    rgb_to_ycbcr_into(buf, &mut y_plane, &mut cb_plane, &mut cr_plane);
    (y_plane, cb_plane, cr_plane)
}

/// Merges normalized `Y` (`0.0..=1.0`), `Cb` (`-0.5..=0.5`), and `Cr` (`-0.5..=0.5`) planes
/// into `out_data` using Q16.16 fixed-point BT.601 inverse coefficients without reallocating.
#[inline]
pub fn ycbcr_to_rgb_into(
    y_plane: &[f32],
    cb_plane: &[f32],
    cr_plane: &[f32],
    width: usize,
    height: usize,
    out_data: &mut Vec<Color>,
) {
    let len = width * height;
    out_data.clear();
    out_data.reserve(len);

    let safe_len = len
        .min(y_plane.len())
        .min(cb_plane.len())
        .min(cr_plane.len());

    for i in 0..safe_len {
        let y_fp = (y_plane[i] * 65280.0).round() as i32;
        let cb_fp = (cb_plane[i] * 65280.0).round() as i32;
        let cr_fp = (cr_plane[i] * 65280.0).round() as i32;

        // Fixed-point BT.601 inverse:
        // R = Y + 1.402 * Cr      (359 / 256 = 1.40234375)
        // G = Y - 0.344136 * Cb - 0.714136 * Cr (88 / 256 = 0.34375, 183 / 256 = 0.71484375)
        // B = Y + 1.772 * Cb      (454 / 256 = 1.7734375)
        let r = ((y_fp + ((359 * cr_fp) >> 8) + 128) >> 8).clamp(0, 255) as u8;
        let g = ((y_fp - ((88 * cb_fp + 183 * cr_fp) >> 8) + 128) >> 8).clamp(0, 255) as u8;
        let b = ((y_fp + ((454 * cb_fp) >> 8) + 128) >> 8).clamp(0, 255) as u8;

        out_data.push(Color::new(r, g, b));
    }

    if out_data.len() < len {
        out_data.resize(len, Color::BLACK);
    }
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
    let mut data = Vec::with_capacity(width * height);
    ycbcr_to_rgb_into(y_plane, cb_plane, cr_plane, width, height, &mut data);
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

/// Reusable scratch buffers for multi-pass recursive neural super-resolution (`2x -> 4x -> 8x`).
/// Eliminates repeated heap allocations across recursive stages by resizing flat vectors in-place.
#[derive(Default)]
pub struct NeuralScratch {
    pub active_px: Vec<bool>,
    pub feat_l2: Vec<[f32; 4]>,
    pub feat_l3: Vec<[f32; 4]>,
    pub subpixels: Vec<[f32; 4]>,
    pub y_ping: Vec<f32>,
    pub y_pong: Vec<f32>,
}

/// Runs the 4-layer Hybrid FSRCNN-ESPCN super-resolution network on `y_plane` (`w x h`)
/// and writes the `(w * 2) x (h * 2)` output into `out_y` using reusable `scratch` slices.
pub fn upscale_y_neural_2x_into(
    y_plane: &[f32],
    w: usize,
    h: usize,
    out_y: &mut Vec<f32>,
    scratch: &mut NeuralScratch,
) {
    if w == 0 || h == 0 || y_plane.len() < w * h {
        out_y.clear();
        return;
    }

    let num_px = w * h;
    let out_len = (w * 2) * (h * 2);
    out_y.resize(out_len, 0.0);

    scratch.active_px.clear();
    scratch.active_px.resize(num_px, false);
    scratch.feat_l2.resize(num_px, [0.0f32; 4]);

    let mut any_active = false;
    let max_x = w - 1;
    let max_y = h - 1;

    for y in 0..h {
        let ym1 = y.saturating_sub(1);
        let yp1 = (y + 1).min(max_y);

        let r_top = &y_plane[ym1 * w..ym1 * w + w];
        let r_mid = &y_plane[y * w..y * w + w];
        let r_bot = &y_plane[yp1 * w..yp1 * w + w];
        let dst_row = &mut scratch.feat_l2[y * w..(y + 1) * w];
        let act_row = &mut scratch.active_px[y * w..(y + 1) * w];

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
                dst_row[x] = [patch[4], 0.0, 0.0, 0.0];
                continue;
            }

            act_row[x] = true;
            any_active = true;

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

    if !any_active {
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
        return;
    }

    scratch.feat_l3.resize(num_px, [0.0f32; 4]);
    for y in 0..h {
        let ym1 = y.saturating_sub(1);
        let yp1 = (y + 1).min(max_y);

        let r_top = &scratch.feat_l2[ym1 * w..ym1 * w + w];
        let r_mid = &scratch.feat_l2[y * w..y * w + w];
        let r_bot = &scratch.feat_l2[yp1 * w..yp1 * w + w];
        let act_row = &scratch.active_px[y * w..(y + 1) * w];
        let dst_row = &mut scratch.feat_l3[y * w..(y + 1) * w];

        for x in 0..w {
            if !act_row[x] {
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

    scratch.subpixels.resize(num_px, [0.0f32; 4]);
    for y in 0..h {
        let ym1 = y.saturating_sub(1);
        let yp1 = (y + 1).min(max_y);

        let r_top = &scratch.feat_l3[ym1 * w..ym1 * w + w];
        let r_mid = &scratch.feat_l3[y * w..y * w + w];
        let r_bot = &scratch.feat_l3[yp1 * w..yp1 * w + w];
        let y_row = &y_plane[y * w..(y + 1) * w];
        let act_row = &scratch.active_px[y * w..(y + 1) * w];
        let dst_row = &mut scratch.subpixels[y * w..(y + 1) * w];

        for x in 0..w {
            if !act_row[x] {
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

    pixel_shuffle_2x(&scratch.subpixels, w, h, out_y);
}

/// Runs the 4-layer Hybrid FSRCNN-ESPCN super-resolution network on a normalized `Y` luminance plane
/// (`w x h`), returning a `(w * 2) x (h * 2)` luminance plane.
/// Uses the Content-Aware Sparse Skip Engine (`should_enhance_pixel` & 8x8 tile gating) so flat
/// backgrounds (`luma_range < 0.015`) completely bypass all 4 convolutional layers and output
/// `center_y` untouched with 0% neural bias drift.
pub fn upscale_y_neural_2x(y_plane: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out_y = Vec::new();
    let mut scratch = NeuralScratch::default();
    upscale_y_neural_2x_into(y_plane, w, h, &mut out_y, &mut scratch);
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

// =============================================================================
// Task 4: Fast JPEG Denoising / 8x8 DCT Block Artifact Removal Pre-Filter
// =============================================================================

/// Detects whether `buf` exhibits 8x8 JPEG DCT block quantization discontinuities
/// (where block boundary jumps at `x % 8 == 0` or `y % 8 == 0` significantly exceed
/// intra-block smooth variance within each 8x8 tile).
pub fn has_jpeg_block_artifacts(buf: &PixelBuffer) -> bool {
    let w = buf.width;
    let h = buf.height;
    if w < 16 || h < 16 || buf.data.len() < w * h {
        return false;
    }

    let mut boundary_sum: u64 = 0;
    let mut boundary_cnt: u64 = 0;
    let mut interior_sum: u64 = 0;
    let mut interior_cnt: u64 = 0;

    // Sample vertical 8x8 boundaries (x = 8, 16, 24...) vs block midpoints (x = 4, 12, 20...)
    let step_y = (h / 32).max(1);
    let mut y = 0;
    while y < h {
        let row = &buf.data[y * w..(y + 1) * w];
        let mut bx = 8;
        while bx < w {
            let l0 = row[bx - 1].luma_fast() as i32;
            let l1 = row[bx].luma_fast() as i32;
            let diff = (l1 - l0).unsigned_abs() as u64;
            // Count quantization-scale steps (2..=28), ignoring huge structural object edges
            if (2..=28).contains(&diff) {
                boundary_sum += diff;
                boundary_cnt += 1;
            }
            let mx = bx - 4;
            let m0 = row[mx - 1].luma_fast() as i32;
            let m1 = row[mx].luma_fast() as i32;
            let mdiff = (m1 - m0).unsigned_abs() as u64;
            if mdiff <= 28 {
                interior_sum += mdiff;
                interior_cnt += 1;
            }
            bx += 8;
        }
        y += step_y;
    }

    // Sample horizontal 8x8 boundaries (y = 8, 16, 24...) vs block midpoints (y = 4, 12, 20...)
    let step_x = (w / 32).max(1);
    let mut by = 8;
    while by < h {
        let row_prev = &buf.data[(by - 1) * w..by * w];
        let row_curr = &buf.data[by * w..(by + 1) * w];
        let my = by - 4;
        let row_m0 = &buf.data[(my - 1) * w..my * w];
        let row_m1 = &buf.data[my * w..(my + 1) * w];

        let mut x = 0;
        while x < w {
            let l0 = row_prev[x].luma_fast() as i32;
            let l1 = row_curr[x].luma_fast() as i32;
            let diff = (l1 - l0).unsigned_abs() as u64;
            if (2..=28).contains(&diff) {
                boundary_sum += diff;
                boundary_cnt += 1;
            }
            let m0 = row_m0[x].luma_fast() as i32;
            let m1 = row_m1[x].luma_fast() as i32;
            let mdiff = (m1 - m0).unsigned_abs() as u64;
            if mdiff <= 28 {
                interior_sum += mdiff;
                interior_cnt += 1;
            }
            x += step_x;
        }
        by += 8;
    }

    if boundary_cnt < 4 {
        return false;
    }

    let avg_boundary_x10 = (boundary_sum * 10) / boundary_cnt.max(1);
    let avg_interior_x10 = (interior_sum * 10) / interior_cnt.max(1);

    // Trigger when 8x8 boundary discontinuity is >= 25% higher than interior gradient
    avg_boundary_x10 >= 25 && avg_boundary_x10 * 4 > avg_interior_x10.max(8) * 5
}

/// Fast single-pass Bilateral / Box-Sigma Edge-Preserving Denoising Filter (`remove_jpeg_artifacts`).
/// Smooths 8x8 DCT grid blocking and mosquito ringing noise while strictly preserving high-contrast
/// text glyphs, UI borders, and vector edges (`max_channel_diff > sigma`).
pub fn remove_jpeg_artifacts(buf: &mut PixelBuffer) {
    let w = buf.width;
    let h = buf.height;
    if w < 3 || h < 3 || buf.data.len() < w * h {
        return;
    }

    let src = buf.data.clone();
    let max_x = w - 1;
    let max_y = h - 1;

    // 3x3 spatial Gaussian-like integer weights: center = 4, axial = 2, diagonal = 1
    const SPATIAL_W: [u32; 9] = [1, 2, 1, 2, 4, 2, 1, 2, 1];

    for y in 0..h {
        let ym1 = y.saturating_sub(1);
        let yp1 = (y + 1).min(max_y);
        let r_top = &src[ym1 * w..ym1 * w + w];
        let r_mid = &src[y * w..y * w + w];
        let r_bot = &src[yp1 * w..yp1 * w + w];
        let dst_row = &mut buf.data[y * w..(y + 1) * w];

        let on_y_block_edge = (y & 7) == 0 || (y & 7) == 7;

        for x in 0..w {
            let xm1 = x.saturating_sub(1);
            let xp1 = (x + 1).min(max_x);
            let center = r_mid[x];

            let neighbors = [
                r_top[xm1], r_top[x], r_top[xp1],
                r_mid[xm1], center,   r_mid[xp1],
                r_bot[xm1], r_bot[x], r_bot[xp1],
            ];

            // Fast flat-region skip: if all 4 axial neighbors equal center, no denoising needed
            if neighbors[1] == center
                && neighbors[3] == center
                && neighbors[5] == center
                && neighbors[7] == center
            {
                continue;
            }

            let on_x_block_edge = (x & 7) == 0 || (x & 7) == 7;
            // Use wider bilateral range threshold (24) on 8x8 DCT block boundaries,
            // and tighter range threshold (14) inside blocks for mosquito ringing removal.
            // Any high-contrast edge (diff > threshold, such as text or icons) gets weight 0!
            let sigma_thresh: u8 = if on_x_block_edge || on_y_block_edge { 24 } else { 14 };

            let mut sum_r: u32 = 0;
            let mut sum_g: u32 = 0;
            let mut sum_b: u32 = 0;
            let mut sum_w: u32 = 0;

            for k in 0..9 {
                let nk = neighbors[k];
                let dr = center.r.abs_diff(nk.r);
                let dg = center.g.abs_diff(nk.g);
                let db = center.b.abs_diff(nk.b);
                let max_d = dr.max(dg).max(db);

                if max_d <= sigma_thresh {
                    // Range weight: 2 for very close values (<= sigma/2), 1 up to sigma_thresh
                    let range_w = if max_d <= (sigma_thresh >> 1) { 2 } else { 1 };
                    let w_k = SPATIAL_W[k] * range_w;
                    sum_r += (nk.r as u32) * w_k;
                    sum_g += (nk.g as u32) * w_k;
                    sum_b += (nk.b as u32) * w_k;
                    sum_w += w_k;
                }
            }

            if sum_w > 0 {
                let half_w = sum_w >> 1;
                dst_row[x] = Color::new(
                    ((sum_r + half_w) / sum_w) as u8,
                    ((sum_g + half_w) / sum_w) as u8,
                    ((sum_b + half_w) / sum_w) as u8,
                );
            }
        }
    }
}

// =============================================================================
// Task 5: True Pixel Art / Vector Mode Detector & Scale2x / HQ2x Scaler
// =============================================================================

/// Content classifier that detects whether `buf` is Pixel Art, retro game graphics, or crisp
/// low-palette vector iconography:
/// 1. Samples up to 1024 pixels across the buffer and counts unique quantized RGB colors (`< 64`).
/// 2. Verifies high sharp grid alignment: adjacent pixel transitions are dominated by either
///    exact flat runs (`delta == 0`) or crisp step edges (`delta >= 20`), with minimal
///    continuous photographic gradient ramps (`1 <= delta <= 12`).
pub fn detect_pixel_art(buf: &PixelBuffer) -> bool {
    let w = buf.width;
    let h = buf.height;
    let total = w * h;
    if w < 4 || h < 4 || buf.data.len() < total {
        return false;
    }

    // 12-bit quantized color presence bitset (4096 bits = 64 x u64 words on the stack, zero heap alloc!)
    let mut color_bitset = [0u64; 64];
    let mut unique_colors = 0usize;

    let sample_step = (total / 1024).max(1);
    let mut idx = 0;
    while idx < total {
        let c = buf.data[idx];
        // Quantize to 4 bits per channel (16x16x16 = 4096 buckets)
        let bucket = (((c.r >> 4) as usize) << 8)
            | (((c.g >> 4) as usize) << 4)
            | ((c.b >> 4) as usize);
        let word = bucket >> 6;
        let bit = 1u64 << (bucket & 63);
        if (color_bitset[word] & bit) == 0 {
            color_bitset[word] |= bit;
            unique_colors += 1;
            if unique_colors >= 64 {
                return false;
            }
        }
        idx += sample_step;
    }

    // Require at least 2 distinct colors (a 1-color solid blank canvas doesn't need special pixel-art scaling)
    if unique_colors < 2 {
        return false;
    }

    // Measure sharp grid alignment vs smooth photographic gradients
    let mut flat_runs: u32 = 0;
    let mut sharp_steps: u32 = 0;
    let mut smooth_ramps: u32 = 0;

    let row_step = (h / 32).max(1);
    let col_step = (w / 32).max(1);
    let mut y = 0;
    while y < h {
        let row = &buf.data[y * w..(y + 1) * w];
        let mut x = 0;
        while x + 1 < w {
            let c0 = row[x];
            let c1 = row[x + 1];
            let delta = c0
                .r
                .abs_diff(c1.r)
                .max(c0.g.abs_diff(c1.g))
                .max(c0.b.abs_diff(c1.b));
            if delta == 0 {
                flat_runs += 1;
            } else if delta >= 20 {
                sharp_steps += 1;
            } else if delta <= 12 {
                smooth_ramps += 1;
            }
            x += col_step;
        }
        y += row_step;
    }

    let total_pairs = (flat_runs + sharp_steps + smooth_ramps).max(1);
    // Pixel art has >= 1 sharp edge, < 64 unique colors, and < 20% smooth photographic gradient transitions
    sharp_steps >= 1 && (smooth_ramps * 5 < total_pairs)
}

/// Pure-Rust **Scale2x (AdvMame2x)** directional 2x pixel-art scaler.
/// Expands each pixel `P` into a `2x2` quad `[E0, E1, E2, E3]` using 4-connected neighbors
/// `B` (north), `D` (west), `F` (east), `H` (south) with zero interpolation blur:
/// - `E0 = D == B && B != H && D != F ? D : P`
/// - `E1 = B == F && B != H && D != F ? F : P`
/// - `E2 = D == H && B != H && D != F ? D : P`
/// - `E3 = H == F && B != H && D != F ? F : P`
pub fn scale2x(buf: &PixelBuffer) -> PixelBuffer {
    let w = buf.width;
    let h = buf.height;
    if w == 0 || h == 0 || buf.data.len() < w * h {
        return PixelBuffer::new(0, 0, Color::BLACK);
    }

    let out_w = w * 2;
    let out_h = h * 2;
    let mut out_data = vec![Color::BLACK; out_w * out_h];
    let max_x = w - 1;
    let max_y = h - 1;

    for y in 0..h {
        let ym1 = y.saturating_sub(1);
        let yp1 = (y + 1).min(max_y);

        let r_top = &buf.data[ym1 * w..ym1 * w + w];
        let r_mid = &buf.data[y * w..y * w + w];
        let r_bot = &buf.data[yp1 * w..yp1 * w + w];

        let top_out_start = (y * 2) * out_w;
        let bot_out_start = (y * 2 + 1) * out_w;
        let (out_top_row, rest) = out_data[top_out_start..bot_out_start + out_w].split_at_mut(out_w);
        let out_bot_row = &mut rest[..out_w];

        for x in 0..w {
            let xm1 = x.saturating_sub(1);
            let xp1 = (x + 1).min(max_x);

            let b = r_top[x];
            let d = r_mid[xm1];
            let p = r_mid[x];
            let f = r_mid[xp1];
            let h_px = r_bot[x];

            let (e0, e1, e2, e3) = if b != h_px && d != f {
                (
                    if d == b { d } else { p },
                    if b == f { f } else { p },
                    if d == h_px { d } else { p },
                    if h_px == f { f } else { p },
                )
            } else {
                (p, p, p, p)
            };

            let dx = x * 2;
            out_top_row[dx] = e0;
            out_top_row[dx + 1] = e1;
            out_bot_row[dx] = e2;
            out_bot_row[dx + 1] = e3;
        }
    }

    PixelBuffer {
        width: out_w,
        height: out_h,
        data: out_data,
    }
}

/// Scales `buf` to `(target_w, target_h)` using recursive **Scale2x / HQ2x** passes (`2x -> 4x -> 8x`)
/// followed by exact integer nearest-neighbor grid mapping so pixel-art and vector sprites retain
/// 100% razor-sharp blocky edges without bilinear/Lanczos smoothing.
pub fn hq2x_scale(buf: &PixelBuffer, target_w: usize, target_h: usize) -> PixelBuffer {
    if buf.width == 0 || buf.height == 0 || buf.data.is_empty() {
        return PixelBuffer::new(0, 0, Color::BLACK);
    }
    let out_w = if target_w > 0 { target_w } else { buf.width * 2 };
    let out_h = if target_h > 0 { target_h } else { buf.height * 2 };

    let mut curr = scale2x(buf);
    if out_w >= buf.width * 4 || out_h >= buf.height * 4 {
        curr = scale2x(&curr);
    }
    if out_w >= buf.width * 8 || out_h >= buf.height * 8 {
        curr = scale2x(&curr);
    }

    if curr.width == out_w && curr.height == out_h {
        return curr;
    }

    // Nearest-neighbor grid snap to exact (out_w, out_h)
    let mut final_data = vec![Color::BLACK; out_w * out_h];
    for oy in 0..out_h {
        let sy = ((oy * curr.height) + (curr.height >> 1)) / out_h;
        let sy_clamped = sy.min(curr.height - 1);
        let src_row = &curr.data[sy_clamped * curr.width..(sy_clamped + 1) * curr.width];
        let dst_row = &mut final_data[oy * out_w..(oy + 1) * out_w];
        for (ox, dst_px) in dst_row.iter_mut().enumerate() {
            let sx = ((ox * curr.width) + (curr.width >> 1)) / out_w;
            *dst_px = src_row[sx.min(curr.width - 1)];
        }
    }

    PixelBuffer {
        width: out_w,
        height: out_h,
        data: final_data,
    }
}

/// Upscales `buf` using the YCbCr Hybrid FSRCNN-ESPCN neural pipeline and dynamically chains
/// recursive passes for deep zooming (`2x -> 4x -> 8x`) using reusable `NeuralScratch` buffers:
/// - For `>= 8x` target scales (`out_w >= buf.width * 8` or `out_h >= buf.height * 8`), chains
///   `upscale_y_neural_2x_into` three times (`1x -> 2x -> 4x -> 8x`) with zero intermediate heap churn.
/// - For `>= 4x` target scales (`out_w >= buf.width * 4` or `out_h >= buf.height * 4`), chains
///   `upscale_y_neural_2x_into` twice (`1x -> 2x -> 4x`).
/// - Whenever `out_w != neural.width` or `out_h != neural.height`, feeds the neural output into
///   `fsr_easu` to bridge to the exact `(out_w, out_h)` target resolution.
fn upscale_neural_adaptive(buf: &PixelBuffer, out_w: usize, out_h: usize) -> PixelBuffer {
    let w = buf.width;
    let h = buf.height;

    let neural_buf = if out_w >= w * 8 || out_h >= h * 8 {
        let (y_plane, cb_plane, cr_plane) = rgb_to_ycbcr(buf);
        let mut scratch = NeuralScratch::default();
        let mut y_2x = Vec::new();
        let mut y_4x = Vec::new();
        let mut y_8x = Vec::new();
        upscale_y_neural_2x_into(&y_plane, w, h, &mut y_2x, &mut scratch);
        upscale_y_neural_2x_into(&y_2x, w * 2, h * 2, &mut y_4x, &mut scratch);
        upscale_y_neural_2x_into(&y_4x, w * 4, h * 4, &mut y_8x, &mut scratch);
        let cb_8x = upscale_chroma_bilinear(&cb_plane, w, h, 8);
        let cr_8x = upscale_chroma_bilinear(&cr_plane, w, h, 8);
        ycbcr_to_rgb(&y_8x, &cb_8x, &cr_8x, w * 8, h * 8)
    } else if out_w >= w * 4 || out_h >= h * 4 {
        let (y_plane, cb_plane, cr_plane) = rgb_to_ycbcr(buf);
        let mut scratch = NeuralScratch::default();
        let mut y_2x = Vec::new();
        let mut y_4x = Vec::new();
        upscale_y_neural_2x_into(&y_plane, w, h, &mut y_2x, &mut scratch);
        upscale_y_neural_2x_into(&y_2x, w * 2, h * 2, &mut y_4x, &mut scratch);
        let cb_4x = upscale_chroma_bilinear(&cb_plane, w, h, 4);
        let cr_4x = upscale_chroma_bilinear(&cr_plane, w, h, 4);
        ycbcr_to_rgb(&y_4x, &cb_4x, &cr_4x, w * 4, h * 4)
    } else {
        upscale_neural_2x(buf)
    };

    if out_w != neural_buf.width || out_h != neural_buf.height {
        fsr_easu(&neural_buf, out_w, out_h)
    } else {
        neural_buf
    }
}

/// Master Super-Resolution Dispatcher with Pixel-Art Override & Adaptive JPEG Denoising:
/// - Automatically detects (or honors `force_pixel_art`) pixel-art / low-palette sprites and switches
///   to crisp `hq2x_scale` (`Scale2x` recursive 2x/4x/8x + nearest-neighbor snap).
/// - Automatically detects 8x8 JPEG DCT block artifacts (`has_jpeg_block_artifacts`) and runs
///   `remove_jpeg_artifacts` before neural/FSR upscaling so compression blocks are never sharpened.
/// - Supports 2x, 4x, and 8x deep-zoom recursive neural chaining.
pub fn upscale_frame_with_options(
    buf: &PixelBuffer,
    mode: UpscaleMode,
    target_w: usize,
    target_h: usize,
    force_pixel_art: bool,
) -> PixelBuffer {
    if buf.width == 0 || buf.height == 0 || buf.data.is_empty() {
        return PixelBuffer::new(0, 0, Color::BLACK);
    }

    let out_w = if target_w > 0 { target_w } else { buf.width * 2 };
    let out_h = if target_h > 0 { target_h } else { buf.height * 2 };

    if mode == UpscaleMode::Off && !force_pixel_art {
        return buf.clone();
    }

    // 1. Pixel-Art / Vector Mode Bypass: preserve crisp blocky pixel grids via Scale2x
    if force_pixel_art || detect_pixel_art(buf) {
        return hq2x_scale(buf, out_w, out_h);
    }

    // 2. Adaptive JPEG 8x8 Block Artifact Denoising Pre-Filter
    let denoised_storage;
    let clean_buf = if has_jpeg_block_artifacts(buf) {
        let mut copy = buf.clone();
        remove_jpeg_artifacts(&mut copy);
        denoised_storage = copy;
        &denoised_storage
    } else {
        buf
    };

    match mode {
        UpscaleMode::Off => clean_buf.clone(),
        UpscaleMode::FsrOnly => {
            let easu = fsr_easu(clean_buf, out_w, out_h);
            fsr_rcas(&easu, 0.85)
        }
        UpscaleMode::NeuralOnly => upscale_neural_adaptive(clean_buf, out_w, out_h),
        UpscaleMode::Combined => {
            let neural = upscale_neural_adaptive(clean_buf, out_w, out_h);
            fsr_rcas(&neural, 0.80)
        }
        UpscaleMode::MultiPass => {
            if out_w >= clean_buf.width * 8 || out_h >= clean_buf.height * 8 {
                // For >= 8x deep zoom: chain two 2x neural passes (4x) -> 1.5x EASU -> third 2x neural pass -> exact EASU bridge -> RCAS
                let stage1_4x = upscale_neural_adaptive(clean_buf, clean_buf.width * 4, clean_buf.height * 4);
                let mid_w = ((stage1_4x.width * 3) / 2).max(out_w / 2).max(stage1_4x.width + 1);
                let mid_h = ((stage1_4x.height * 3) / 2).max(out_h / 2).max(stage1_4x.height + 1);
                let easu_mid = fsr_easu(&stage1_4x, mid_w, mid_h);
                let stage2_neural = upscale_neural_2x(&easu_mid);
                let bridged = if stage2_neural.width != out_w || stage2_neural.height != out_h {
                    fsr_easu(&stage2_neural, out_w, out_h)
                } else {
                    stage2_neural
                };
                fsr_rcas(&bridged, 0.85)
            } else if out_w >= clean_buf.width * 4 || out_h >= clean_buf.height * 4 {
                // For >= 4x scales: run 2x neural detail pass -> 1.5x EASU pre-bridge -> second 2x neural pass -> exact EASU bridge -> RCAS
                let stage1_neural = upscale_neural_2x(clean_buf);
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
                let mid_w = ((clean_buf.width * 3) / 2).max(clean_buf.width + 1);
                let mid_h = ((clean_buf.height * 3) / 2).max(clean_buf.height + 1);
                let easu = fsr_easu(clean_buf, mid_w, mid_h);
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

/// Master Super-Resolution Dispatcher: processes `buf` according to the selected `UpscaleMode`.
pub fn upscale_frame(
    buf: &PixelBuffer,
    mode: UpscaleMode,
    target_w: usize,
    target_h: usize,
) -> PixelBuffer {
    upscale_frame_with_options(buf, mode, target_w, target_h, false)
}

/// Entrypoint for the renderer pipeline: checks `target_w` and `target_h` and runs `upscale_frame`
/// (including recursive `upscale_y_neural_2x_into` for `>= 4x` / `>= 8x` or `fsr_easu` scale bridging)
/// before subpixel sampling or protocol encoding.
#[inline]
pub fn apply_upscale_pipeline(
    buf: &PixelBuffer,
    mode: UpscaleMode,
    target_w: usize,
    target_h: usize,
) -> PixelBuffer {
    upscale_frame_with_options(buf, mode, target_w, target_h, false)
}

/// Extended entrypoint for the renderer pipeline allowing explicit manual `force_pixel_art` override.
#[inline]
pub fn apply_upscale_pipeline_ex(
    buf: &PixelBuffer,
    mode: UpscaleMode,
    target_w: usize,
    target_h: usize,
    force_pixel_art: bool,
) -> PixelBuffer {
    upscale_frame_with_options(buf, mode, target_w, target_h, force_pixel_art)
}

/// Pre-Allocated Scratchpad Arena (`SrScratchpad`) for zero-allocation steady-state Super-Resolution.
/// Holds reusable `Vec<f32>` planes (`y_plane`, `cb_plane`, `cr_plane`, `feat_buf_a`, `feat_buf_b`)
/// and intermediate `PixelBuffer` storage resized via `.clear()` and `.resize()` only when
/// terminal or ROI dimensions grow.
#[derive(Clone, Debug, Default)]
pub struct SrScratchpad {
    pub y_plane: Vec<f32>,
    pub cb_plane: Vec<f32>,
    pub cr_plane: Vec<f32>,
    pub feat_buf_a: Vec<[f32; 4]>,
    pub feat_buf_b: Vec<[f32; 4]>,
    pub active_mask: Vec<bool>,
    pub out_y: Vec<f32>,
    pub work_px: Vec<Color>,
    pub roi_crop: PixelBuffer,
}

impl SrScratchpad {
    pub fn new() -> Self {
        Self {
            y_plane: Vec::with_capacity(16384),
            cb_plane: Vec::with_capacity(16384),
            cr_plane: Vec::with_capacity(16384),
            feat_buf_a: Vec::with_capacity(16384),
            feat_buf_b: Vec::with_capacity(16384),
            active_mask: Vec::with_capacity(16384),
            out_y: Vec::with_capacity(16384),
            work_px: Vec::with_capacity(16384),
            roi_crop: PixelBuffer::new(0, 0, Color::BLACK),
        }
    }

    /// Zero-allocation bilateral JPEG artifact remover using `self.work_px` as the source scratch buffer.
    pub fn remove_jpeg_artifacts_in_place(&mut self, buf: &mut PixelBuffer) {
        let w = buf.width;
        let h = buf.height;
        let num_px = w * h;
        if w < 3 || h < 3 || buf.data.len() < num_px {
            return;
        }

        self.work_px.clear();
        self.work_px.extend_from_slice(&buf.data[..num_px]);
        let src = &self.work_px[..num_px];
        let max_x = w - 1;
        let max_y = h - 1;
        const SPATIAL_W: [u32; 9] = [1, 2, 1, 2, 4, 2, 1, 2, 1];

        for y in 0..h {
            let ym1 = y.saturating_sub(1);
            let yp1 = (y + 1).min(max_y);
            let r_top = &src[ym1 * w..ym1 * w + w];
            let r_mid = &src[y * w..y * w + w];
            let r_bot = &src[yp1 * w..yp1 * w + w];
            let dst_row = &mut buf.data[y * w..(y + 1) * w];
            let on_y_block_edge = (y & 7) == 0 || (y & 7) == 7;

            for x in 0..w {
                let xm1 = x.saturating_sub(1);
                let xp1 = (x + 1).min(max_x);
                let center = r_mid[x];

                let neighbors = [
                    r_top[xm1], r_top[x], r_top[xp1],
                    r_mid[xm1], center,   r_mid[xp1],
                    r_bot[xm1], r_bot[x], r_bot[xp1],
                ];

                if neighbors[1] == center
                    && neighbors[3] == center
                    && neighbors[5] == center
                    && neighbors[7] == center
                {
                    continue;
                }

                let on_x_block_edge = (x & 7) == 0 || (x & 7) == 7;
                let sigma_thresh: u8 = if on_x_block_edge || on_y_block_edge { 24 } else { 14 };

                let mut sum_r: u32 = 0;
                let mut sum_g: u32 = 0;
                let mut sum_b: u32 = 0;
                let mut sum_w: u32 = 0;

                for k in 0..9 {
                    let nk = neighbors[k];
                    let max_d = center
                        .r
                        .abs_diff(nk.r)
                        .max(center.g.abs_diff(nk.g))
                        .max(center.b.abs_diff(nk.b));

                    if max_d <= sigma_thresh {
                        let range_w = if max_d <= (sigma_thresh >> 1) { 2 } else { 1 };
                        let w_k = SPATIAL_W[k] * range_w;
                        sum_r += (nk.r as u32) * w_k;
                        sum_g += (nk.g as u32) * w_k;
                        sum_b += (nk.b as u32) * w_k;
                        sum_w += w_k;
                    }
                }

                if sum_w > 0 {
                    let half_w = sum_w >> 1;
                    dst_row[x] = Color::new(
                        ((sum_r + half_w) / sum_w) as u8,
                        ((sum_g + half_w) / sum_w) as u8,
                        ((sum_b + half_w) / sum_w) as u8,
                    );
                }
            }
        }
    }

    /// Zero-allocation Y-channel Neural CNN refinement (Layers 1 -> 2 -> 3 with residual skip)
    /// operating in-place on `buf` using pre-allocated `SrScratchpad` planes.
    pub fn refine_y_neural_1x_in_place(&mut self, buf: &mut PixelBuffer) {
        let w = buf.width;
        let h = buf.height;
        let num_px = w * h;
        if w < 3 || h < 3 || buf.data.len() < num_px {
            return;
        }

        rgb_to_ycbcr_into(buf, &mut self.y_plane, &mut self.cb_plane, &mut self.cr_plane);
        self.active_mask.clear();
        self.active_mask.resize(num_px, false);
        self.feat_buf_a.clear();
        self.feat_buf_a.resize(num_px, [0.0f32; 4]);

        let mut any_active = false;
        let max_x = w - 1;
        let max_y = h - 1;

        for y in 0..h {
            let ym1 = y.saturating_sub(1);
            let yp1 = (y + 1).min(max_y);

            let r_top = &self.y_plane[ym1 * w..ym1 * w + w];
            let r_mid = &self.y_plane[y * w..y * w + w];
            let r_bot = &self.y_plane[yp1 * w..yp1 * w + w];
            let dst_row = &mut self.feat_buf_a[y * w..(y + 1) * w];
            let act_row = &mut self.active_mask[y * w..(y + 1) * w];

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
                    dst_row[x] = [patch[4], 0.0, 0.0, 0.0];
                    continue;
                }

                act_row[x] = true;
                any_active = true;

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

        if !any_active {
            return;
        }

        self.out_y.clear();
        self.out_y.resize(num_px, 0.0);
        for y in 0..h {
            let ym1 = y.saturating_sub(1);
            let yp1 = (y + 1).min(max_y);

            let r_top = &self.feat_buf_a[ym1 * w..ym1 * w + w];
            let r_mid = &self.feat_buf_a[y * w..y * w + w];
            let r_bot = &self.feat_buf_a[yp1 * w..yp1 * w + w];
            let act_row = &self.active_mask[y * w..(y + 1) * w];
            let dst_y = &mut self.out_y[y * w..(y + 1) * w];

            for x in 0..w {
                if !act_row[x] {
                    dst_y[x] = r_mid[x][0].clamp(0.0, 1.0);
                    continue;
                }

                let xm1 = x.saturating_sub(1);
                let xp1 = (x + 1).min(max_x);

                let p = [
                    r_top[xm1], r_top[x], r_top[xp1],
                    r_mid[xm1], r_mid[x], r_mid[xp1],
                    r_bot[xm1], r_bot[x], r_bot[xp1],
                ];
                let residual_y = p[4][0];
                let wk = &L3_WEIGHTS[0];
                let mut acc = L3_BIAS[0];
                for k in 0..9 {
                    let pk = p[k];
                    let w_ic = wk[k];
                    acc += pk[0] * w_ic[0]
                        + pk[1] * w_ic[1]
                        + pk[2] * w_ic[2]
                        + pk[3] * w_ic[3];
                }
                dst_y[x] = (prelu(acc, L3_PRELU[0]) + residual_y).clamp(0.0, 1.0);
            }
        }

        ycbcr_to_rgb_into(
            &self.out_y,
            &self.cb_plane,
            &self.cr_plane,
            w,
            h,
            &mut buf.data,
        );
    }

    /// Zero-allocation 5-tap AMD FSR 1.0 RCAS contrast-adaptive sharpener using `self.work_px`.
    pub fn fsr_rcas_in_place(&mut self, buf: &mut PixelBuffer, sharpness: f32) {
        let w = buf.width;
        let h = buf.height;
        let num_px = w * h;
        if w == 0 || h == 0 || buf.data.len() < num_px {
            return;
        }
        let sharp = sharpness.clamp(0.0, 1.0);
        if sharp <= 1e-4 {
            return;
        }

        const RCAS_LIMIT: f32 = 0.1875;
        self.work_px.clear();
        self.work_px.extend_from_slice(&buf.data[..num_px]);
        let src = &self.work_px[..num_px];
        let max_x = w - 1;
        let max_y = h - 1;

        for y in 0..h {
            let ym1 = y.saturating_sub(1);
            let yp1 = (y + 1).min(max_y);

            let row_n = &src[ym1 * w..ym1 * w + w];
            let row_c = &src[y * w..y * w + w];
            let row_s = &src[yp1 * w..yp1 * w + w];
            let dst_row = &mut buf.data[y * w..(y + 1) * w];

            for x in 0..w {
                let xm1 = x.saturating_sub(1);
                let xp1 = (x + 1).min(max_x);

                let c = row_c[x];
                let n = row_n[x];
                let s = row_s[x];
                let w_px = row_c[xm1];
                let e = row_c[xp1];

                if c == n && c == s && c == w_px && c == e {
                    continue;
                }

                let lc = c.luma_fast();
                let ln = n.luma_fast();
                let ls = s.luma_fast();
                let lw = w_px.luma_fast();
                let le = e.luma_fast();

                let min4_u = ln.min(ls).min(lw).min(le);
                let max4_u = ln.max(ls).max(lw).max(le);
                let min5_u = min4_u.min(lc);
                let max5_u = max4_u.max(lc);

                // Integer-domain flat-patch suppression: skips 75%+ of pixels with zero float operations
                if max5_u.saturating_sub(min5_u) < 4 || (max5_u >= 251 && min5_u <= 4) {
                    continue;
                }

                let inv255 = 1.0 / 255.0;
                let ln_f = ln as f32 * inv255;
                let ls_f = ls as f32 * inv255;
                let lw_f = lw as f32 * inv255;
                let le_f = le as f32 * inv255;
                let lc_f = lc as f32 * inv255;
                let min4 = ln_f.min(ls_f).min(lw_f).min(le_f);
                let max4 = ln_f.max(ls_f).max(lw_f).max(le_f);
                let min5 = min4.min(lc_f);
                let max5 = max4.max(lc_f);

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

                dst_row[x] = Color::new(
                    (out_r + 0.5) as u8,
                    (out_g + 0.5) as u8,
                    (out_b + 0.5) as u8,
                );
            }
        }
    }

    /// Enhances a sub-rectangle `[sx0..sx1, sy0..sy1]` inside `subpixel_buf` according to its
    /// DOM-Guided Variable-Rate Shading `kind` (`0` = Photo/Raster, `1` = SVG/Vector/Icon, `2` = Video)
    /// and the active `UpscaleMode`, with zero heap allocations once `self.roi_crop` is warm.
    pub fn enhance_roi_subrect_in_place(
        &mut self,
        subpixel_buf: &mut PixelBuffer,
        sx0: usize,
        sy0: usize,
        sx1: usize,
        sy1: usize,
        kind: u8,
        mode: UpscaleMode,
    ) {
        let x0 = sx0.min(subpixel_buf.width);
        let y0 = sy0.min(subpixel_buf.height);
        let x1 = sx1.min(subpixel_buf.width);
        let y1 = sy1.min(subpixel_buf.height);
        if x0 >= x1 || y0 >= y1 {
            return;
        }
        let rw = x1 - x0;
        let rh = y1 - y0;
        if rw < 2 || rh < 2 {
            return;
        }

        // Copy ROI sub-rectangle into reusable `self.roi_crop`
        self.roi_crop.width = rw;
        self.roi_crop.height = rh;
        self.roi_crop.data.clear();
        self.roi_crop.data.reserve(rw * rh);
        for y in y0..y1 {
            let row_start = y * subpixel_buf.width + x0;
            self.roi_crop
                .data
                .extend_from_slice(&subpixel_buf.data[row_start..row_start + rw]);
        }

        // Temporarily swap out `roi_crop` so we can pass `&mut crop` to `self` methods without borrow conflict
        let mut crop = std::mem::replace(&mut self.roi_crop, PixelBuffer::new(0, 0, Color::BLACK));

        match kind {
            2 => {
                // Kind 2 (Active Video ROI): Single-pass subpixel RCAS at maximum framerate
                self.fsr_rcas_in_place(&mut crop, 0.82);
            }
            1 => {
                // Kind 1 (SVG / Vector / Icon ROI): Edge-preserving crisp RCAS pass without smoothing
                self.fsr_rcas_in_place(&mut crop, 0.90);
            }
            _ => {
                // Kind 0 (Photo / Raster ROI): Run selected UpscaleMode strictly on this ROI sub-rect
                match mode {
                    UpscaleMode::Off => {}
                    UpscaleMode::FsrOnly => {
                        self.fsr_rcas_in_place(&mut crop, 0.85);
                    }
                    UpscaleMode::NeuralOnly => {
                        self.refine_y_neural_1x_in_place(&mut crop);
                        self.fsr_rcas_in_place(&mut crop, 0.35);
                    }
                    UpscaleMode::Combined => {
                        self.refine_y_neural_1x_in_place(&mut crop);
                        self.fsr_rcas_in_place(&mut crop, 0.80);
                    }
                    UpscaleMode::MultiPass => {
                        self.remove_jpeg_artifacts_in_place(&mut crop);
                        self.refine_y_neural_1x_in_place(&mut crop);
                        self.fsr_rcas_in_place(&mut crop, 0.88);
                    }
                }
            }
        }

        // Write enhanced ROI sub-rectangle back into `subpixel_buf`
        for ry in 0..rh {
            let dst_start = (y0 + ry) * subpixel_buf.width + x0;
            let src_start = ry * rw;
            subpixel_buf.data[dst_start..dst_start + rw]
                .copy_from_slice(&crop.data[src_start..src_start + rw]);
        }

        self.roi_crop = crop;
    }
}

/// Runs Layers 1, 2, and 3 (3x3 Feature Extraction -> 1x1 Shrinking -> 3x3 Non-Linear Mapping
/// with Residual Skip Connection) in YCbCr space at 1x resolution `(w x h)` without 2x spatial expansion.
/// Designed specifically to run on the compact `(cols, rows * 2)` terminal subpixel grid (~5,000 pixels)
/// so full Y-channel neural edge & micro-contrast refinement completes in < 0.4ms.
pub fn refine_y_neural_1x(buf: &PixelBuffer) -> PixelBuffer {
    let mut out = buf.clone();
    let mut scratch = SrScratchpad::new();
    scratch.refine_y_neural_1x_in_place(&mut out);
    out
}

/// Applies the selected `UpscaleMode` enhancement (Y-channel Neural refinement / RCAS contrast
/// sharpening / JPEG Denoising) STRICTLY on the compact terminal subpixel grid `(cols, rows * 2)`
/// after downsampling from a full Chromium frame.
/// Reduces pixel processing count from ~800,000 pixels to ~5,000–11,000 subpixels per frame (98% CPU reduction).
#[inline]
pub fn apply_terminal_subpixel_rcas(subpixel_buf: &PixelBuffer, mode: UpscaleMode) -> PixelBuffer {
    let mut out = subpixel_buf.clone();
    let mut scratch = SrScratchpad::new();
    let w = out.width;
    let h = out.height;
    scratch.enhance_roi_subrect_in_place(&mut out, 0, 0, w, h, 0, mode);
    out
}

/// Fast-path single-pass subpixel sharpener (`fsr_rcas`) used when an active `<video>` element
/// is playing in the viewport, bypassing multi-layer CPU convolutions for tear-free video FPS.
#[inline]
pub fn apply_video_box_fast_rcas(subpixel_buf: &PixelBuffer) -> PixelBuffer {
    fsr_rcas(subpixel_buf, 0.82)
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
            let (fy, fcb, fcr) = bt601_rgb_to_ycbcr_fixed(c);
            let frec = bt601_ycbcr_to_rgb_fixed(fy, fcb, fcr);
            assert!(
                c.r.abs_diff(frec.r) <= 3
                    && c.g.abs_diff(frec.g) <= 3
                    && c.b.abs_diff(frec.b) <= 3,
                "BT.601 8-bit fixed-point error > 3 for {:?}",
                c
            );
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
        // Synthetic 16x16 photographic vertical blurred step-edge (> 64 unique colors so it uses continuous SR)
        let mut buf = PixelBuffer::new(16, 16, Color::new(51, 51, 51));
        for y in 0..16 {
            for x in 0..16 {
                let base: u8 = if x < 7 {
                    51
                } else if x == 7 {
                    107
                } else if x == 8 {
                    148
                } else {
                    204
                };
                // Add subtle per-row/col chroma variation so unique color count >= 64 (photographic image)
                let r = base.saturating_add((y as u8) * 2);
                let g = base;
                let b = base.saturating_sub((x as u8) * 2);
                buf.set_pixel(x, y, Color::new(r, g, b));
            }
        }

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
    fn test_recursive_chaining_4x_and_8x() {
        // Create an 8x8 photographic patch (> 64 colors) and test 4x (32x32) and 8x (64x64) recursive neural chaining
        let mut buf = PixelBuffer::new(8, 8, Color::BLACK);
        for y in 0..8 {
            for x in 0..8 {
                buf.set_pixel(
                    x,
                    y,
                    Color::new((x * 30 + 10) as u8, (y * 30 + 15) as u8, ((x + y) * 16) as u8),
                );
            }
        }

        for mode in [UpscaleMode::NeuralOnly, UpscaleMode::Combined, UpscaleMode::MultiPass] {
            let out_4x = apply_upscale_pipeline(&buf, mode, 32, 32);
            assert_eq!(out_4x.width, 32);
            assert_eq!(out_4x.height, 32);
            assert_eq!(out_4x.data.len(), 32 * 32);

            let out_8x = apply_upscale_pipeline(&buf, mode, 64, 64);
            assert_eq!(out_8x.width, 64);
            assert_eq!(out_8x.height, 64);
            assert_eq!(out_8x.data.len(), 64 * 64);
        }
    }

    #[test]
    fn test_jpeg_artifact_removal_and_edge_preservation() {
        // Construct a 16x16 image with synthetic 8x8 DCT block step artifact (100 vs 114 at x=8)
        // plus a sharp 1px high-contrast text stroke (value 10 on 114 background at x=12).
        let mut buf = PixelBuffer::new(16, 16, Color::new(100, 100, 100));
        for y in 0..16 {
            for x in 8..16 {
                buf.set_pixel(x, y, Color::new(114, 114, 114));
            }
            // High-contrast dark stroke at x = 12 (diff = 104 >> sigma_thresh)
            buf.set_pixel(12, y, Color::new(10, 10, 10));
        }

        assert!(
            has_jpeg_block_artifacts(&buf),
            "Should detect synthetic 8x8 JPEG block boundary jump"
        );

        let before_jump = buf.get_pixel(8, 4).r.abs_diff(buf.get_pixel(7, 4).r);
        remove_jpeg_artifacts(&mut buf);
        let after_jump = buf.get_pixel(8, 4).r.abs_diff(buf.get_pixel(7, 4).r);

        assert!(
            after_jump < before_jump,
            "8x8 JPEG block boundary jump should be smoothed: before={}, after={}",
            before_jump,
            after_jump
        );
        // High-contrast 1px stroke at x = 12 must remain crisp (unblurred)
        assert_eq!(buf.get_pixel(12, 4), Color::new(10, 10, 10));
    }

    #[test]
    fn test_pixel_art_detector_and_scale2x() {
        // 1. Construct a 16x16 4-color pixel-art sprite
        let bg = Color::new(20, 24, 46);
        let fg = Color::new(255, 80, 40);
        let accent = Color::new(250, 220, 60);
        let mut sprite = PixelBuffer::new(16, 16, bg);
        for y in 4..12 {
            for x in 4..12 {
                sprite.set_pixel(x, y, fg);
            }
        }
        sprite.set_pixel(5, 5, accent);
        sprite.set_pixel(10, 5, accent);

        assert!(
            detect_pixel_art(&sprite),
            "4-color sharp sprite should be classified as pixel art"
        );

        let scaled_2x = scale2x(&sprite);
        assert_eq!(scaled_2x.width, 32);
        assert_eq!(scaled_2x.height, 32);
        // Scale2x never invents interpolated blended colors; every output pixel must belong to the original palette!
        for px in &scaled_2x.data {
            assert!(
                *px == bg || *px == fg || *px == accent,
                "Scale2x invented non-palette color {:?}",
                px
            );
        }

        let scaled_4x = hq2x_scale(&sprite, 64, 64);
        assert_eq!(scaled_4x.width, 64);
        assert_eq!(scaled_4x.height, 64);
        for px in &scaled_4x.data {
            assert!(*px == bg || *px == fg || *px == accent);
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

