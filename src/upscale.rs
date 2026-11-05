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
    // Sub-pixel 0: Top-Left (biased toward top-left neighbors + negative Gx/Gy steering)
    [
        [ 0.045, -0.020, -0.020,  0.015],
        [ 0.135,  0.000, -0.040,  0.025],
        [-0.010,  0.010,  0.000,  0.000],
        [ 0.135, -0.040,  0.000,  0.025],
        [ 0.740, -0.140, -0.140,  0.110],
        [-0.020,  0.030,  0.000, -0.010],
        [-0.010,  0.000,  0.010,  0.000],
        [-0.020,  0.000,  0.030, -0.010],
        [ 0.005,  0.000,  0.000, -0.005],
    ],
    // Sub-pixel 1: Top-Right (biased toward top-right neighbors + positive Gx, negative Gy steering)
    [
        [-0.010, -0.010,  0.000,  0.000],
        [ 0.135,  0.000, -0.040,  0.025],
        [ 0.045,  0.020, -0.020,  0.015],
        [-0.020, -0.030,  0.000, -0.010],
        [ 0.740,  0.140, -0.140,  0.110],
        [ 0.135,  0.040,  0.000,  0.025],
        [ 0.005,  0.000,  0.000, -0.005],
        [-0.020,  0.000,  0.030, -0.010],
        [-0.010,  0.000,  0.010,  0.000],
    ],
    // Sub-pixel 2: Bottom-Left (biased toward bottom-left neighbors + negative Gx, positive Gy steering)
    [
        [-0.010,  0.000, -0.010,  0.000],
        [-0.020,  0.000, -0.030, -0.010],
        [ 0.005,  0.000,  0.000, -0.005],
        [ 0.135, -0.040,  0.000,  0.025],
        [ 0.740, -0.140,  0.140,  0.110],
        [-0.020,  0.030,  0.000, -0.010],
        [ 0.045, -0.020,  0.020,  0.015],
        [ 0.135,  0.000,  0.040,  0.025],
        [-0.010,  0.010,  0.000,  0.000],
    ],
    // Sub-pixel 3: Bottom-Right (biased toward bottom-right neighbors + positive Gx/Gy steering)
    [
        [ 0.005,  0.000,  0.000, -0.005],
        [-0.020,  0.000, -0.030, -0.010],
        [-0.010,  0.000, -0.010,  0.000],
        [-0.020, -0.030,  0.000, -0.010],
        [ 0.740,  0.140,  0.140,  0.110],
        [ 0.135,  0.040,  0.000,  0.025],
        [-0.010, -0.010,  0.000,  0.000],
        [ 0.135,  0.000,  0.040,  0.025],
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
pub fn upscale_y_neural_2x(y_plane: &[f32], w: usize, h: usize) -> Vec<f32> {
    if w == 0 || h == 0 || y_plane.len() < w * h {
        return Vec::new();
    }

    let num_px = w * h;

    // -------------------------------------------------------------------------
    // Layer 1 (Feature Extraction: 3x3 Conv, 1 -> 8 channels + PReLU)
    // combined with Layer 2 (FSRCNN Shrinking: 1x1 Conv, 8 -> 4 channels + PReLU)
    // Fused per-pixel so the 8-channel intermediate stays in CPU registers!
    // -------------------------------------------------------------------------
    let mut feat_l2 = vec![[0.0f32; 4]; num_px];
    let max_x = w - 1;
    let max_y = h - 1;

    for y in 0..h {
        let ym1 = y.saturating_sub(1);
        let yp1 = (y + 1).min(max_y);

        let r_top = &y_plane[ym1 * w..ym1 * w + w];
        let r_mid = &y_plane[y * w..y * w + w];
        let r_bot = &y_plane[yp1 * w..yp1 * w + w];
        let dst_row = &mut feat_l2[y * w..(y + 1) * w];

        for x in 0..w {
            let xm1 = x.saturating_sub(1);
            let xp1 = (x + 1).min(max_x);

            let patch = [
                r_top[xm1], r_top[x], r_top[xp1],
                r_mid[xm1], r_mid[x], r_mid[xp1],
                r_bot[xm1], r_bot[x], r_bot[xp1],
            ];

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
        let dst_row = &mut feat_l3[y * w..(y + 1) * w];

        for x in 0..w {
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
        let dst_row = &mut subpixels[y * w..(y + 1) * w];

        for x in 0..w {
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

            let lc = fast_luma_norm(c);
            let ln = fast_luma_norm(n);
            let ls = fast_luma_norm(s);
            let lw = fast_luma_norm(w_px);
            let le = fast_luma_norm(e);

            let min4 = ln.min(ls).min(lw).min(le);
            let max4 = ln.max(ls).max(lw).max(le);
            let min5 = min4.min(lc);
            let max5 = max4.max(lc);
            let contrast = max5 - min5;

            // Skip flat regions (<1.5% contrast) and already-saturated full-scale 1px borders
            // so we never introduce noise amplification or ringing around high-contrast lines.
            if contrast < 0.015 || (max5 > 0.985 && min5 < 0.015) {
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

/// Master Super-Resolution Dispatcher: processes `buf` according to the selected `UpscaleMode`.
///
/// - `UpscaleMode::Off`: Returns `buf` unchanged so the renderer uses fast `sample_subpixel_box`.
/// - `UpscaleMode::FsrOnly`: Runs AMD FSR 1.0 `fsr_easu` to `(target_w, target_h)` followed by `fsr_rcas`.
/// - `UpscaleMode::NeuralOnly`: Runs `upscale_neural_2x` (YCbCr Hybrid FSRCNN-ESPCN 2x pass).
/// - `UpscaleMode::Combined`: Runs `upscale_neural_2x` followed by `fsr_rcas` contrast sharpening.
/// - `UpscaleMode::MultiPass`: Runs `fsr_easu` spatial pass -> `upscale_neural_2x` detail pass -> `fsr_rcas` sharpening.
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
        UpscaleMode::NeuralOnly => upscale_neural_2x(buf),
        UpscaleMode::Combined => {
            let neural = upscale_neural_2x(buf);
            fsr_rcas(&neural, 0.80)
        }
        UpscaleMode::MultiPass => {
            // Stage 1: Edge-Adaptive Spatial Upsampling (1.5x intermediate or target_w/2 pre-scale)
            let mid_w = (out_w / 2).max(buf.width + (buf.width / 2)).min(buf.width * 2);
            let mid_h = (out_h / 2).max(buf.height + (buf.height / 2)).min(buf.height * 2);
            let easu = fsr_easu(buf, mid_w, mid_h);
            // Stage 2: YCbCr Hybrid FSRCNN-ESPCN 2x sub-pixel detail synthesis
            let neural = upscale_neural_2x(&easu);
            // Stage 3: Robust Contrast-Adaptive Sharpening
            fsr_rcas(&neural, 0.85)
        }
    }
}

/// Alias entrypoint for the renderer pipeline: runs `upscale_frame` before subpixel sampling or protocol encoding.
#[inline]
pub fn apply_upscale_pipeline(
    buf: &PixelBuffer,
    mode: UpscaleMode,
    target_w: usize,
    target_h: usize,
) -> PixelBuffer {
    upscale_frame(buf, mode, target_w, target_h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ycbcr_roundtrip() {
        let mut buf = PixelBuffer::new(4, 4, Color::new(26, 115, 232));
        buf.set_pixel(1, 1, Color::WHITE);
        buf.set_pixel(2, 2, Color::BLACK);
        buf.set_pixel(3, 0, Color::new(255, 102, 0));

        let (y, cb, cr) = rgb_to_ycbcr(&buf);
        let reconstructed = ycbcr_to_rgb(&y, &cb, &cr, 4, 4);

        for i in 0..buf.data.len() {
            let orig = buf.data[i];
            let rec = reconstructed.data[i];
            assert!(orig.r.abs_diff(rec.r) <= 1);
            assert!(orig.g.abs_diff(rec.g) <= 1);
            assert!(orig.b.abs_diff(rec.b) <= 1);
        }
    }

    #[test]
    fn test_dc_invariance_no_checkerboard() {
        let c = Color::new(200, 210, 220);
        let buf = PixelBuffer::new(8, 8, c);
        let up = upscale_neural_2x(&buf);

        assert_eq!(up.width, 16);
        assert_eq!(up.height, 16);

        // Every upscaled pixel on a flat canvas must match the original color within +-1 rounding
        for px in &up.data {
            assert!(px.r.abs_diff(c.r) <= 1);
            assert!(px.g.abs_diff(c.g) <= 1);
            assert!(px.b.abs_diff(c.b) <= 1);
        }
    }

    #[test]
    fn test_step_edge_sharpening() {
        let mut y_plane = vec![0.2f32; 8 * 8];
        for y in 0..8 {
            for x in 4..8 {
                y_plane[y * 8 + x] = 0.8f32;
            }
        }
        let up_y = upscale_y_neural_2x(&y_plane, 8, 8);
        assert_eq!(up_y.len(), 16 * 16);

        // Sub-pixel right before the step edge (col 7) and right after (col 8) should maintain high contrast
        let left_edge = up_y[8 * 16 + 7];
        let right_edge = up_y[8 * 16 + 8];
        assert!(right_edge - left_edge > 0.45);
    }

    #[test]
    fn test_fsr_easu_anti_ringing_and_dimensions() {
        let mut buf = PixelBuffer::new(6, 6, Color::new(40, 80, 120));
        for y in 0..6 {
            for x in 3..6 {
                buf.set_pixel(x, y, Color::new(200, 220, 240));
            }
        }
        let easu = fsr_easu(&buf, 12, 12);
        assert_eq!(easu.width, 12);
        assert_eq!(easu.height, 12);

        // Verify anti-ringing clamping: no pixel may undershoot 40 or overshoot 240
        for px in &easu.data {
            assert!(px.r >= 40 && px.r <= 200);
            assert!(px.g >= 80 && px.g <= 220);
            assert!(px.b >= 120 && px.b <= 240);
        }
    }

    #[test]
    fn test_fsr_rcas_preserves_flat_and_sharpens_edges() {
        let mut buf = PixelBuffer::new(6, 6, Color::new(128, 128, 128));
        // Flat region should remain strictly untouched
        let flat_rcas = fsr_rcas(&buf, 0.9);
        assert_eq!(flat_rcas.get_pixel(3, 3), Color::new(128, 128, 128));

        // Soft edge: center is brighter than neighbors
        buf.set_pixel(3, 3, Color::new(180, 180, 180));
        let edge_rcas = fsr_rcas(&buf, 0.9);
        assert!(edge_rcas.get_pixel(3, 3).r > 180);
    }

    #[test]
    fn test_upscale_mode_cycle_and_dispatcher() {
        assert_eq!(UpscaleMode::Off.next(), UpscaleMode::FsrOnly);
        assert_eq!(UpscaleMode::FsrOnly.next(), UpscaleMode::NeuralOnly);
        assert_eq!(UpscaleMode::NeuralOnly.next(), UpscaleMode::Combined);
        assert_eq!(UpscaleMode::Combined.next(), UpscaleMode::MultiPass);
        assert_eq!(UpscaleMode::MultiPass.next(), UpscaleMode::Off);

        let buf = PixelBuffer::new(4, 4, Color::new(100, 150, 200));
        let cmb = upscale_frame(&buf, UpscaleMode::Combined, 8, 8);
        assert_eq!(cmb.width, 8);
        assert_eq!(cmb.height, 8);
    }
}
