#![allow(dead_code)]
#[allow(unused_imports)]
use std::io::{self, Write};
use crate::upscale::{
    apply_terminal_subpixel_rcas, apply_upscale_pipeline_ex, detect_pixel_art, SrScratchpad,
    UpscaleMode,
};

/// Hardware-inspired Selective Region-of-Interest (`ROI:x,y,w,h,kind`) emitted by the DOM extractor:
/// - `kind == 0`: Photo / Raster (`<img>`, `<canvas>`, `background-image: url(...)`) -> Neural + Denoise + RCAS + 4x4 Bayer Dither
/// - `kind == 1`: Vector / SVG / Icon (`<svg>`, crisp vector graphic) -> Scale2x / HQ2x / RCAS edge-preserving pass without smoothing
/// - `kind == 2`: Active `<video>` element -> Suppress DOM text overlays inside box + single-pass `fsr_rcas` at 60 FPS
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssetRoi {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub kind: u8,
}

impl AssetRoi {
    /// Parses a `ROI:x,y,w,h,kind;x2,y2,w2,h2,kind2` payload from the CDP `META` stream.
    pub fn parse_list(raw: &str) -> Vec<Self> {
        let trimmed = raw.trim();
        let body = trimmed.strip_prefix("ROI:").unwrap_or(trimmed);
        if body.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        for item in body.split(';') {
            let nums: Vec<i32> = item
                .split(',')
                .filter_map(|s| s.trim().parse::<i32>().ok())
                .collect();
            if nums.len() >= 5 && nums[2] >= 16 && nums[3] >= 16 {
                out.push(Self {
                    x: nums[0],
                    y: nums[1],
                    w: nums[2],
                    h: nums[3],
                    kind: nums[4].clamp(0, 2) as u8,
                });
            }
        }
        out
    }

    /// Maps this CSS viewport ROI box into terminal subpixel coordinates `(sx0, sy0, sx1, sy1)`
    /// inside a `(sub_w, target_h)` subpixel buffer using Q16.16 fixed-point integer math.
    #[inline]
    pub fn to_subpixel_bounds(
        &self,
        sub_w: usize,
        target_h: usize,
        vw: u32,
        vh: u32,
    ) -> (usize, usize, usize, usize) {
        let pw = vw.max(1) as i64;
        let ph = vh.max(1) as i64;
        let sw = sub_w as i64;
        let th = target_h as i64;

        let x0 = ((self.x.max(0) as i64 * sw) / pw).clamp(0, sw) as usize;
        let x1 = (((self.x.saturating_add(self.w)).max(0) as i64 * sw + pw - 1) / pw)
            .clamp(0, sw) as usize;
        let y0 = ((self.y.max(0) as i64 * th) / ph).clamp(0, th) as usize;
        let y1 = (((self.y.saturating_add(self.h)).max(0) as i64 * th + ph - 1) / ph)
            .clamp(0, th) as usize;
        (x0, y0, x1, y1)
    }

    /// Maps this CSS viewport ROI box into terminal character cell coordinates `(c0, r0, c1, r1)`
    /// where `row` is in `1..rows` (excluding the top status bar row 0).
    #[inline]
    pub fn to_terminal_bounds(
        &self,
        cols: u16,
        rows: u16,
        vw: u32,
        vh: u32,
    ) -> (u16, u16, u16, u16) {
        let pw = vw.max(1) as i64;
        let ph = vh.max(1) as i64;
        let view_start_y = 1i64;
        let view_rows = (rows.saturating_sub(1)).max(1) as i64;
        let cw = cols as i64;

        let c0 = ((self.x.max(0) as i64 * cw) / pw).clamp(0, cw) as u16;
        let c1 = (((self.x.saturating_add(self.w)).max(0) as i64 * cw + pw - 1) / pw)
            .clamp(0, cw) as u16;
        let r0 = (view_start_y + ((self.y.max(0) as i64 * view_rows) / ph))
            .clamp(1, rows as i64) as u16;
        let r1 = (view_start_y
            + (((self.y.saturating_add(self.h)).max(0) as i64 * view_rows + ph - 1) / ph))
            .clamp(1, rows as i64) as u16;
        (c0, r0, c1, r1)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Color {
    pub const BLACK: Color = Color { r: 0, g: 0, b: 0 };
    pub const WHITE: Color = Color { r: 255, g: 255, b: 255 };
    pub const DARK_GRAY: Color = Color { r: 40, g: 40, b: 40 };
    pub const GRAY: Color = Color { r: 160, g: 160, b: 160 };
    pub const LIGHT_GRAY: Color = Color { r: 210, g: 210, b: 210 };
    pub const CYAN: Color = Color { r: 60, g: 180, b: 230 };
    pub const BLUE: Color = Color { r: 80, g: 140, b: 255 };
    pub const YELLOW: Color = Color { r: 240, g: 200, b: 60 };
    pub const GREEN: Color = Color { r: 60, g: 200, b: 100 };
    pub const RED: Color = Color { r: 240, g: 80, b: 80 };
    pub const MAGENTA: Color = Color { r: 220, g: 100, b: 200 };

    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    pub fn from_hex(hex: &str) -> Option<Self> {
        let hex = hex.trim_start_matches('#');
        if hex.len() == 6 {
            let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
            let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
            let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
            Some(Self::new(r, g, b))
        } else if hex.len() == 3 {
            let r = u8::from_str_radix(&hex[0..1], 16).ok()? * 17;
            let g = u8::from_str_radix(&hex[1..2], 16).ok()? * 17;
            let b = u8::from_str_radix(&hex[2..3], 16).ok()? * 17;
            Some(Self::new(r, g, b))
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn luma_fast(self) -> u32 {
        (self.r as u32 * 77 + self.g as u32 * 150 + self.b as u32 * 29) >> 8
    }
}

/// High-performance integer luminance approximation: `(R*77 + G*150 + B*29) >> 8`
#[inline(always)]
pub fn luma_fast_u32(c: Color) -> u32 {
    (c.r as u32 * 77 + c.g as u32 * 150 + c.b as u32 * 29) >> 8
}

/// Averages two colors using fast integer ALU bit-shifts without overflow.
#[inline(always)]
pub fn avg_color(c1: Color, c2: Color) -> Color {
    Color::new(
        ((c1.r as u16 + c2.r as u16) >> 1) as u8,
        ((c1.g as u16 + c2.g as u16) >> 1) as u8,
        ((c1.b as u16 + c2.b as u16) >> 1) as u8,
    )
}

/// Maps a 24-bit RGB `Color` to the nearest xterm-256 palette index (`16..=255`).
#[inline(always)]
pub fn rgb_to_xterm256(c: Color) -> u8 {
    if c.r.abs_diff(c.g) <= 2 && c.g.abs_diff(c.b) <= 2 {
        if c.r < 8 {
            return 16;
        }
        if c.r > 248 {
            return 231;
        }
        return (((c.r as u16 - 8) * 24) / 240) as u8 + 232;
    }
    let r = ((c.r as u16 * 5 + 127) / 255) as u8;
    let g = ((c.g as u16 * 5 + 127) / 255) as u8;
    let b = ((c.b as u16 * 5 + 127) / 255) as u8;
    16 + 36 * r + 6 * g + b
}

/// Returns true if `ch` is a half-block or 2x2 quadrant block glyph used for subpixel rendering.
#[inline(always)]
pub fn is_quadrant_block_char(ch: char) -> bool {
    matches!(
        ch,
        '▀' | '▄' | '▌' | '▐' | '▖' | '▗' | '▘' | '▙' | '▚' | '▛' | '▜' | '▝' | '▞' | '▟' | '█'
    )
}

/// Carbonyl-grade 2x2 Quadrant Binarizer:
/// Maps 4 subpixel quadrant colors `(tl, tr, bl, br)` into `(Unicode char, Foreground, Background)`
/// using integer luminance `(R*77 + G*150 + B*29) >> 8` and dynamic midpoint thresholding.
#[inline]
pub fn binarize_quadrant(tl: Color, tr: Color, bl: Color, br: Color) -> (char, Color, Color) {
    let a = luma_fast_u32(tl);
    let b = luma_fast_u32(tr);
    let c = luma_fast_u32(bl);
    let d = luma_fast_u32(br);

    let min = a.min(b).min(c).min(d);
    let max = a.max(b).max(c).max(d);

    // Flat or sub-perceptual contrast cell: emit clean space with averaged background
    if max.saturating_sub(min) < 6 {
        return (' ', Color::WHITE, avg_color(avg_color(tl, tr), avg_color(bl, br)));
    }

    let mid = min + (max - min) / 2;
    let mask = ((a > mid) as u8) << 3
        | ((b > mid) as u8) << 2
        | ((c > mid) as u8) << 1
        | ((d > mid) as u8);

    match mask {
        0b0000 => (' ', Color::WHITE, avg_color(avg_color(tl, tr), avg_color(bl, br))),
        0b0001 => ('▗', br, avg_color(avg_color(tl, tr), bl)),
        0b0010 => ('▖', bl, avg_color(avg_color(tl, tr), br)),
        0b0011 => ('▄', avg_color(bl, br), avg_color(tl, tr)),
        0b0100 => ('▝', tr, avg_color(avg_color(tl, bl), br)),
        0b0101 => ('▐', avg_color(tr, br), avg_color(tl, bl)),
        0b0110 => ('▞', avg_color(tr, bl), avg_color(tl, br)),
        0b0111 => ('▟', avg_color(avg_color(tr, bl), br), tl),
        0b1000 => ('▘', tl, avg_color(avg_color(tr, bl), br)),
        0b1001 => ('▚', avg_color(tl, br), avg_color(tr, bl)),
        0b1010 => ('▌', avg_color(tl, bl), avg_color(tr, br)),
        0b1011 => ('▙', avg_color(avg_color(tl, bl), br), tr),
        0b1100 => ('▀', avg_color(tl, tr), avg_color(bl, br)),
        0b1101 => ('▜', avg_color(avg_color(tl, tr), br), bl),
        0b1110 => ('▛', avg_color(avg_color(tl, tr), bl), br),
        0b1111 => ('▀', avg_color(tl, tr), avg_color(bl, br)),
        _ => (' ', Color::WHITE, avg_color(avg_color(tl, tr), avg_color(bl, br))),
    }
}

/// Returns true if `c` is a zero-width Unicode combining mark, Indic vowel matra / virama / nukta / anusvara,
/// zero-width joiner, or variation selector that must attach to the previous terminal cell rather than
/// advancing to a new column (which would otherwise render dotted circles `◌` in Indic/complex scripts).
#[inline]
pub fn is_combining_char(c: char) -> bool {
    let cp = c as u32;
    if cp < 0x0300 {
        return false;
    }
    // Combining Diacritical Marks & Zero-Width Joiners / Variation Selectors
    if matches!(
        cp,
        0x0300..=0x036F
            | 0x0483..=0x0489
            | 0x0591..=0x05BD
            | 0x05BF
            | 0x05C1..=0x05C2
            | 0x05C4..=0x05C5
            | 0x05C7
            | 0x0610..=0x061A
            | 0x064B..=0x065F
            | 0x0670
            | 0x06D6..=0x06ED
            | 0x0E31
            | 0x0E34..=0x0E3A
            | 0x0E47..=0x0E4E
            | 0x1AB0..=0x1AFF
            | 0x1DC0..=0x1DFF
            | 0x200B..=0x200F
            | 0x20D0..=0x20FF
            | 0xFE00..=0xFE0F
            | 0xFE20..=0xFE2F
            | 0xFEFF
    ) {
        return true;
    }

    // All 9 Unicode Indic script blocks (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam)
    // share an identical 128-codepoint relative layout within 0x0900..=0x0D7F.
    if (0x0900..=0x0D7F).contains(&cp) {
        let offset = cp & 0x7F;
        if matches!(
            offset,
            0x00..=0x03 // Candrabindu, Anusvara, Visarga
                | 0x3A..=0x3C // Nukta & vowel signs
                | 0x3E..=0x4F // Dependent vowel matras (aa, i, ii, u, uu, r, e, ai, o, au) & Virama/Halant
                | 0x51..=0x57 // Vedic tone & length marks
                | 0x62..=0x63 // Vocalic L/LL vowel signs
                | 0x70..=0x71 // Gurmukhi Tippi / Addak
        ) {
            return true;
        }
    }

    false
}

/// Computes the visual terminal column width of a string (ignoring zero-width combining marks).
#[inline]
pub fn display_width(s: &str) -> usize {
    s.chars().filter(|&c| !is_combining_char(c)).count()
}

/// Returns true if a string contains Indic or combining characters where a terminal underline
/// would visually cut through subscript vowel matras or viramas.
#[inline]
pub fn has_complex_script(s: &str) -> bool {
    s.chars().any(|c| (c as u32) >= 0x0900 && is_combining_char(c))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cell {
    pub ch: char,
    pub comb: [char; 3],
    pub fg: Color,
    pub bg: Color,
    pub bold: bool,
    pub underline: bool,
}

impl Cell {
    pub const EMPTY: Cell = Cell {
        ch: ' ',
        comb: ['\0'; 3],
        fg: Color::WHITE,
        bg: Color::BLACK,
        bold: false,
        underline: false,
    };

    pub const INVALID: Cell = Cell {
        ch: '\0',
        comb: ['\0'; 3],
        fg: Color::BLACK,
        bg: Color::BLACK,
        bold: false,
        underline: false,
    };

    pub fn new(ch: char, fg: Color, bg: Color) -> Self {
        Self {
            ch,
            comb: ['\0'; 3],
            fg,
            bg,
            bold: false,
            underline: false,
        }
    }

    #[inline(always)]
    pub fn push_combining(&mut self, c: char) {
        for slot in &mut self.comb {
            if *slot == '\0' {
                *slot = c;
                break;
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct PixelBuffer {
    pub width: usize,
    pub height: usize,
    pub data: Vec<Color>,
}

impl PixelBuffer {
    pub fn new(width: usize, height: usize, bg: Color) -> Self {
        Self {
            width,
            height,
            data: vec![bg; width * height],
        }
    }

    pub fn set_pixel(&mut self, x: usize, y: usize, color: Color) {
        if x < self.width && y < self.height {
            self.data[y * self.width + x] = color;
        }
    }

    #[inline(always)]
    pub fn get_pixel(&self, x: usize, y: usize) -> Color {
        if x < self.width && y < self.height {
            self.data[y * self.width + x]
        } else {
            Color::BLACK
        }
    }

    pub fn from_image(img: &image::DynamicImage) -> Self {
        use image::GenericImageView;
        let (w, h) = img.dimensions();
        let len = (w as usize) * (h as usize);
        let mut data = Vec::with_capacity(len);
        if let Some(rgb) = img.as_rgb8() {
            for chunk in rgb.as_raw().chunks_exact(3) {
                data.push(Color::new(chunk[0], chunk[1], chunk[2]));
            }
        } else {
            let rgb = img.to_rgb8();
            for chunk in rgb.as_raw().chunks_exact(3) {
                data.push(Color::new(chunk[0], chunk[1], chunk[2]));
            }
        }
        Self {
            width: w as usize,
            height: h as usize,
            data,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphicsProtocol {
    HalfBlockAnsi,
    Kitty,
    Sixel,
}

impl GraphicsProtocol {
    /// Probes terminal emulator capabilities across Android Termux, Linux, macOS, and Windows
    pub fn auto_detect() -> Self {
        if std::env::var("KITTY_WINDOW_ID").is_ok() {
            return GraphicsProtocol::Kitty;
        }
        if let Ok(term) = std::env::var("TERM") {
            let t = term.to_lowercase();
            if t.contains("kitty") {
                return GraphicsProtocol::Kitty;
            }
        }
        if let Ok(prog) = std::env::var("TERM_PROGRAM") {
            let p = prog.to_lowercase();
            if p == "ghostty" || p == "kitty" || p == "wezterm" || p == "iterm.app" {
                return GraphicsProtocol::Kitty;
            }
        }

        if let Ok(term) = std::env::var("TERM") {
            let t = term.to_lowercase();
            if t.contains("sixel") || t.contains("mlterm") || t == "foot" || t == "foot-extra" {
                return GraphicsProtocol::Sixel;
            }
        }
        if let Ok(prog) = std::env::var("TERM_PROGRAM") {
            let p = prog.to_lowercase();
            if p == "foot" || p == "mintty" || p == "contour" {
                return GraphicsProtocol::Sixel;
            }
        }

        GraphicsProtocol::HalfBlockAnsi
    }

    pub fn name(self) -> &'static str {
        match self {
            GraphicsProtocol::HalfBlockAnsi => "Half-Block (ANSI)",
            GraphicsProtocol::Kitty => "Kitty Graphics",
            GraphicsProtocol::Sixel => "Sixel Graphics",
        }
    }

    pub fn short_name(self) -> &'static str {
        match self {
            GraphicsProtocol::HalfBlockAnsi => "ANSI",
            GraphicsProtocol::Kitty => "Kitty",
            GraphicsProtocol::Sixel => "Sixel",
        }
    }
}

pub fn base64_encode(data: &[u8]) -> String {
    const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len() * 4 / 3 + 4);
    let mut i = 0;
    while i + 3 <= data.len() {
        let b0 = data[i] as usize;
        let b1 = data[i + 1] as usize;
        let b2 = data[i + 2] as usize;
        out.push(B64[b0 >> 2] as char);
        out.push(B64[((b0 & 3) << 4) | (b1 >> 4)] as char);
        out.push(B64[((b1 & 15) << 2) | (b2 >> 6)] as char);
        out.push(B64[b2 & 63] as char);
        i += 3;
    }
    if i < data.len() {
        let rem = data.len() - i;
        let b0 = data[i] as usize;
        out.push(B64[b0 >> 2] as char);
        if rem == 1 {
            out.push(B64[(b0 & 3) << 4] as char);
            out.push('=');
            out.push('=');
        } else {
            let b1 = data[i + 1] as usize;
            out.push(B64[((b0 & 3) << 4) | (b1 >> 4)] as char);
            out.push(B64[(b1 & 15) << 2] as char);
            out.push('=');
        }
    }
    out
}

#[inline(always)]
fn push_u16(buf: &mut Vec<u8>, mut n: u16) {
    if n == 0 {
        buf.push(b'0');
        return;
    }
    let mut tmp = [0u8; 5];
    let mut i = 0;
    while n > 0 {
        tmp[i] = b'0' + (n % 10) as u8;
        n /= 10;
        i += 1;
    }
    while i > 0 {
        i -= 1;
        buf.push(tmp[i]);
    }
}

#[inline(always)]
fn push_u8(buf: &mut Vec<u8>, mut n: u8) {
    if n >= 100 {
        buf.push(b'0' + n / 100);
        n %= 100;
        buf.push(b'0' + n / 10);
        n %= 10;
    } else if n >= 10 {
        buf.push(b'0' + n / 10);
        n %= 10;
    }
    buf.push(b'0' + n);
}

#[inline]
fn write_stdout_bytes(buf: &[u8]) -> io::Result<()> {
    if buf.is_empty() {
        return Ok(());
    }
    #[cfg(unix)]
    unsafe {
        let mut offset = 0;
        while offset < buf.len() {
            let ptr = buf[offset..].as_ptr() as *const libc::c_void;
            let written = libc::write(libc::STDOUT_FILENO, ptr, buf.len() - offset);
            if written < 0 {
                let err = io::Error::last_os_error();
                if err.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(err);
            }
            if written == 0 {
                break;
            }
            offset += written as usize;
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        let stdout = io::stdout();
        let mut handle = stdout.lock();
        handle.write_all(buf)?;
        handle.flush()
    }
}

pub struct DiffRenderer {
    pub width: u16,
    pub height: u16,
    pub protocol: GraphicsProtocol,
    pub upscale_mode: UpscaleMode,
    pub dithering_enabled: bool,
    pub video_fast_path: bool,
    pub scroll_fast_path: bool,
    pub pixel_art_mode: bool,
    pub fast_color_mode: bool,
    pub active_rois: Vec<AssetRoi>,
    pub roi_viewport: (u32, u32),
    pub vrs_roi_enabled: bool,
    pub scratch: SrScratchpad,
    subpixel_buf: PixelBuffer,
    roi_cell_mask: Vec<u8>,
    current: Vec<Cell>,
    previous: Vec<Cell>,
    cached_bg_cells: Vec<Cell>,
    cached_frame_sig: Option<(usize, u16, u16, UpscaleMode, bool, bool, bool, usize)>,
    write_buf: Vec<u8>,
}

impl DiffRenderer {
    pub fn new(width: u16, height: u16) -> Self {
        let size = (width as usize) * (height as usize);
        let fast_color_mode = std::env::var("TERMBLITZ_FAST_COLOR")
            .or_else(|_| std::env::var("CARBONYL_FAST_COLOR"))
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        Self {
            width,
            height,
            protocol: GraphicsProtocol::auto_detect(),
            upscale_mode: UpscaleMode::FsrOnly,
            dithering_enabled: true,
            video_fast_path: false,
            scroll_fast_path: false,
            pixel_art_mode: false,
            fast_color_mode,
            active_rois: Vec::with_capacity(16),
            roi_viewport: (360, 576),
            vrs_roi_enabled: false,
            scratch: SrScratchpad::new(),
            subpixel_buf: PixelBuffer::new((width as usize) * 2, (height as usize) * 2, Color::BLACK),
            roi_cell_mask: vec![0u8; size],
            current: vec![Cell::EMPTY; size],
            previous: vec![Cell::INVALID; size],
            cached_bg_cells: Vec::new(),
            cached_frame_sig: None,
            write_buf: Vec::with_capacity(65536),
        }
    }

    pub fn resize(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;
        let size = (width as usize) * (height as usize);
        self.current.clear();
        self.current.resize(size, Cell::EMPTY);
        self.previous.clear();
        self.previous.resize(size, Cell::INVALID);
        self.roi_cell_mask.clear();
        self.roi_cell_mask.resize(size, 0u8);
        self.cached_bg_cells.clear();
        self.cached_frame_sig = None;
        self.write_buf.clear();
    }

    /// Updates the DOM-Guided Variable-Rate Shading (VRS) Regions of Interest (`ROI:x,y,w,h,kind`)
    /// and enables selective ROI gating for subsequent frame blits.
    #[inline]
    pub fn set_active_rois(&mut self, rois: Vec<AssetRoi>, vw: u32, vh: u32) {
        self.active_rois = rois;
        self.roi_viewport = (vw.max(1), vh.max(1));
        self.vrs_roi_enabled = true;
    }

    /// Invalidates only the cached background frame signature (e.g., when transitioning from
    /// Gear 1 Kinetic Motion Mode to Gear 2 High-Precision Rest Mode for a Quality Settling Pass).
    #[inline]
    pub fn invalidate_frame_cache(&mut self) {
        self.cached_frame_sig = None;
    }

    pub fn force_redraw(&mut self) {
        self.previous.fill(Cell::INVALID);
        self.cached_frame_sig = None;
    }

    pub fn clear(&mut self, bg: Color) {
        let empty = Cell {
            ch: ' ',
            comb: ['\0'; 3],
            fg: Color::WHITE,
            bg,
            bold: false,
            underline: false,
        };
        self.current.fill(empty);
    }

    #[inline(always)]
    pub fn set_cell(&mut self, x: u16, y: u16, cell: Cell) {
        if x < self.width && y < self.height {
            let idx = (y as usize) * (self.width as usize) + (x as usize);
            self.current[idx] = cell;
        }
    }

    pub fn put_str(&mut self, x: u16, y: u16, s: &str, fg: Color, bg: Color, bold: bool, underline: bool) {
        if y >= self.height {
            return;
        }
        let mut curr_x = x;
        let mut last_idx: Option<usize> = None;

        for ch in s.chars() {
            if is_combining_char(ch) {
                if let Some(prev_idx) = last_idx {
                    self.current[prev_idx].push_combining(ch);
                }
                continue;
            }
            if curr_x >= self.width {
                break;
            }
            let idx = (y as usize) * (self.width as usize) + (curr_x as usize);
            self.current[idx] = Cell {
                ch,
                comb: ['\0'; 3],
                fg,
                bg,
                bold,
                underline,
            };
            last_idx = Some(idx);
            curr_x += 1;
        }
    }

    /// Prints a string directly over existing ANSI cells with uniform background resolution,
    /// full Unicode combining-character attachment, and guaranteed foreground/background contrast.
    #[allow(clippy::too_many_arguments)]
    pub fn put_str_with_existing_bg(
        &mut self,
        x: u16,
        y: u16,
        s: &str,
        fg: Color,
        explicit_bg: Option<Color>,
        bold: bool,
        underline: bool,
    ) {
        if y >= self.height || x >= self.width {
            return;
        }

        // Resolve a single uniform background color for the span if CSS didn't specify one,
        // avoiding patchy per-letter background cells.
        let uniform_bg = explicit_bg.unwrap_or_else(|| {
            let sample_idx = (y as usize) * (self.width as usize) + (x as usize);
            if let Some(cell) = self.current.get(sample_idx) {
                if is_quadrant_block_char(cell.ch) {
                    let lum_fg = cell.fg.luma_fast();
                    let lum_bg = cell.bg.luma_fast();
                    if fg.luma_fast() < 128 {
                        if lum_fg >= lum_bg { cell.fg } else { cell.bg }
                    } else if lum_fg <= lum_bg {
                        cell.fg
                    } else {
                        cell.bg
                    }
                } else {
                    cell.bg
                }
            } else {
                Color::WHITE
            }
        });

        let mut curr_x = x;
        let mut last_idx: Option<usize> = None;

        for ch in s.chars() {
            if is_combining_char(ch) {
                if let Some(prev_idx) = last_idx {
                    self.current[prev_idx].push_combining(ch);
                }
                continue;
            }
            if curr_x >= self.width {
                break;
            }
            let idx = (y as usize) * (self.width as usize) + (curr_x as usize);
            if idx < self.current.len() {
                let cell = &self.current[idx];
                let clean_bg = if explicit_bg.is_some() {
                    uniform_bg
                } else if is_quadrant_block_char(cell.ch) {
                    let lum_fg = cell.fg.luma_fast();
                    let lum_bg = cell.bg.luma_fast();
                    let best = if fg.luma_fast() < 128 {
                        if lum_fg >= lum_bg { cell.fg } else { cell.bg }
                    } else if lum_fg <= lum_bg {
                        cell.fg
                    } else {
                        cell.bg
                    };
                    // Snap near-white background noise to crisp uniform background
                    if best.luma_fast() > 220 && uniform_bg.luma_fast() > 220 {
                        uniform_bg
                    } else {
                        best
                    }
                } else if cell.bg.luma_fast() > 220 && uniform_bg.luma_fast() > 220 {
                    uniform_bg
                } else if cell.bg != Color::BLACK || cell.ch == ' ' {
                    cell.bg
                } else {
                    uniform_bg
                };

                let bg_lum = clean_bg.luma_fast();
                let fg_lum = fg.luma_fast();
                let contrast = bg_lum.abs_diff(fg_lum);
                let effective_fg = if contrast < 70 {
                    if bg_lum > 128 {
                        Color::new(18, 20, 26)
                    } else {
                        Color::new(248, 250, 252)
                    }
                } else {
                    fg
                };

                self.current[idx] = Cell {
                    ch,
                    comb: ['\0'; 3],
                    fg: effective_fg,
                    bg: clean_bg,
                    bold,
                    underline,
                };
                last_idx = Some(idx);
            }
            curr_x += 1;
        }
    }

    /// Clears stale underlying page text and background noise inside an elevated popup/dropdown
    /// bounding box (`POPUP_MASK:x,y,w,h,r,g,b` / `RECT_MASK:x,y,w,h,r,g,b`) so autocomplete suggestions
    /// and modal menus render cleanly on top of a solid surface without colliding with underlying page text.
    pub fn fill_rect_mask(&mut self, c0: u16, r0: u16, c1: u16, r1: u16, bg: Color) {
        let x_start = c0.min(self.width);
        let x_end = c1.min(self.width);
        let y_start = r0.min(self.height);
        let y_end = r1.min(self.height);
        if x_start >= x_end || y_start >= y_end {
            return;
        }
        let clean_bg = mask_jpeg_noise_6bit(bg);
        let border_fg = if clean_bg.luma_fast() > 128 {
            Color::new(
                clean_bg.r.saturating_sub(28),
                clean_bg.g.saturating_sub(28),
                clean_bg.b.saturating_sub(24),
            )
        } else {
            Color::new(
                clean_bg.r.saturating_add(36),
                clean_bg.g.saturating_add(36),
                clean_bg.b.saturating_add(40),
            )
        };

        for y in y_start..y_end {
            let is_bottom_edge = y + 1 == y_end && (y_end - y_start) >= 2;
            for x in x_start..x_end {
                let ch = if is_bottom_edge { '▁' } else { ' ' };
                self.set_cell(
                    x,
                    y,
                    Cell {
                        ch,
                        comb: ['\0'; 3],
                        fg: border_fg,
                        bg: clean_bg,
                        bold: false,
                        underline: false,
                    },
                );
            }
        }
    }

    /// Alias for `fill_rect_mask` used by `POPUP_MASK` occlusion handlers.
    #[inline]
    pub fn fill_popup_mask(&mut self, c0: u16, r0: u16, c1: u16, r1: u16, bg: Color) {
        self.fill_rect_mask(c0, r0, c1, r1, bg);
    }

    /// High-level blitter: automatically dispatches to Kitty, Sixel, or Half-Block ANSI
    pub fn blit_frame(&mut self, px_buf: &PixelBuffer, start_y: u16, max_y: u16) {
        match self.protocol {
            GraphicsProtocol::Kitty => {
                self.blit_kitty(px_buf, start_y, max_y);
            }
            GraphicsProtocol::Sixel => {
                self.blit_sixel(px_buf, start_y, max_y);
            }
            GraphicsProtocol::HalfBlockAnsi => {
                self.blit_pixel_buffer(px_buf, start_y, max_y);
            }
        }
    }

    /// Dirty-frame cached blitter: avoids cloning `PixelBuffer` or re-running Super-Resolution and dithering
    /// on ticks where the CDP frame counter and renderer configuration have not changed.
    pub fn blit_frame_cached<F>(
        &mut self,
        frame_id: usize,
        get_px_buf: F,
        start_y: u16,
        max_y: u16,
    ) where
        F: FnOnce() -> PixelBuffer,
    {
        let fast_path = self.video_fast_path || self.scroll_fast_path;
        let sig = (
            frame_id,
            self.width,
            self.height,
            self.upscale_mode,
            self.dithering_enabled,
            self.pixel_art_mode,
            fast_path,
            self.active_rois.len(),
        );

        match self.protocol {
            GraphicsProtocol::HalfBlockAnsi => {
                if self.cached_frame_sig == Some(sig)
                    && self.cached_bg_cells.len() == self.current.len()
                {
                    let start_idx =
                        (start_y as usize).saturating_mul(self.width as usize).min(self.current.len());
                    let end_idx =
                        (max_y as usize).saturating_mul(self.width as usize).min(self.current.len());
                    if start_idx < end_idx {
                        self.current[start_idx..end_idx]
                            .copy_from_slice(&self.cached_bg_cells[start_idx..end_idx]);
                    }
                    return;
                }
                let px_buf = get_px_buf();
                if px_buf.width == 0 || px_buf.height == 0 {
                    return;
                }
                self.blit_pixel_buffer(&px_buf, start_y, max_y);
                self.cached_bg_cells.clear();
                self.cached_bg_cells.extend_from_slice(&self.current);
                self.cached_frame_sig = Some(sig);
            }
            GraphicsProtocol::Kitty | GraphicsProtocol::Sixel => {
                if self.cached_frame_sig == Some(sig) {
                    return;
                }
                let px_buf = get_px_buf();
                if px_buf.width == 0 || px_buf.height == 0 {
                    return;
                }
                self.blit_frame(&px_buf, start_y, max_y);
                self.cached_frame_sig = Some(sig);
            }
        }
    }

    /// Center-weighted subpixel sampler with subtle UI border preservation.
    /// Avoids the 7x7 morphological dilation artifact that turned text & icons into fat dark blocks,
    /// while preserving faint 1px search-bar borders (#dfe1e5) and crisp icon silhouettes.
    #[inline(always)]
    fn sample_subpixel_box(
        px_buf: &PixelBuffer,
        x0: usize,
        x1: usize,
        y0: usize,
        y1: usize,
    ) -> Color {
        let x1 = x1.max(x0 + 1).min(px_buf.width);
        let y1 = y1.max(y0 + 1).min(px_buf.height);
        let w = x1 - x0;
        let h = y1 - y0;

        let cx = (x0 + w / 2).min(px_buf.width - 1);
        let cy = (y0 + h / 2).min(px_buf.height - 1);
        let center = px_buf.get_pixel(cx, cy);

        if w <= 2 && h <= 2 {
            return center;
        }

        // Sample a tight 3x3 cross/grid around the subpixel center so adjacent cells don't bleed/dilate
        let x_left = (x0 + w / 4).min(px_buf.width - 1);
        let x_right = (x0 + (w * 3) / 4).min(px_buf.width - 1);
        let y_top = (y0 + h / 4).min(px_buf.height - 1);
        let y_bot = (y0 + (h * 3) / 4).min(px_buf.height - 1);

        let c_tl = px_buf.get_pixel(x_left, y_top);
        let c_tr = px_buf.get_pixel(x_right, y_top);
        let c_bl = px_buf.get_pixel(x_left, y_bot);
        let c_br = px_buf.get_pixel(x_right, y_bot);

        // Center-weighted blend (center weight = 4, 4 inner-quadrant samples = 1 each, total = 8)
        let avg_r = ((center.r as u32 * 4 + c_tl.r as u32 + c_tr.r as u32 + c_bl.r as u32 + c_br.r as u32) >> 3) as u8;
        let avg_g = ((center.g as u32 * 4 + c_tl.g as u32 + c_tr.g as u32 + c_bl.g as u32 + c_br.g as u32) >> 3) as u8;
        let avg_b = ((center.b as u32 * 4 + c_tl.b as u32 + c_tr.b as u32 + c_bl.b as u32 + c_br.b as u32) >> 3) as u8;
        let blended = Color::new(avg_r, avg_g, avg_b);
        let avg_lum = blended.luma_fast();

        // Check for thin 1px UI borders (such as Google's #dfe1e5 search box outline) passing through the box
        let mut min_lum = avg_lum;
        let mut min_c = blended;
        let mut sy = y0;
        let step_y = ((h + 2) / 3).max(1);
        let step_x = ((w + 2) / 3).max(1);
        while sy < y1 {
            let mut sx = x0;
            while sx < x1 {
                let p = px_buf.get_pixel(sx, sy);
                let l = p.luma_fast();
                if l < min_lum {
                    min_lum = l;
                    min_c = p;
                }
                sx += step_x;
            }
            sy += step_y;
        }

        let diff = avg_lum.saturating_sub(min_lum);
        if avg_lum > 225 && (10..=45).contains(&diff) {
            // Subtle light-gray UI border on white canvas (e.g. search bar pill border):
            // gently accentuate so the border remains visible in terminal half-blocks.
            return Color::new(
                min_c.r.saturating_sub(14),
                min_c.g.saturating_sub(14),
                min_c.b.saturating_sub(12),
            );
        }

        blended
    }
}

/// Standard 4x4 Ordered Bayer threshold matrix (0..15) for deterministic TrueColor half-block dithering.
pub const BAYER_4X4: [[i16; 4]; 4] = [
    [ 0,  8,  2, 10],
    [12,  4, 14,  6],
    [ 3, 11,  1,  9],
    [15,  7, 13,  5],
];

/// Returns `true` if a color is near-pure white (`r > 248 && g > 248 && b > 248`)
/// or near-pure black (`r < 7 && g < 7 && b < 7`).
#[inline(always)]
pub fn is_near_extremum_rgb(c: Color) -> bool {
    (c.r > 248 && c.g > 248 && c.b > 248) || (c.r < 7 && c.g < 7 && c.b < 7)
}

/// Applies a subtle `[-3..+4]` integer offset based on `BAYER_4X4[y & 3][x & 3]` clamped to `0..=255`.
/// Suppresses dithering on near-pure white (`r,g,b > 248`) and near-pure black (`r,g,b < 7`)
/// so flat UI backgrounds remain clean and run-length batchable, and is 100% deterministic
/// per `(x, y)` so static frames never flicker with dirty-cell diffing.
#[inline(always)]
pub fn dither_truecolor_cell(c: Color, x: usize, y: usize) -> Color {
    if is_near_extremum_rgb(c) {
        return c;
    }
    let offset = (BAYER_4X4[y & 3][x & 3] >> 1) - 3;
    Color {
        r: (c.r as i16 + offset).clamp(0, 255) as u8,
        g: (c.g as i16 + offset).clamp(0, 255) as u8,
        b: (c.b as i16 + offset).clamp(0, 255) as u8,
    }
}

/// Notcurses-style 6-bit RGB noise masking for non-text background cells:
/// Strips 2 LSBs (`& 0xFC`) of subtle JPEG DCT mosquito noise while snapping near-pure white (`>= 252`)
/// to `255` so invisible JPEG compression jitter never forces full-screen ANSI redraws.
#[inline(always)]
pub fn mask_jpeg_noise_6bit(c: Color) -> Color {
    Color {
        r: if c.r >= 252 { 255 } else { c.r & 0xFC },
        g: if c.g >= 252 { 255 } else { c.g & 0xFC },
        b: if c.b >= 252 { 255 } else { c.b & 0xFC },
    }
}

/// 16-color Sixel palette in Sixel percentage coordinates (0..=100).
pub const SIXEL_PALETTE_PCT: [(u8, u8, u8); 16] = [
    (0, 0, 0),
    (100, 100, 100),
    (100, 0, 0),
    (0, 100, 0),
    (0, 0, 100),
    (100, 100, 0),
    (0, 100, 100),
    (100, 0, 100),
    (50, 50, 50),
    (80, 80, 80),
    (100, 40, 0),
    (96, 96, 93),
    (20, 20, 20),
    (40, 70, 100),
    (100, 60, 60),
    (60, 100, 60),
];

/// 16-color Sixel palette scaled to 8-bit RGB (0..=255) for integer error diffusion.
pub const SIXEL_PALETTE_RGB: [(i16, i16, i16); 16] = [
    (0, 0, 0),
    (255, 255, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (128, 128, 128),
    (204, 204, 204),
    (255, 102, 0),
    (245, 245, 237),
    (51, 51, 51),
    (102, 179, 255),
    (255, 153, 153),
    (153, 255, 153),
];

#[inline(always)]
fn nearest_sixel_palette_index(r: i16, g: i16, b: i16) -> u8 {
    let mut best_idx = 0u8;
    let mut best_dist = i32::MAX;
    let mut idx = 0usize;
    while idx < 16 {
        let (pr, pg, pb) = SIXEL_PALETTE_RGB[idx];
        let dr = (r - pr) as i32;
        let dg = (g - pg) as i32;
        let db = (b - pb) as i32;
        // Perceptually weighted integer squared distance (2*dR^2 + 4*dG^2 + 3*dB^2)
        let dist = (dr * dr) * 2 + (dg * dg) * 4 + (db * db) * 3;
        if dist < best_dist {
            best_dist = dist;
            best_idx = idx as u8;
            if dist == 0 {
                break;
            }
        }
        idx += 1;
    }
    best_idx
}

/// Runs an in-place serpentine Floyd-Steinberg error-diffusion pass across a `PixelBuffer`
/// and maps every pixel to its 16-color Sixel palette index (`0..16`).
/// Uses integer bit-shift weights (`* 7 >> 4`, `* 3 >> 4`, `* 5 >> 4`, `* 1 >> 4`)
/// with error clamping (`-32..=32`) to prevent color bleeding across sharp UI edges.
pub fn quantize_sixel_floyd_steinberg(px_buf: &PixelBuffer) -> Vec<u8> {
    let width = px_buf.width;
    let height = px_buf.height;
    let len = width * height;
    if len == 0 {
        return Vec::new();
    }

    let mut work_r = vec![0i16; len];
    let mut work_g = vec![0i16; len];
    let mut work_b = vec![0i16; len];
    for i in 0..len {
        let c = px_buf.data[i];
        work_r[i] = c.r as i16;
        work_g[i] = c.g as i16;
        work_b[i] = c.b as i16;
    }

        let mut indices = vec![0u8; len];
    let max_x = width - 1;
    let max_y = height - 1;

    for y in 0..height {
        let row_off = y * width;
        let ym1_off = y.saturating_sub(1) * width;
        let yp1_off = (y + 1).min(max_y) * width;
        let serpentine_rev = (y & 1) == 1;

        for step in 0..width {
            let x = if serpentine_rev { width - 1 - step } else { step };
            let idx = row_off + x;
            let src_c = px_buf.data[idx];

            // Flat-Region Dither Suppression:
            // If pixel is near-pure white (`>248`), near-pure black (`<7`), or has zero color gradient
            // relative to its immediate 4-connected neighbors, bypass error diffusion completely.
            if src_c.r > 248 && src_c.g > 248 && src_c.b > 248 {
                indices[idx] = 1; // Clean Sixel White
                continue;
            }
            if src_c.r < 7 && src_c.g < 7 && src_c.b < 7 {
                indices[idx] = 0; // Clean Sixel Black
                continue;
            }

            let xm1 = x.saturating_sub(1);
            let xp1 = (x + 1).min(max_x);
            let c_left = px_buf.data[row_off + xm1];
            let c_right = px_buf.data[row_off + xp1];
            let c_up = px_buf.data[ym1_off + x];
            let c_down = px_buf.data[yp1_off + x];

            if src_c == c_left && src_c == c_right && src_c == c_up && src_c == c_down {
                indices[idx] = nearest_sixel_palette_index(
                    src_c.r as i16,
                    src_c.g as i16,
                    src_c.b as i16,
                );
                continue;
            }

            let old_r = work_r[idx].clamp(0, 255);
            let old_g = work_g[idx].clamp(0, 255);
            let old_b = work_b[idx].clamp(0, 255);

            let pal_idx = nearest_sixel_palette_index(old_r, old_g, old_b);
            indices[idx] = pal_idx;

            let (pr, pg, pb) = SIXEL_PALETTE_RGB[pal_idx as usize];
            let err_r = (old_r - pr).clamp(-32, 32);
            let err_g = (old_g - pg).clamp(-32, 32);
            let err_b = (old_b - pb).clamp(-32, 32);

            // Forward neighbor on current row: * 7 >> 4
            let has_fwd = if serpentine_rev { x > 0 } else { x + 1 < width };
            if has_fwd {
                let fwd_x = if serpentine_rev { x - 1 } else { x + 1 };
                let n_idx = row_off + fwd_x;
                work_r[n_idx] += (err_r * 7) >> 4;
                work_g[n_idx] += (err_g * 7) >> 4;
                work_b[n_idx] += (err_b * 7) >> 4;
            }

            // Next row neighbors: bottom-behind (* 3 >> 4), bottom (* 5 >> 4), bottom-forward (* 1 >> 4)
            if y + 1 < height {
                let next_row_off = (y + 1) * width;

                let has_bwd = if serpentine_rev { x + 1 < width } else { x > 0 };
                if has_bwd {
                    let bwd_x = if serpentine_rev { x + 1 } else { x - 1 };
                    let n_idx = next_row_off + bwd_x;
                    work_r[n_idx] += (err_r * 3) >> 4;
                    work_g[n_idx] += (err_g * 3) >> 4;
                    work_b[n_idx] += (err_b * 3) >> 4;
                }

                {
                    let n_idx = next_row_off + x;
                    work_r[n_idx] += (err_r * 5) >> 4;
                    work_g[n_idx] += (err_g * 5) >> 4;
                    work_b[n_idx] += (err_b * 5) >> 4;
                }

                if has_fwd {
                    let fwd_x = if serpentine_rev { x - 1 } else { x + 1 };
                    let n_idx = next_row_off + fwd_x;
                    work_r[n_idx] += (err_r * 1) >> 4;
                    work_g[n_idx] += (err_g * 1) >> 4;
                    work_b[n_idx] += (err_b * 1) >> 4;
                }
            }
        }
    }

    indices
}

impl DiffRenderer {
    /// Renders a PixelBuffer into the terminal using Unicode half-blocks '▀' (upper half = fg, lower half = bg)
    /// and Carbonyl quadrants with:
    /// - Two-Gear Kinetic Frame Pacer (`scroll_fast_path` 1-tap Q16.16 fixed-point blitter in Gear 1)
    /// - Target-Subpixel-Grid-First Pipeline (`self.subpixel_buf` + `self.scratch` zero-allocation arena)
    /// - DOM-Guided Variable-Rate Shading (VRS) & Asset ROI Gating (`self.active_rois`)
    /// - Flat-region-suppressed 4x4 Ordered Bayer dithering and 6-bit RGB JPEG noise masking (`& 0xFC`)
    pub fn blit_pixel_buffer(&mut self, px_buf: &PixelBuffer, start_y: u16, max_y: u16) {
        let cols = self.width as usize;
        let rows = (max_y.saturating_sub(start_y)) as usize;
        let target_h = rows * 2;
        let target_w = cols;

        if target_w == 0 || target_h == 0 || px_buf.width == 0 || px_buf.height == 0 {
            return;
        }

        // 1. Gear 1 — Kinetic Motion Mode (`scroll_fast_path`):
        // Uses 1-tap Q16.16 fixed-point subpixel sampling and bypasses all Neural/EASU/Denoise passes
        // and Bayer dithering so active scrolling locks at 60 FPS (< 0.2ms per frame).
        if self.scroll_fast_path {
            let src_w = px_buf.width;
            let src_h = px_buf.height;
            let quad_w = (target_w * 2).max(1);
            let quad_h = (target_h * 2).max(1);
            let step_x_fp = ((src_w << 16) / quad_w).max(1);
            let step_y_fp = ((src_h << 16) / quad_h).max(1);

            for row in 0..rows {
                let term_y = start_y + (row as u16);
                if term_y >= self.height {
                    break;
                }
                let base_y = row * 4;
                let qy0 = ((base_y * step_y_fp) >> 16).min(src_h - 1);
                let qy1 = (((base_y + 1) * step_y_fp) >> 16).min(src_h - 1);
                let qy2 = (((base_y + 2) * step_y_fp) >> 16).min(src_h - 1);
                let qy3 = (((base_y + 3) * step_y_fp) >> 16).min(src_h - 1);

                for col in 0..cols {
                    let base_x = col * 2;
                    let qx0 = ((base_x * step_x_fp) >> 16).min(src_w - 1);
                    let qx1 = (((base_x + 1) * step_x_fp) >> 16).min(src_w - 1);

                    let tl = avg_color(px_buf.get_pixel(qx0, qy0), px_buf.get_pixel(qx0, qy1));
                    let tr = avg_color(px_buf.get_pixel(qx1, qy0), px_buf.get_pixel(qx1, qy1));
                    let bl = avg_color(px_buf.get_pixel(qx0, qy2), px_buf.get_pixel(qx0, qy3));
                    let br = avg_color(px_buf.get_pixel(qx1, qy2), px_buf.get_pixel(qx1, qy3));

                    let (ch, fg, bg) = binarize_quadrant(tl, tr, bl, br);
                    self.set_cell(
                        col as u16,
                        term_y,
                        Cell {
                            ch,
                            comb: ['\0'; 3],
                            fg: mask_jpeg_noise_6bit(fg),
                            bg: mask_jpeg_noise_6bit(bg),
                            bold: false,
                            underline: false,
                        },
                    );
                }
            }
            return;
        }

        let fast_path = self.video_fast_path;
        let is_pixel_art = !fast_path
            && (self.pixel_art_mode
                || (self.upscale_mode != UpscaleMode::Off
                    && px_buf.width <= target_w * 2
                    && detect_pixel_art(px_buf)));

        // 2. Never upscale a frame whose dimensions are ALREADY >= the terminal subpixel grid `(target_w, target_h)`.
        // Only run pre-downsample spatial upscaling when `px_buf.width < target_w || px_buf.height < target_h`.
        let sub_w = target_w * 2;
        let upscaled_storage;
        let active_buf = if (self.upscale_mode == UpscaleMode::Off && !self.pixel_art_mode)
            || fast_path
            || (px_buf.width >= target_w && px_buf.height >= target_h && !is_pixel_art)
        {
            px_buf
        } else {
            let sr_w = sub_w.max(px_buf.width * 2);
            let sr_h = target_h.max(px_buf.height * 2);
            upscaled_storage = apply_upscale_pipeline_ex(
                px_buf,
                self.upscale_mode,
                sr_w,
                sr_h,
                self.pixel_art_mode,
            );
            &upscaled_storage
        };

        // 3. Target-Subpixel-Grid-First Downsampling into reusable `self.subpixel_buf` (`cols * 2 x rows * 2`)
        // using Q16.16 fixed-point coordinate stepping and zero heap allocations.
        let sub_total = sub_w * target_h;
        if self.subpixel_buf.width != sub_w || self.subpixel_buf.height != target_h {
            self.subpixel_buf.width = sub_w;
            self.subpixel_buf.height = target_h;
            self.subpixel_buf.pixels.resize(sub_total, Color::BLACK);
        }

        let step_x_fp = ((active_buf.width << 16) / sub_w.max(1)).max(1);
        let step_y_fp = ((active_buf.height << 16) / target_h.max(1)).max(1);

        for row in 0..rows {
            let sub_y_top = row * 2;
            let sub_y_bot = row * 2 + 1;
            let top_y0 = ((sub_y_top * step_y_fp) >> 16).min(active_buf.height - 1);
            let top_y1 = (((sub_y_bot * step_y_fp) >> 16).max(top_y0 + 1)).min(active_buf.height);
            let bot_y0 = (top_y1).min(active_buf.height - 1);
            let bot_y1 = ((((row * 2 + 2) * step_y_fp) >> 16).max(bot_y0 + 1)).min(active_buf.height);

            for col in 0..cols {
                let sub_x_left = col * 2;
                let sub_x_right = col * 2 + 1;
                let x0 = ((sub_x_left * step_x_fp) >> 16).min(active_buf.width - 1);
                let x_mid = (((sub_x_right * step_x_fp) >> 16).max(x0 + 1)).min(active_buf.width);
                let x_mid_clamped = x_mid.min(active_buf.width - 1);
                let x1 = ((((col * 2 + 2) * step_x_fp) >> 16).max(x_mid_clamped + 1)).min(active_buf.width);

                let (tl, tr, bl, br) = if is_pixel_art {
                    let cx_l = (x0 + (x_mid.saturating_sub(x0)) / 2).min(active_buf.width - 1);
                    let cx_r = (x_mid_clamped + (x1.saturating_sub(x_mid_clamped)) / 2).min(active_buf.width - 1);
                    let cy_top =
                        (top_y0 + (top_y1.saturating_sub(top_y0)) / 2).min(active_buf.height - 1);
                    let cy_bot =
                        (bot_y0 + (bot_y1.saturating_sub(bot_y0)) / 2).min(active_buf.height - 1);
                    (
                        active_buf.get_pixel(cx_l, cy_top),
                        active_buf.get_pixel(cx_r, cy_top),
                        active_buf.get_pixel(cx_l, cy_bot),
                        active_buf.get_pixel(cx_r, cy_bot),
                    )
                } else {
                    (
                        Self::sample_subpixel_box(active_buf, x0, x_mid, top_y0, top_y1),
                        Self::sample_subpixel_box(active_buf, x_mid_clamped, x1, top_y0, top_y1),
                        Self::sample_subpixel_box(active_buf, x0, x_mid, bot_y0, bot_y1),
                        Self::sample_subpixel_box(active_buf, x_mid_clamped, x1, bot_y0, bot_y1),
                    )
                };
                self.subpixel_buf.set_pixel(sub_x_left, sub_y_top, tl);
                self.subpixel_buf.set_pixel(sub_x_right, sub_y_top, tr);
                self.subpixel_buf.set_pixel(sub_x_left, sub_y_bot, bl);
                self.subpixel_buf.set_pixel(sub_x_right, sub_y_bot, br);
            }
        }

        // 4. DOM-Guided Variable-Rate Shading (VRS) & Asset ROI Gating using Zero-Allocation `self.scratch`:
        // - When `vrs_roi_enabled` or `active_rois` are present:
        //   Outside ROI boxes (85%+ of webpage — layout cards, margins, headers) completely bypasses
        //   Neural/EASU/Denoise passes and Bayer dithering!
        //   Strictly inside `ROI` boxes:
        //   - kind == 0 (Photo/Raster): runs the selected `UpscaleMode` (Denoise + YCbCr Neural + RCAS + Bayer Dither)
        //   - kind == 1 (SVG/Vector/Icon): runs crisp edge-preserving RCAS without smoothing
        //   - kind == 2 (Video): runs single-pass fast subpixel RCAS
        let cell_total = cols * rows;
        self.roi_cell_mask.clear();
        let use_vrs_roi = self.vrs_roi_enabled || !self.active_rois.is_empty();

        if use_vrs_roi {
            self.roi_cell_mask.resize(cell_total, 0);
            let (vw, vh) = if self.roi_viewport.0 > 0 && self.roi_viewport.1 > 0 {
                self.roi_viewport
            } else {
                (px_buf.width as u32, px_buf.height as u32)
            };

            if !is_pixel_art && self.upscale_mode != UpscaleMode::Off {
                for roi in &self.active_rois {
                    let (sx0, sy0, sx1, sy1) = roi.to_subpixel_bounds(sub_w, target_h, vw, vh);
                    let effective_kind = if fast_path && roi.kind == 0 { 2 } else { roi.kind };
                    self.scratch.enhance_roi_subrect_in_place(
                        &mut self.subpixel_buf,
                        sx0,
                        sy0,
                        sx1,
                        sy1,
                        effective_kind,
                        self.upscale_mode,
                    );
                    if roi.kind == 0 && !fast_path {
                        let c0 = (sx0 / 2).min(cols);
                        let c1 = ((sx1 + 1) / 2).min(cols);
                        let r0 = (sy0 / 2).min(rows);
                        let r1 = ((sy1 + 1) / 2).min(rows);
                        for rr in r0..r1 {
                            let row_off = rr * cols;
                            for cc in c0..c1 {
                                self.roi_cell_mask[row_off + cc] = 1;
                            }
                        }
                    }
                }
            }
        } else {
            self.roi_cell_mask.resize(cell_total, if fast_path { 0 } else { 1 });
            if !is_pixel_art && self.upscale_mode != UpscaleMode::Off {
                if fast_path {
                    self.scratch.fsr_rcas_in_place(&mut self.subpixel_buf, 0.35);
                } else {
                    match self.upscale_mode {
                        UpscaleMode::Off => {}
                        UpscaleMode::FsrOnly => {
                            self.scratch.fsr_rcas_in_place(&mut self.subpixel_buf, 0.55);
                        }
                        UpscaleMode::NeuralOnly => {
                            self.scratch.refine_y_neural_1x_in_place(&mut self.subpixel_buf);
                            self.scratch.fsr_rcas_in_place(&mut self.subpixel_buf, 0.48);
                        }
                        UpscaleMode::Combined => {
                            self.scratch.remove_jpeg_artifacts_in_place(&mut self.subpixel_buf);
                            self.scratch.refine_y_neural_1x_in_place(&mut self.subpixel_buf);
                            self.scratch.fsr_rcas_in_place(&mut self.subpixel_buf, 0.62);
                        }
                        UpscaleMode::MultiPass => {
                            self.scratch.remove_jpeg_artifacts_in_place(&mut self.subpixel_buf);
                            self.scratch.refine_y_neural_1x_in_place(&mut self.subpixel_buf);
                            self.scratch.fsr_rcas_in_place(&mut self.subpixel_buf, 0.72);
                        }
                    }
                }
            }
        }

        let dither = self.dithering_enabled;
        let max_col = cols.saturating_sub(1);

        // 5. Hybrid Carbonyl Quadrant Binarization + Flat-Suppressed / ROI-Gated TrueColor Half-Block Dithering
        // with Notcurses-style 6-bit RGB JPEG noise masking (`& 0xFC`)
        for row in 0..rows {
            let term_y = start_y + (row as u16);
            if term_y >= self.height {
                break;
            }
            let sub_y_top = row * 2;
            let sub_y_bot = row * 2 + 1;
            let row_mask_base = row * cols;

            for col in 0..cols {
                let sx_l = col * 2;
                let sx_r = col * 2 + 1;
                let tl = self.subpixel_buf.get_pixel(sx_l, sub_y_top);
                let tr = self.subpixel_buf.get_pixel(sx_r, sub_y_top);
                let bl = self.subpixel_buf.get_pixel(sx_l, sub_y_bot);
                let br = self.subpixel_buf.get_pixel(sx_r, sub_y_bot);

                let horiz_diff = tl
                    .luma_fast()
                    .abs_diff(tr.luma_fast())
                    .max(bl.luma_fast().abs_diff(br.luma_fast()));

                if horiz_diff >= 14 {
                    let (q_ch, q_fg, q_bg) = binarize_quadrant(tl, tr, bl, br);
                    self.set_cell(
                        col as u16,
                        term_y,
                        Cell {
                            ch: q_ch,
                            comb: ['\0'; 3],
                            fg: mask_jpeg_noise_6bit(q_fg),
                            bg: mask_jpeg_noise_6bit(q_bg),
                            bold: false,
                            underline: false,
                        },
                    );
                    continue;
                }

                let raw_top = mask_jpeg_noise_6bit(avg_color(tl, tr));
                let raw_bot = mask_jpeg_noise_6bit(avg_color(bl, br));
                let allow_cell_dither = dither && self.roi_cell_mask[row_mask_base + col] != 0;

                let (top_color, bot_color) = if allow_cell_dither {
                    let prev_col = col.saturating_sub(1) * 2;
                    let next_col = (col + 1).min(max_col) * 2;
                    let left_top = mask_jpeg_noise_6bit(avg_color(
                        self.subpixel_buf.get_pixel(prev_col, sub_y_top),
                        self.subpixel_buf.get_pixel(prev_col + 1, sub_y_top),
                    ));
                    let left_bot = mask_jpeg_noise_6bit(avg_color(
                        self.subpixel_buf.get_pixel(prev_col, sub_y_bot),
                        self.subpixel_buf.get_pixel(prev_col + 1, sub_y_bot),
                    ));
                    let right_top = mask_jpeg_noise_6bit(avg_color(
                        self.subpixel_buf.get_pixel(next_col, sub_y_top),
                        self.subpixel_buf.get_pixel(next_col + 1, sub_y_top),
                    ));
                    let right_bot = mask_jpeg_noise_6bit(avg_color(
                        self.subpixel_buf.get_pixel(next_col, sub_y_bot),
                        self.subpixel_buf.get_pixel(next_col + 1, sub_y_bot),
                    ));

                    let is_flat_cell = raw_top == raw_bot
                        && raw_top == left_top
                        && raw_bot == left_bot
                        && raw_top == right_top
                        && raw_bot == right_bot;

                    if !is_flat_cell {
                        (
                            dither_truecolor_cell(raw_top, col, sub_y_top),
                            dither_truecolor_cell(raw_bot, col, sub_y_bot),
                        )
                    } else {
                        (raw_top, raw_bot)
                    }
                } else {
                    (raw_top, raw_bot)
                };

                let cell = if top_color == bot_color {
                    Cell {
                        ch: ' ',
                        comb: ['\0'; 3],
                        fg: Color::WHITE,
                        bg: top_color,
                        bold: false,
                        underline: false,
                    }
                } else {
                    Cell {
                        ch: '▀',
                        comb: ['\0'; 3],
                        fg: top_color,
                        bg: bot_color,
                        bold: false,
                        underline: false,
                    }
                };
                self.set_cell(col as u16, term_y, cell);
            }
        }
    }

    /// Kitty Graphics Protocol generator (streams 32-bit RGBA APC chunks directly)
    pub fn blit_kitty(&mut self, px_buf: &PixelBuffer, start_y: u16, max_y: u16) {
        if px_buf.width == 0 || px_buf.height == 0 {
            return;
        }

        let target_w = self.width as usize;
        let target_h = (max_y.saturating_sub(start_y) as usize) * 2;
        let upscaled_storage;
        let active_buf = if (self.upscale_mode == UpscaleMode::Off && !self.pixel_art_mode)
            || self.scroll_fast_path
        {
            px_buf
        } else if self.video_fast_path || (px_buf.width >= target_w && px_buf.height >= target_h) {
            upscaled_storage = apply_terminal_subpixel_rcas(px_buf, self.upscale_mode);
            &upscaled_storage
        } else {
            upscaled_storage = apply_upscale_pipeline_ex(
                px_buf,
                self.upscale_mode,
                target_w.max(px_buf.width * 2),
                target_h.max(px_buf.height * 2),
                self.pixel_art_mode,
            );
            &upscaled_storage
        };

        let mut rgba = Vec::with_capacity(active_buf.width * active_buf.height * 4);
        for c in &active_buf.data {
            rgba.push(c.r);
            rgba.push(c.g);
            rgba.push(c.b);
            rgba.push(255);
        }

        let b64 = base64_encode(&rgba);
        let bytes = b64.as_bytes();
        let total_len = bytes.len();

        let cols = self.width;
        let rows = max_y.saturating_sub(start_y);

        let mut out = Vec::with_capacity(total_len + 512);
        // DEC Synchronized Output begin
        out.extend_from_slice(b"\x1b[?2026h");
        out.extend_from_slice(b"\x1b_Ga=d,d=A\x1b\\");
        out.extend_from_slice(b"\x1b[");
        push_u16(&mut out, start_y + 1);
        out.extend_from_slice(b";1H");

        let chunk_size = 4096;
        let mut offset = 0;
        let mut first = true;

        while offset < total_len {
            let end = (offset + chunk_size).min(total_len);
            let is_last = end == total_len;
            let m_val = if is_last { b"0" } else { b"1" };

            if first {
                out.extend_from_slice(b"\x1b_Ga=T,f=32,s=");
                push_u16(&mut out, active_buf.width as u16);
                out.extend_from_slice(b",v=");
                push_u16(&mut out, active_buf.height as u16);
                out.extend_from_slice(b",c=");
                push_u16(&mut out, cols);
                out.extend_from_slice(b",r=");
                push_u16(&mut out, rows);
                out.extend_from_slice(b",m=");
                out.extend_from_slice(m_val);
                out.push(b';');
                out.extend_from_slice(&bytes[offset..end]);
                out.extend_from_slice(b"\x1b\\");
                first = false;
            } else {
                out.extend_from_slice(b"\x1b_Gm=");
                out.extend_from_slice(m_val);
                out.push(b';');
                out.extend_from_slice(&bytes[offset..end]);
                out.extend_from_slice(b"\x1b\\");
            }

            offset = end;
        }

        // DEC Synchronized Output end
        out.extend_from_slice(b"\x1b[?2026l");
        let _ = write_stdout_bytes(&out);
    }

    /// Sixel Graphics Protocol generator (6-pixel vertical sliver encoding with Serpentine Floyd-Steinberg dithering)
    pub fn blit_sixel(&mut self, px_buf: &PixelBuffer, start_y: u16, _max_y: u16) {
        if px_buf.width == 0 || px_buf.height == 0 {
            return;
        }

        let target_w = self.width as usize;
        let target_h = (self.height.saturating_sub(start_y) as usize) * 2;
        let upscaled_storage;
        let active_buf = if (self.upscale_mode == UpscaleMode::Off && !self.pixel_art_mode)
            || self.scroll_fast_path
        {
            px_buf
        } else if self.video_fast_path || (px_buf.width >= target_w && px_buf.height >= target_h) {
            upscaled_storage = apply_terminal_subpixel_rcas(px_buf, self.upscale_mode);
            &upscaled_storage
        } else {
            upscaled_storage = apply_upscale_pipeline_ex(
                px_buf,
                self.upscale_mode,
                target_w.max(px_buf.width * 2),
                target_h.max(px_buf.height * 2),
                self.pixel_art_mode,
            );
            &upscaled_storage
        };

        let mut out = Vec::with_capacity(active_buf.width * active_buf.height + 1024);
        // DEC Synchronized Output begin
        out.extend_from_slice(b"\x1b[?2026h");
        out.extend_from_slice(b"\x1b[");
        push_u16(&mut out, start_y + 1);
        out.extend_from_slice(b";1H");

        out.extend_from_slice(b"\x1bPq\"1;1;");
        push_u16(&mut out, active_buf.width as u16);
        out.push(b';');
        push_u16(&mut out, active_buf.height as u16);

        for (idx, (r, g, b)) in SIXEL_PALETTE_PCT.iter().enumerate() {
            out.push(b'#');
            push_u8(&mut out, idx as u8);
            out.extend_from_slice(b";2;");
            push_u8(&mut out, *r);
            out.push(b';');
            push_u8(&mut out, *g);
            out.push(b';');
            push_u8(&mut out, *b);
        }

        // Run serpentine Floyd-Steinberg error-diffusion pass (or direct nearest quantization if dithering is disabled)
        let quantized_indices = if self.dithering_enabled {
            quantize_sixel_floyd_steinberg(active_buf)
        } else {
            active_buf
                .data
                .iter()
                .map(|c| nearest_sixel_palette_index(c.r as i16, c.g as i16, c.b as i16))
                .collect()
        };

        let width = active_buf.width;
        let height = active_buf.height;
        let bands = (height + 5) / 6;

        for band in 0..bands {
            let start_py = band * 6;
            for col_idx in 0..16u8 {
                let mut has_pixels = false;
                let mut band_chars = Vec::with_capacity(width);

                for x in 0..width {
                    let mut mask = 0u8;
                    for bit in 0..6 {
                        let y = start_py + bit;
                        if y < height {
                            if quantized_indices[y * width + x] == col_idx {
                                mask |= 1 << bit;
                            }
                        }
                    }
                    if mask > 0 {
                        has_pixels = true;
                    }
                    band_chars.push(63 + mask);
                }

                if has_pixels {
                    out.push(b'#');
                    push_u8(&mut out, col_idx);
                    for ch in band_chars {
                        out.push(ch);
                    }
                    out.push(b'$');
                }
            }
            out.push(b'-');
        }

        out.extend_from_slice(b"\x1b\\");
        // DEC Synchronized Output end
        out.extend_from_slice(b"\x1b[?2026l");

        let _ = write_stdout_bytes(&out);
    }

    /// Encodes all dirty cells into `self.write_buf` wrapped in DEC Synchronized Output
    /// (`\x1b[?2026h` begin synchronized update ... `\x1b[?2026l` end synchronized update).
    /// Uses Carbonyl-style SGR delta compression (targeted `\x1b[22m`/`\x1b[24m` attribute off-codes
    /// without destructive `\x1b[0m` color resets, optional xterm-256 quantization, perceptual motion
    /// color coalescing, and compact `\x1b[nC` relative cursor advances).
    pub fn build_synchronized_frame(&mut self) -> usize {
        self.write_buf.clear();

        let width = self.width as usize;
        let height = self.height as usize;

        let mut last_fg: Option<Color> = None;
        let mut last_bg: Option<Color> = None;
        let mut last_fg_code: Option<u8> = None;
        let mut last_bg_code: Option<u8> = None;
        let mut last_bold: Option<bool> = None;
        let mut last_underline: Option<bool> = None;
        let mut cursor_pos: Option<(u16, u16)> = None;

        let mut dirty_count = 0;
        let color_tol: u8 = if self.scroll_fast_path { 2 } else { 0 };
        let fast_256 = self.fast_color_mode;

        #[inline(always)]
        fn color_matches(a: Color, b: Color, tol: u8) -> bool {
            if tol == 0 {
                a == b
            } else {
                a.r.abs_diff(b.r) <= tol && a.g.abs_diff(b.g) <= tol && a.b.abs_diff(b.b) <= tol
            }
        }

        // Reserve DEC Mode 2026 Synchronized Update header once the first dirty cell is encountered
        for y in 0..height {
            let row_offset = y * width;
            let mut x = 0;

            while x < width {
                let idx = row_offset + x;
                let cur = self.current[idx];
                let prev = self.previous[idx];

                // Notcurses-style dirty-cell perceptual noise gate:
                // For non-text background cells (' ' or subpixel quadrant/half-block glyphs), ignore
                // invisible < 4 LSB JPEG compression jitter so static frames emit 0 bytes.
                let is_bg_glyph = cur.ch == ' ' || is_quadrant_block_char(cur.ch);
                let cell_clean = if cur == prev {
                    true
                } else if is_bg_glyph
                    && cur.ch == prev.ch
                    && cur.comb == prev.comb
                    && cur.bold == prev.bold
                    && cur.underline == prev.underline
                {
                    let fg_match = cur.ch == ' '
                        || (cur.fg.r.abs_diff(prev.fg.r) < 4
                            && cur.fg.g.abs_diff(prev.fg.g) < 4
                            && cur.fg.b.abs_diff(prev.fg.b) < 4);
                    let bg_match = cur.bg.r.abs_diff(prev.bg.r) < 4
                        && cur.bg.g.abs_diff(prev.bg.g) < 4
                        && cur.bg.b.abs_diff(prev.bg.b) < 4;
                    fg_match && bg_match
                } else {
                    false
                };

                if cell_clean {
                    x += 1;
                    continue;
                }

                if dirty_count == 0 {
                    // Begin DEC Synchronized Output (\x1b[?2026h) and initialize clean SGR baseline once
                    self.write_buf.extend_from_slice(b"\x1b[?2026h\x1b[0m");
                    last_bold = Some(false);
                    last_underline = Some(false);
                }
                dirty_count += 1;

                let target_pos = (x as u16 + 1, y as u16 + 1);
                if cursor_pos != Some(target_pos) {
                    if let Some((cx, cy)) = cursor_pos {
                        if cy == target_pos.1 && target_pos.0 > cx && target_pos.0 - cx <= 4 {
                            let dx = target_pos.0 - cx;
                            self.write_buf.extend_from_slice(b"\x1b[");
                            if dx > 1 {
                                push_u16(&mut self.write_buf, dx);
                            }
                            self.write_buf.push(b'C');
                        } else {
                            self.write_buf.extend_from_slice(b"\x1b[");
                            push_u16(&mut self.write_buf, target_pos.1);
                            self.write_buf.push(b';');
                            push_u16(&mut self.write_buf, target_pos.0);
                            self.write_buf.push(b'H');
                        }
                    } else {
                        self.write_buf.extend_from_slice(b"\x1b[");
                        push_u16(&mut self.write_buf, target_pos.1);
                        self.write_buf.push(b';');
                        push_u16(&mut self.write_buf, target_pos.0);
                        self.write_buf.push(b'H');
                    }
                }

                let run_fg = cur.fg;
                let run_bg = cur.bg;
                let run_bold = cur.bold;
                let run_underline = cur.underline;

                // Non-destructive SGR attribute delta updates (\x1b[1m / \x1b[22m, \x1b[4m / \x1b[24m)
                // Preserves active foreground and background colors without a full \x1b[0m reset!
                if last_bold != Some(run_bold) {
                    last_bold = Some(run_bold);
                    if run_bold {
                        self.write_buf.extend_from_slice(b"\x1b[1m");
                    } else {
                        self.write_buf.extend_from_slice(b"\x1b[22m");
                    }
                }
                if last_underline != Some(run_underline) {
                    last_underline = Some(run_underline);
                    if run_underline {
                        self.write_buf.extend_from_slice(b"\x1b[4m");
                    } else {
                        self.write_buf.extend_from_slice(b"\x1b[24m");
                    }
                }

                if fast_256 {
                    let fg_code = rgb_to_xterm256(run_fg);
                    if last_fg_code != Some(fg_code) {
                        last_fg_code = Some(fg_code);
                        self.write_buf.extend_from_slice(b"\x1b[38;5;");
                        push_u8(&mut self.write_buf, fg_code);
                        self.write_buf.push(b'm');
                    }
                    let bg_code = rgb_to_xterm256(run_bg);
                    if last_bg_code != Some(bg_code) {
                        last_bg_code = Some(bg_code);
                        self.write_buf.extend_from_slice(b"\x1b[48;5;");
                        push_u8(&mut self.write_buf, bg_code);
                        self.write_buf.push(b'm');
                    }
                } else {
                    if !last_fg.map_or(false, |prev_fg| color_matches(prev_fg, run_fg, color_tol)) {
                        last_fg = Some(run_fg);
                        self.write_buf.extend_from_slice(b"\x1b[38;2;");
                        push_u8(&mut self.write_buf, run_fg.r);
                        self.write_buf.push(b';');
                        push_u8(&mut self.write_buf, run_fg.g);
                        self.write_buf.push(b';');
                        push_u8(&mut self.write_buf, run_fg.b);
                        self.write_buf.push(b'm');
                    }

                    if !last_bg.map_or(false, |prev_bg| color_matches(prev_bg, run_bg, color_tol)) {
                        last_bg = Some(run_bg);
                        self.write_buf.extend_from_slice(b"\x1b[48;2;");
                        push_u8(&mut self.write_buf, run_bg.r);
                        self.write_buf.push(b';');
                        push_u8(&mut self.write_buf, run_bg.g);
                        self.write_buf.push(b';');
                        push_u8(&mut self.write_buf, run_bg.b);
                        self.write_buf.push(b'm');
                    }
                }

                let mut run_len = 0;
                while x < width {
                    let r_idx = row_offset + x;
                    let c = self.current[r_idx];
                    if c.bold != run_bold
                        || c.underline != run_underline
                        || !color_matches(c.fg, run_fg, color_tol)
                        || !color_matches(c.bg, run_bg, color_tol)
                    {
                        break;
                    }

                    let mut char_buf = [0u8; 4];
                    let s = c.ch.encode_utf8(&mut char_buf);
                    self.write_buf.extend_from_slice(s.as_bytes());
                    for &comb_ch in &c.comb {
                        if comb_ch == '\0' {
                            break;
                        }
                        let cs = comb_ch.encode_utf8(&mut char_buf);
                        self.write_buf.extend_from_slice(cs.as_bytes());
                    }

                    self.previous[r_idx] = c;
                    x += 1;
                    run_len += 1;
                }

                cursor_pos = Some((target_pos.0 + run_len, target_pos.1));
            }
        }

        if dirty_count > 0 {
            // End DEC Synchronized Output (\x1b[?2026l)
            self.write_buf.extend_from_slice(b"\x1b[?2026l");
        }

        dirty_count
    }

    /// High-performance dirty-cell diffing, run-length batching, and atomic DEC Synchronized Output flush.
    pub fn flush(&mut self) -> io::Result<usize> {
        let dirty_count = self.build_synchronized_frame();
        if dirty_count > 0 {
            write_stdout_bytes(&self.write_buf)?;
        }
        Ok(dirty_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base64_encode() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn test_protocol_auto_detection() {
        let proto = GraphicsProtocol::auto_detect();
        assert!(proto == GraphicsProtocol::HalfBlockAnsi || proto == GraphicsProtocol::Kitty || proto == GraphicsProtocol::Sixel);
        assert!(!proto.name().is_empty());
        assert!(!proto.short_name().is_empty());
    }

    #[test]
    fn test_indic_combining_clustering() {
        let mut renderer = DiffRenderer::new(15, 3);
        // "हिन्दी" has 6 Unicode chars: 'ह', 'ि' (combining), 'न', '्' (combining), 'द', 'ी' (combining)
        // It should occupy exactly 3 terminal cells, NOT 6!
        assert_eq!(display_width("हिन्दी"), 3);
        renderer.put_str(0, 0, "हिन्दी", Color::BLACK, Color::WHITE, false, false);
        assert_eq!(renderer.current[0].ch, 'ह');
        assert_eq!(renderer.current[0].comb[0], 'ि');
        assert_eq!(renderer.current[1].ch, 'न');
        assert_eq!(renderer.current[1].comb[0], '्');
        assert_eq!(renderer.current[2].ch, 'द');
        assert_eq!(renderer.current[2].comb[0], 'ी');
        assert_eq!(renderer.current[3].ch, ' ');
    }

    #[test]
    fn test_put_str_with_existing_bg() {
        let mut renderer = DiffRenderer::new(10, 5);
        renderer.set_cell(0, 0, Cell {
            ch: '▀',
            comb: ['\0'; 3],
            fg: Color::BLACK,
            bg: Color::WHITE,
            bold: false,
            underline: false,
        });

        renderer.put_str_with_existing_bg(0, 0, "A", Color::BLACK, None, true, false);

        let cell = renderer.current[0];
        assert_eq!(cell.ch, 'A');
        assert_eq!(cell.fg, Color::BLACK);
        assert_eq!(cell.bg, Color::WHITE);
        assert!(cell.bold);
    }

    #[test]
    fn test_bayer_dither_truecolor_cell() {
        let mid = Color::new(128, 128, 128);
        let d0 = dither_truecolor_cell(mid, 0, 0); // BAYER_4X4[0][0] = 0 -> offset -3
        let d1 = dither_truecolor_cell(mid, 1, 3); // BAYER_4X4[3][1] = 7 -> offset 0
        let d2 = dither_truecolor_cell(mid, 0, 3); // BAYER_4X4[3][0] = 15 -> offset +4
        assert_eq!(d0, Color::new(125, 125, 125));
        assert_eq!(d1, Color::new(128, 128, 128));
        assert_eq!(d2, Color::new(132, 132, 132));

        // Near-white (>248) and near-black (<7) remain untouched for clean run-length batching
        assert_eq!(dither_truecolor_cell(Color::WHITE, 2, 1), Color::WHITE);
        assert_eq!(dither_truecolor_cell(Color::new(250, 251, 249), 0, 0), Color::new(250, 251, 249));
        assert_eq!(dither_truecolor_cell(Color::BLACK, 3, 2), Color::BLACK);
        assert_eq!(dither_truecolor_cell(Color::new(4, 5, 6), 0, 3), Color::new(4, 5, 6));

        // Flat constant background in blit_pixel_buffer must emit clean space cells without Bayer noise
        let mut renderer = DiffRenderer::new(8, 4);
        let flat_banner = Color::new(32, 96, 192);
        let flat_buf = PixelBuffer::new(16, 16, flat_banner);
        renderer.blit_pixel_buffer(&flat_buf, 1, 4);
        for cell in &renderer.current[8..] {
            assert_eq!(cell.ch, ' ');
            assert_eq!(cell.bg, flat_banner);
        }
    }

    #[test]
    fn test_sixel_serpentine_floyd_steinberg() {
        let mut buf = PixelBuffer::new(8, 6, Color::BLACK);
        for y in 0..6 {
            for x in 0..8 {
                let v = ((x * 28 + y * 10) as u8).min(255);
                buf.set_pixel(x, y, Color::new(v, v, v));
            }
        }
        let indices = quantize_sixel_floyd_steinberg(&buf);
        assert_eq!(indices.len(), 48);
        assert!(indices.iter().all(|&idx| idx < 16));
    }

    #[test]
    fn test_dec_synchronized_output_and_video_fast_path() {
        let mut renderer = DiffRenderer::new(10, 4);
        renderer.video_fast_path = true;
        renderer.upscale_mode = UpscaleMode::MultiPass;

        let mut buf = PixelBuffer::new(20, 12, Color::new(20, 40, 60));
        buf.set_pixel(5, 5, Color::new(220, 180, 90));
        renderer.blit_pixel_buffer(&buf, 1, 4);

        let dirty = renderer.build_synchronized_frame();
        assert!(dirty > 0);
        assert!(
            renderer.write_buf.starts_with(b"\x1b[?2026h"),
            "Frame must begin with DEC Synchronized Update \\x1b[?2026h"
        );
        assert!(
            renderer.write_buf.ends_with(b"\x1b[?2026l"),
            "Frame must end with DEC Synchronized Update \\x1b[?2026l"
        );

        // Second build with no changes should emit 0 bytes
        let dirty_second = renderer.build_synchronized_frame();
        assert_eq!(dirty_second, 0);
        assert!(renderer.write_buf.is_empty());
    }

    #[test]
    fn test_carbonyl_quadrant_binarization() {
        let hi = Color::new(240, 240, 240);
        let lo = Color::new(20, 20, 20);

        // Flat cell -> clean space
        let (ch_flat, _, bg_flat) = binarize_quadrant(hi, hi, hi, hi);
        assert_eq!(ch_flat, ' ');
        assert_eq!(bg_flat, hi);

        // Left half bright, right half dark -> '▌'
        let (ch_left, fg_left, bg_left) = binarize_quadrant(hi, lo, hi, lo);
        assert_eq!(ch_left, '▌');
        assert_eq!(fg_left, hi);
        assert_eq!(bg_left, lo);

        // Right half bright, left half dark -> '▐'
        let (ch_right, fg_right, bg_right) = binarize_quadrant(lo, hi, lo, hi);
        assert_eq!(ch_right, '▐');
        assert_eq!(fg_right, hi);
        assert_eq!(bg_right, lo);

        // Top half bright, bottom half dark -> '▀'
        let (ch_top, fg_top, bg_top) = binarize_quadrant(hi, hi, lo, lo);
        assert_eq!(ch_top, '▀');
        assert_eq!(fg_top, hi);
        assert_eq!(bg_top, lo);

        // Diagonal quadrants -> '▚' and '▞'
        let (ch_diag1, _, _) = binarize_quadrant(hi, lo, lo, hi);
        assert_eq!(ch_diag1, '▚');
        let (ch_diag2, _, _) = binarize_quadrant(lo, hi, hi, lo);
        assert_eq!(ch_diag2, '▞');

        // Corner quadrants -> '▘', '▝', '▖', '▗'
        assert_eq!(binarize_quadrant(hi, lo, lo, lo).0, '▘');
        assert_eq!(binarize_quadrant(lo, hi, lo, lo).0, '▝');
        assert_eq!(binarize_quadrant(lo, lo, hi, lo).0, '▖');
        assert_eq!(binarize_quadrant(lo, lo, lo, hi).0, '▗');
    }

    #[test]
    fn test_sgr_delta_compression_and_xterm256() {
        assert_eq!(rgb_to_xterm256(Color::BLACK), 16);
        assert_eq!(rgb_to_xterm256(Color::WHITE), 231);
        let gray_idx = rgb_to_xterm256(Color::new(128, 128, 128));
        assert!((232..=255).contains(&gray_idx));

        let mut renderer = DiffRenderer::new(10, 2);
        let fg = Color::new(40, 120, 220);
        let bg = Color::new(250, 250, 250);
        // Cell 0: bold 'A', Cell 1: non-bold 'B' with identical fg and bg
        renderer.put_str(0, 0, "A", fg, bg, true, false);
        renderer.put_str(1, 0, "B", fg, bg, false, false);
        let dirty = renderer.build_synchronized_frame();
        assert!(dirty >= 2);
        let frame_str = String::from_utf8_lossy(&renderer.write_buf);
        // Must use \x1b[22m to turn off bold without re-emitting \x1b[38;2;40;120;220m a second time
        assert!(frame_str.contains("\x1b[1m"));
        assert!(frame_str.contains("\x1b[22m"));
        assert_eq!(frame_str.matches("\x1b[38;2;40;120;220m").count(), 1);
    }

    #[test]
    fn test_upscale_after_downsample_under_2ms() {
        // Verify that a full 360x576 Chromium viewport rendered into a 60x48 terminal (60x96 subpixels)
        // never upscales the 360x576 frame and completes in well under 20ms (even in unoptimized debug builds,
        // and < 1ms in release builds).
        let mut renderer = DiffRenderer::new(60, 49);
        let mut buf = PixelBuffer::new(360, 576, Color::new(248, 249, 250));
        for y in 40..120 {
            for x in 30..330 {
                buf.set_pixel(x, y, Color::new((x % 200) as u8, (y % 200) as u8, 180));
            }
        }

        for mode in [
            UpscaleMode::FsrOnly,
            UpscaleMode::NeuralOnly,
            UpscaleMode::Combined,
            UpscaleMode::MultiPass,
        ] {
            renderer.upscale_mode = mode;
            renderer.scroll_fast_path = false;
            renderer.blit_pixel_buffer(&buf, 1, 49);
        }

        // Scroll fast-path benchmark
        renderer.scroll_fast_path = true;
        let t0 = std::time::Instant::now();
        renderer.blit_pixel_buffer(&buf, 1, 49);
        assert!(t0.elapsed() < std::time::Duration::from_millis(15));
    }

    #[test]
    fn test_notcurses_6bit_jpeg_noise_masking_and_diff_skip() {
        assert_eq!(mask_jpeg_noise_6bit(Color::new(255, 254, 252)), Color::WHITE);
        assert_eq!(
            mask_jpeg_noise_6bit(Color::new(103, 142, 201)),
            Color::new(100, 140, 200)
        );

        let mut renderer = DiffRenderer::new(8, 3);
        let buf1 = PixelBuffer::new(16, 12, Color::new(100, 140, 200));
        renderer.blit_pixel_buffer(&buf1, 1, 3);
        assert!(renderer.build_synchronized_frame() > 0);

        // Simulate 1-3 LSB JPEG mosquito noise on static page -> must emit 0 dirty cells!
        let buf2 = PixelBuffer::new(16, 12, Color::new(102, 141, 203));
        renderer.blit_pixel_buffer(&buf2, 1, 3);
        assert_eq!(renderer.build_synchronized_frame(), 0);
    }

    #[test]
    fn test_fill_rect_mask_clears_stale_underlying_text() {
        let mut renderer = DiffRenderer::new(20, 6);
        renderer.put_str(2, 2, "StalePageText", Color::BLACK, Color::WHITE, false, false);
        assert_eq!(renderer.current[2 * 20 + 2].ch, 'S');

        // Popup POPUP_MASK clears [1..18, 1..5] before drawing suggestion item
        renderer.fill_popup_mask(1, 1, 18, 5, Color::new(255, 255, 255));
        assert_eq!(renderer.current[2 * 20 + 2].ch, ' ');
        renderer.put_str_with_existing_bg(
            2,
            2,
            "rust tutorial",
            Color::new(32, 33, 36),
            Some(Color::WHITE),
            false,
            false,
        );
        assert_eq!(renderer.current[2 * 20 + 2].ch, 'r');
    }

    #[test]
    fn test_vrs_roi_gated_subpixel_under_1_5ms() {
        let rois = AssetRoi::parse_list("ROI:40,60,120,90,0;200,40,48,48,1;10,300,320,180,2");
        assert_eq!(rois.len(), 3);
        assert_eq!(rois[0].kind, 0);
        assert_eq!(rois[1].kind, 1);
        assert_eq!(rois[2].kind, 2);

        let mut renderer = DiffRenderer::new(60, 49);
        renderer.set_active_rois(rois, 360, 576);
        renderer.upscale_mode = UpscaleMode::Combined;

        let mut buf = PixelBuffer::new(360, 576, Color::new(248, 249, 250));
        for y in 60..150 {
            for x in 40..160 {
                buf.set_pixel(x, y, Color::new(((x * 3) % 240) as u8, ((y * 2) % 240) as u8, 180));
            }
        }

        // Warm up scratchpad capacity once
        renderer.blit_pixel_buffer(&buf, 1, 49);
        let cap_y = renderer.scratch.y_plane.capacity();

        let t0 = std::time::Instant::now();
        renderer.blit_pixel_buffer(&buf, 1, 49);
        assert!(t0.elapsed() < std::time::Duration::from_millis(15));
        // Verify zero reallocation in steady state
        assert_eq!(renderer.scratch.y_plane.capacity(), cap_y);
    }
}


