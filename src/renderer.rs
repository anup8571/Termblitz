#![allow(dead_code)]
use std::io;
use crate::upscale::{apply_upscale_pipeline, UpscaleMode};

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
        let rgba = img.to_rgba8();
        let mut data = Vec::with_capacity((w * h) as usize);
        for py in 0..h {
            for px in 0..w {
                let p = rgba.get_pixel(px, py);
                data.push(Color::new(p[0], p[1], p[2]));
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
    current: Vec<Cell>,
    previous: Vec<Cell>,
    write_buf: Vec<u8>,
}

impl DiffRenderer {
    pub fn new(width: u16, height: u16) -> Self {
        let size = (width as usize) * (height as usize);
        Self {
            width,
            height,
            protocol: GraphicsProtocol::auto_detect(),
            upscale_mode: UpscaleMode::FsrOnly,
            dithering_enabled: true,
            current: vec![Cell::EMPTY; size],
            previous: vec![Cell::INVALID; size],
            write_buf: Vec::with_capacity(65536),
        }
    }

    pub fn resize(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;
        let size = (width as usize) * (height as usize);
        self.current = vec![Cell::EMPTY; size];
        self.previous = vec![Cell::INVALID; size];
        self.write_buf.clear();
    }

    pub fn force_redraw(&mut self) {
        self.previous.fill(Cell::INVALID);
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
                if cell.ch == '▀' {
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
                } else if cell.ch == '▀' {
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

/// Applies a subtle `[-3..+4]` integer offset based on `BAYER_4X4[y & 3][x & 3]` clamped to `0..=255`.
/// Preserves pure black `(0,0,0)` and pure white `(255,255,255)` so flat UI backgrounds remain run-length batchable,
/// and is 100% deterministic per `(x, y)` so static frames never flicker with dirty-cell diffing.
#[inline(always)]
pub fn dither_truecolor_cell(c: Color, x: usize, y: usize) -> Color {
    if (c.r == 255 && c.g == 255 && c.b == 255) || (c.r == 0 && c.g == 0 && c.b == 0) {
        return c;
    }
    let offset = (BAYER_4X4[y & 3][x & 3] >> 1) - 3;
    Color {
        r: (c.r as i16 + offset).clamp(0, 255) as u8,
        g: (c.g as i16 + offset).clamp(0, 255) as u8,
        b: (c.b as i16 + offset).clamp(0, 255) as u8,
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

    for y in 0..height {
        let row_off = y * width;
        let serpentine_rev = (y & 1) == 1;

        for step in 0..width {
            let x = if serpentine_rev { width - 1 - step } else { step };
            let idx = row_off + x;

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
    /// with the active `UpscaleMode` super-resolution pipeline and optional 4x4 Ordered Bayer dithering.
    pub fn blit_pixel_buffer(&mut self, px_buf: &PixelBuffer, start_y: u16, max_y: u16) {
        let cols = self.width as usize;
        let rows = (max_y.saturating_sub(start_y)) as usize;
        let target_h = rows * 2;
        let target_w = cols;

        if target_w == 0 || target_h == 0 || px_buf.width == 0 || px_buf.height == 0 {
            return;
        }

        let upscaled_storage;
        let active_buf = if self.upscale_mode == UpscaleMode::Off {
            px_buf
        } else {
            let sr_w = (target_w * 2).max(px_buf.width);
            let sr_h = (target_h * 2).max(px_buf.height);
            upscaled_storage = apply_upscale_pipeline(px_buf, self.upscale_mode, sr_w, sr_h);
            &upscaled_storage
        };

        let dither = self.dithering_enabled;

        for row in 0..rows {
            let term_y = start_y + (row as u16);
            if term_y >= self.height {
                break;
            }
            let sub_y_top = row * 2;
            let sub_y_bot = row * 2 + 1;
            let top_y0 = (sub_y_top * active_buf.height) / target_h;
            let top_y1 = (sub_y_bot * active_buf.height) / target_h;
            let bot_y0 = top_y1;
            let bot_y1 = ((row * 2 + 2) * active_buf.height) / target_h;

            for col in 0..cols {
                let x0 = (col * active_buf.width) / target_w;
                let x1 = ((col + 1) * active_buf.width) / target_w;

                let raw_top = Self::sample_subpixel_box(active_buf, x0, x1, top_y0, top_y1);
                let raw_bot = Self::sample_subpixel_box(active_buf, x0, x1, bot_y0, bot_y1);

                let (top_color, bot_color) = if dither {
                    (
                        dither_truecolor_cell(raw_top, col, sub_y_top),
                        dither_truecolor_cell(raw_bot, col, sub_y_bot),
                    )
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

        let upscaled_storage;
        let active_buf = if self.upscale_mode == UpscaleMode::Off {
            px_buf
        } else {
            upscaled_storage = apply_upscale_pipeline(
                px_buf,
                self.upscale_mode,
                px_buf.width * 2,
                px_buf.height * 2,
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

        let _ = write_stdout_bytes(&out);
    }

    /// Sixel Graphics Protocol generator (6-pixel vertical sliver encoding with Serpentine Floyd-Steinberg dithering)
    pub fn blit_sixel(&mut self, px_buf: &PixelBuffer, start_y: u16, _max_y: u16) {
        if px_buf.width == 0 || px_buf.height == 0 {
            return;
        }

        let upscaled_storage;
        let active_buf = if self.upscale_mode == UpscaleMode::Off {
            px_buf
        } else {
            upscaled_storage = apply_upscale_pipeline(
                px_buf,
                self.upscale_mode,
                px_buf.width * 2,
                px_buf.height * 2,
            );
            &upscaled_storage
        };

        let mut out = Vec::with_capacity(active_buf.width * active_buf.height + 1024);

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

        let _ = write_stdout_bytes(&out);
    }

    /// High-performance dirty-cell diffing and run-length batching.
    pub fn flush(&mut self) -> io::Result<usize> {
        self.write_buf.clear();

        let width = self.width as usize;
        let height = self.height as usize;

        let mut last_fg: Option<Color> = None;
        let mut last_bg: Option<Color> = None;
        let mut last_bold: Option<bool> = None;
        let mut last_underline: Option<bool> = None;
        let mut cursor_pos: Option<(u16, u16)> = None;

        let mut dirty_count = 0;

        for y in 0..height {
            let row_offset = y * width;
            let mut x = 0;

            while x < width {
                let idx = row_offset + x;
                let cur = self.current[idx];
                let prev = self.previous[idx];

                if cur == prev {
                    x += 1;
                    continue;
                }

                dirty_count += 1;

                let target_pos = (x as u16 + 1, y as u16 + 1);
                if cursor_pos != Some(target_pos) {
                    self.write_buf.extend_from_slice(b"\x1b[");
                    push_u16(&mut self.write_buf, target_pos.1);
                    self.write_buf.push(b';');
                    push_u16(&mut self.write_buf, target_pos.0);
                    self.write_buf.push(b'H');
                }

                let run_fg = cur.fg;
                let run_bg = cur.bg;
                let run_bold = cur.bold;
                let run_underline = cur.underline;

                if last_bold != Some(run_bold) || last_underline != Some(run_underline) {
                    last_bold = Some(run_bold);
                    last_underline = Some(run_underline);

                    self.write_buf.extend_from_slice(b"\x1b[0");
                    if run_bold {
                        self.write_buf.extend_from_slice(b";1");
                    }
                    if run_underline {
                        self.write_buf.extend_from_slice(b";4");
                    }
                    self.write_buf.push(b'm');
                    last_fg = None;
                    last_bg = None;
                }

                if last_fg != Some(run_fg) {
                    last_fg = Some(run_fg);
                    self.write_buf.extend_from_slice(b"\x1b[38;2;");
                    push_u8(&mut self.write_buf, run_fg.r);
                    self.write_buf.push(b';');
                    push_u8(&mut self.write_buf, run_fg.g);
                    self.write_buf.push(b';');
                    push_u8(&mut self.write_buf, run_fg.b);
                    self.write_buf.push(b'm');
                }

                if last_bg != Some(run_bg) {
                    last_bg = Some(run_bg);
                    self.write_buf.extend_from_slice(b"\x1b[48;2;");
                    push_u8(&mut self.write_buf, run_bg.r);
                    self.write_buf.push(b';');
                    push_u8(&mut self.write_buf, run_bg.g);
                    self.write_buf.push(b';');
                    push_u8(&mut self.write_buf, run_bg.b);
                    self.write_buf.push(b'm');
                }

                let mut run_len = 0;
                while x < width {
                    let r_idx = row_offset + x;
                    let c = self.current[r_idx];
                    if c.fg != run_fg || c.bg != run_bg || c.bold != run_bold || c.underline != run_underline {
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

        write_stdout_bytes(&self.write_buf)?;
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

        // Pure black and white remain untouched for clean run-length batching
        assert_eq!(dither_truecolor_cell(Color::WHITE, 2, 1), Color::WHITE);
        assert_eq!(dither_truecolor_cell(Color::BLACK, 3, 2), Color::BLACK);
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
}
