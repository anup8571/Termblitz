#![allow(dead_code)]
use std::io::{self, Write};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cell {
    pub ch: char,
    pub fg: Color,
    pub bg: Color,
    pub bold: bool,
    pub underline: bool,
}

impl Cell {
    pub const EMPTY: Cell = Cell {
        ch: ' ',
        fg: Color::WHITE,
        bg: Color::BLACK,
        bold: false,
        underline: false,
    };

    pub const INVALID: Cell = Cell {
        ch: '\0',
        fg: Color::BLACK,
        bg: Color::BLACK,
        bold: false,
        underline: false,
    };

    pub fn new(ch: char, fg: Color, bg: Color) -> Self {
        Self {
            ch,
            fg,
            bg,
            bold: false,
            underline: false,
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
        let mut curr_x = x;
        for ch in s.chars() {
            if curr_x >= self.width {
                break;
            }
            self.set_cell(
                curr_x,
                y,
                Cell {
                    ch,
                    fg,
                    bg,
                    bold,
                    underline,
                },
            );
            curr_x += 1;
        }
    }

    /// Prints a string directly over existing ANSI cells while eliminating blurry background text bleed
    /// and enforcing minimum luminance contrast so overlaid terminal characters are always razor-sharp.
    pub fn put_str_with_existing_bg(
        &mut self,
        x: u16,
        y: u16,
        s: &str,
        fg: Color,
        fallback_bg: Color,
        bold: bool,
        underline: bool,
    ) {
        let mut curr_x = x;
        for ch in s.chars() {
            if curr_x >= self.width || y >= self.height {
                break;
            }
            let idx = (y as usize) * (self.width as usize) + (curr_x as usize);
            if idx < self.current.len() {
                let cell = &self.current[idx];
                let clean_bg = if cell.ch == '▀' {
                    let lum_fg = cell.fg.luma_fast();
                    let lum_bg = cell.bg.luma_fast();
                    let text_lum = fg.luma_fast();
                    if text_lum < 128 {
                        if lum_fg >= lum_bg { cell.fg } else { cell.bg }
                    } else if lum_fg <= lum_bg {
                        cell.fg
                    } else {
                        cell.bg
                    }
                } else if cell.bg != Color::BLACK || cell.ch == ' ' {
                    cell.bg
                } else {
                    fallback_bg
                };

                // Guarantee readability if background and foreground have insufficient contrast
                let bg_lum = clean_bg.luma_fast();
                let fg_lum = fg.luma_fast();
                let contrast = bg_lum.abs_diff(fg_lum);
                let effective_fg = if contrast < 65 {
                    if bg_lum > 128 {
                        Color::new(18, 20, 26)
                    } else {
                        Color::new(245, 247, 250)
                    }
                } else {
                    fg
                };

                self.current[idx] = Cell {
                    ch,
                    fg: effective_fg,
                    bg: clean_bg,
                    bold,
                    underline,
                };
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

        if w <= 1 && h <= 1 {
            return px_buf.get_pixel(x0.min(px_buf.width - 1), y0.min(px_buf.height - 1));
        }
        if w <= 2 && h <= 2 {
            let cx = x0 + w / 2;
            let cy = y0 + h / 2;
            return px_buf.get_pixel(cx.min(px_buf.width - 1), cy.min(px_buf.height - 1));
        }

        let cx = x0 + w / 2;
        let cy = y0 + h / 2;
        let center = px_buf.get_pixel(cx.min(px_buf.width - 1), cy.min(px_buf.height - 1));

        let step_x = ((w + 3) / 4).max(1);
        let step_y = ((h + 3) / 4).max(1);

        let mut min_lum = 255u32;
        let mut min_color = center;
        let mut max_lum = 0u32;
        let mut max_color = center;
        let mut sum_r = 0u32;
        let mut sum_g = 0u32;
        let mut sum_b = 0u32;
        let mut count = 0u32;

        let mut sy = y0;
        while sy < y1 {
            let mut sx = x0;
            while sx < x1 {
                let c = px_buf.get_pixel(sx.min(px_buf.width - 1), sy.min(px_buf.height - 1));
                let lum = c.luma_fast();
                if lum < min_lum {
                    min_lum = lum;
                    min_color = c;
                }
                if lum > max_lum {
                    max_lum = lum;
                    max_color = c;
                }
                sum_r += c.r as u32;
                sum_g += c.g as u32;
                sum_b += c.b as u32;
                count += 1;
                sx += step_x;
            }
            sy += step_y;
        }

        if count == 0 {
            return center;
        }

        let avg_r = (sum_r / count) as u8;
        let avg_g = (sum_g / count) as u8;
        let avg_b = (sum_b / count) as u8;
        let avg_lum = (sum_r * 77 + sum_g * 150 + sum_b * 29) >> 8;
        let avg_lum = avg_lum / count;

        let dark_contrast = avg_lum.saturating_sub(min_lum);
        let light_contrast = max_lum.saturating_sub(avg_lum);

        if dark_contrast >= 35 && dark_contrast >= light_contrast {
            Color::new(
                ((min_color.r as u32 * 3 + avg_r as u32 * 2) / 5) as u8,
                ((min_color.g as u32 * 3 + avg_g as u32 * 2) / 5) as u8,
                ((min_color.b as u32 * 3 + avg_b as u32 * 2) / 5) as u8,
            )
        } else if light_contrast >= 35 {
            Color::new(
                ((max_color.r as u32 * 3 + avg_r as u32 * 2) / 5) as u8,
                ((max_color.g as u32 * 3 + avg_g as u32 * 2) / 5) as u8,
                ((max_color.b as u32 * 3 + avg_b as u32 * 2) / 5) as u8,
            )
        } else {
            Color::new(avg_r, avg_g, avg_b)
        }
    }

    /// Renders a PixelBuffer into the terminal using Unicode half-blocks '▀' (upper half = fg, lower half = bg).
    pub fn blit_pixel_buffer(&mut self, px_buf: &PixelBuffer, start_y: u16, max_y: u16) {
        let cols = self.width as usize;
        let rows = (max_y.saturating_sub(start_y)) as usize;
        let target_h = rows * 2;
        let target_w = cols;

        if target_w == 0 || target_h == 0 || px_buf.width == 0 || px_buf.height == 0 {
            return;
        }

        for row in 0..rows {
            let term_y = start_y + (row as u16);
            if term_y >= self.height {
                break;
            }
            let top_y0 = (row * 2 * px_buf.height) / target_h;
            let top_y1 = ((row * 2 + 1) * px_buf.height) / target_h;
            let bot_y0 = top_y1;
            let bot_y1 = ((row * 2 + 2) * px_buf.height) / target_h;

            for col in 0..cols {
                let x0 = (col * px_buf.width) / target_w;
                let x1 = ((col + 1) * px_buf.width) / target_w;

                let top_color = Self::sample_subpixel_box(px_buf, x0, x1, top_y0, top_y1);
                let bot_color = Self::sample_subpixel_box(px_buf, x0, x1, bot_y0, bot_y1);

                let cell = if top_color == bot_color {
                    Cell {
                        ch: ' ',
                        fg: Color::WHITE,
                        bg: top_color,
                        bold: false,
                        underline: false,
                    }
                } else {
                    Cell {
                        ch: '▀',
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

        let mut rgba = Vec::with_capacity(px_buf.width * px_buf.height * 4);
        for c in &px_buf.data {
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
                push_u16(&mut out, px_buf.width as u16);
                out.extend_from_slice(b",v=");
                push_u16(&mut out, px_buf.height as u16);
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

    /// Sixel Graphics Protocol generator (6-pixel vertical sliver encoding)
    pub fn blit_sixel(&mut self, px_buf: &PixelBuffer, start_y: u16, _max_y: u16) {
        if px_buf.width == 0 || px_buf.height == 0 {
            return;
        }

        let mut out = Vec::with_capacity(px_buf.width * px_buf.height + 1024);

        out.extend_from_slice(b"\x1b[");
        push_u16(&mut out, start_y + 1);
        out.extend_from_slice(b";1H");

        out.extend_from_slice(b"\x1bPq\"1;1;");
        push_u16(&mut out, px_buf.width as u16);
        out.push(b';');
        push_u16(&mut out, px_buf.height as u16);

        let colors = [
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

        for (idx, (r, g, b)) in colors.iter().enumerate() {
            out.push(b'#');
            push_u8(&mut out, idx as u8);
            out.extend_from_slice(b";2;");
            push_u8(&mut out, *r as u8);
            out.push(b';');
            push_u8(&mut out, *g as u8);
            out.push(b';');
            push_u8(&mut out, *b as u8);
        }

        let map_color_to_idx = |c: Color| -> u8 {
            if c.r > 240 && c.g > 80 && c.g < 140 && c.b < 40 {
                return 10;
            }
            if c.r > 230 && c.g > 230 && c.b > 220 {
                return 11;
            }
            let luma = c.luma_fast();
            if luma < 30 {
                0
            } else if luma > 225 {
                1
            } else if c.r > 180 && c.g < 100 && c.b < 100 {
                2
            } else if c.g > 160 && c.r < 100 {
                3
            } else if c.b > 180 && c.r < 100 {
                4
            } else if c.r > 180 && c.g > 180 && c.b < 100 {
                5
            } else if c.g > 160 && c.b > 160 {
                6
            } else {
                8
            }
        };

        let width = px_buf.width;
        let height = px_buf.height;
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
                            let c = px_buf.get_pixel(x, y);
                            if map_color_to_idx(c) == col_idx {
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
    fn test_pixel_buffer_scaling() {
        let mut buf = PixelBuffer::new(100, 100, Color::RED);
        buf.set_pixel(0, 0, Color::BLUE);
        assert_eq!(buf.get_pixel(0, 0), Color::BLUE);
        assert_eq!(buf.get_pixel(1, 1), Color::RED);
    }

    #[test]
    fn test_put_str_with_existing_bg() {
        let mut renderer = DiffRenderer::new(10, 5);
        renderer.set_cell(0, 0, Cell {
            ch: '▀',
            fg: Color::BLACK,
            bg: Color::WHITE,
            bold: false,
            underline: false,
        });

        renderer.put_str_with_existing_bg(0, 0, "A", Color::BLACK, Color::LIGHT_GRAY, true, false);

        let cell = renderer.current[0];
        assert_eq!(cell.ch, 'A');
        assert_eq!(cell.fg, Color::BLACK);
        assert_eq!(cell.bg, Color::WHITE);
        assert!(cell.bold);
    }
}
