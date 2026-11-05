#![allow(dead_code)]
use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::renderer::{display_width, Cell, Color, DiffRenderer, PixelBuffer};
use crate::upscale::UpscaleMode;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RenderMode {
    Native,    // Option B: HTML/DOM text parser, ultra-fast, zero Chromium overhead
    CdpHybrid, // Option A + C: ANSI half-block background + real terminal character DOM overlay + mobile CSS reflow
    CdpRaw,    // Option C: Pure ANSI bitmap with mobile viewport reflow, for viewing full images/canvas without text overlay
}

impl RenderMode {
    pub fn next(self, has_chromium: bool) -> Self {
        if !has_chromium {
            return RenderMode::Native;
        }
        match self {
            RenderMode::Native => RenderMode::CdpHybrid,
            RenderMode::CdpHybrid => RenderMode::CdpRaw,
            RenderMode::CdpRaw => RenderMode::Native,
        }
    }

    pub fn is_cdp(self) -> bool {
        matches!(self, RenderMode::CdpHybrid | RenderMode::CdpRaw)
    }

    pub fn name(self) -> &'static str {
        match self {
            RenderMode::Native => "Native",
            RenderMode::CdpHybrid => "Hybrid",
            RenderMode::CdpRaw => "Raw",
        }
    }
}

pub type EngineMode = RenderMode;

#[derive(Clone, Debug)]
pub struct TextOverlaySpan {
    pub text: String,
    pub px_x: i32,
    pub px_y: i32,
    pub px_w: i32,
    pub px_h: i32,
    pub fg: Color,
    pub bg: Option<Color>,
    pub bold: bool,
    pub is_link: bool,
    pub is_input: bool,
    pub link_url: Option<String>,
}

#[derive(Clone, Debug)]
pub struct InteractiveLink {
    pub x: u16,
    pub y: u16,
    pub width: u16,
    pub url: String,
}

#[derive(Clone, Debug)]
pub struct TextSpan {
    pub text: String,
    pub fg: Color,
    pub bg: Option<Color>,
    pub bold: bool,
    pub underline: bool,
    pub link_url: Option<String>,
}

#[derive(Clone, Debug)]
pub struct RenderedLine {
    pub spans: Vec<TextSpan>,
    pub row_bg: Option<Color>,
    pub is_full_width_bg: bool,
}

/// Computes logical CSS viewport dimensions that preserve the terminal's 1:2 character cell aspect ratio
/// without relying on CSS style.zoom (which distorts getBoundingClientRect coordinates).
pub fn compute_viewport_size(cols: u16, rows: u16, zoom: f32) -> (u32, u32) {
    let safe_zoom = zoom.clamp(0.5, 3.0);
    let base_w: f32 = if cols <= 68 {
        360.0
    } else if cols <= 100 {
        480.0
    } else {
        (cols as f32) * 7.0
    };
    let width_px = (base_w / safe_zoom).round().clamp(240.0, 1280.0) as u32;
    let view_rows = rows.saturating_sub(1).max(10) as f32;
    let safe_cols = cols.max(20) as f32;
    let height_px = ((width_px as f32 * (view_rows * 2.0)) / safe_cols)
        .round()
        .clamp(320.0, 2400.0) as u32;
    (width_px, height_px)
}

/// Checks if a URL or domain belongs to a known ad/tracker network (Brave-style shield filter)
pub fn is_ad_or_tracker_url(url: &str) -> bool {
    let u = url.to_lowercase();
    const AD_PATTERNS: &[&str] = &[
        "doubleclick.net",
        "googlesyndication.com",
        "googleadservices.com",
        "adservice.google.",
        "moatads.com",
        "taboola.com",
        "outbrain.com",
        "criteo.com",
        "amazon-adsystem.com",
        "adsafeprotected.com",
        "scorecardresearch.com",
        "quantserve.com",
        "adnxs.com",
        "rubiconproject.com",
        "pubmatic.com",
        "openx.net",
        "smartadserver.com",
        "popads.net",
        "popcash.net",
        "google-analytics.com",
        "googletagmanager.com",
        "fbevents.js",
        "/pagead/",
        "/ads/",
    ];
    AD_PATTERNS.iter().any(|pat| u.contains(pat))
}

/// Decodes all named and numeric (hex/decimal) HTML entities
pub fn decode_html_entities(input: &str) -> String {
    if !input.contains('&') {
        return input.to_string();
    }
    let mut out = String::with_capacity(input.len());
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        if chars[i] == '&' {
            let mut end = i + 1;
            while end < len && end < i + 12 && chars[end] != ';' && chars[end] != ' ' && chars[end] != '<' {
                end += 1;
            }

            if end < len && chars[end] == ';' {
                let entity: String = chars[i + 1..end].iter().collect();
                let decoded = match entity.as_str() {
                    "nbsp" => Some(' '),
                    "amp" => Some('&'),
                    "quot" => Some('"'),
                    "apos" => Some('\''),
                    "lt" => Some('<'),
                    "gt" => Some('>'),
                    "mdash" => Some('—'),
                    "ndash" => Some('–'),
                    "hellip" => Some('…'),
                    "bull" => Some('•'),
                    "trade" => Some('™'),
                    "copy" => Some('©'),
                    "reg" => Some('®'),
                    "deg" => Some('°'),
                    "plusmn" => Some('±'),
                    "times" => Some('×'),
                    "divide" => Some('÷'),
                    "lsquo" | "rsquo" => Some('\''),
                    "ldquo" | "rdquo" => Some('"'),
                    s if s.starts_with("#x") || s.starts_with("#X") => {
                        u32::from_str_radix(&s[2..], 16)
                            .ok()
                            .and_then(std::char::from_u32)
                    }
                    s if s.starts_with('#') => {
                        s[1..].parse::<u32>()
                            .ok()
                            .and_then(std::char::from_u32)
                    }
                    _ => None,
                };

                if let Some(ch) = decoded {
                    out.push(ch);
                    i = end + 1;
                    continue;
                }
            }
        }

        out.push(chars[i]);
        i += 1;
    }
    out
}

pub fn parse_color_from_str(s: &str) -> Option<Color> {
    let s = s.trim().to_lowercase();
    if s.starts_with('#') {
        return Color::from_hex(&s);
    }
    if s.starts_with("rgb(") && s.ends_with(')') {
        let inner = &s[4..s.len() - 1];
        let parts: Vec<&str> = inner.split(',').map(|p| p.trim()).collect();
        if parts.len() == 3 {
            let r = parts[0].parse::<u8>().ok()?;
            let g = parts[1].parse::<u8>().ok()?;
            let b = parts[2].parse::<u8>().ok()?;
            return Some(Color::new(r, g, b));
        }
    }
    if s.starts_with("rgba(") && s.ends_with(')') {
        let inner = &s[5..s.len() - 1];
        let parts: Vec<&str> = inner.split(',').map(|p| p.trim()).collect();
        if parts.len() >= 3 {
            if parts.len() == 4 {
                let alpha = parts[3].parse::<f32>().unwrap_or(1.0);
                if alpha <= 0.05 {
                    return None; // Fully transparent
                }
            }
            let r = parts[0].parse::<u8>().ok()?;
            let g = parts[1].parse::<u8>().ok()?;
            let b = parts[2].parse::<u8>().ok()?;
            return Some(Color::new(r, g, b));
        }
    }
    match s.as_str() {
        "white" => Some(Color::WHITE),
        "black" => Some(Color::BLACK),
        "red" => Some(Color::RED),
        "green" => Some(Color::GREEN),
        "blue" => Some(Color::BLUE),
        "yellow" => Some(Color::YELLOW),
        "cyan" => Some(Color::CYAN),
        "magenta" => Some(Color::MAGENTA),
        "gray" | "grey" => Some(Color::GRAY),
        "lightgray" | "lightgrey" => Some(Color::LIGHT_GRAY),
        "darkgray" | "darkgrey" => Some(Color::DARK_GRAY),
        "orange" => Some(Color::new(255, 102, 0)),
        _ => None,
    }
}

pub fn extract_attr<'a>(tag: &'a str, attr: &str) -> Option<&'a str> {
    let lower_tag = tag.to_lowercase();
    let pattern = format!("{}=", attr.to_lowercase());
    let pos = lower_tag.find(&pattern)?;
    let rest = &tag[pos + pattern.len()..];
    let quote = rest.chars().next()?;
    if quote == '"' || quote == '\'' {
        rest[1..].split(quote).next()
    } else {
        rest.split_whitespace().next()
    }
}

pub fn extract_style_color(style: &str) -> Option<Color> {
    for decl in style.split(';') {
        let mut parts = decl.splitn(2, ':');
        if let (Some(k), Some(v)) = (parts.next(), parts.next()) {
            let key = k.trim().to_lowercase();
            if key == "color" {
                return parse_color_from_str(v);
            }
        }
    }
    None
}

pub fn extract_style_bg(style: &str) -> Option<Color> {
    for decl in style.split(';') {
        let mut parts = decl.splitn(2, ':');
        if let (Some(k), Some(v)) = (parts.next(), parts.next()) {
            let key = k.trim().to_lowercase();
            if key == "background" || key == "background-color" {
                return parse_color_from_str(v);
            }
        }
    }
    None
}

pub struct NativeEngine {
    pub current_url: String,
    pub raw_html: String,
    pub title: String,
    pub body_bg: Color,
    pub lines: Vec<RenderedLine>,
    pub links: Vec<InteractiveLink>,
    pub scroll_y: usize,
    pub max_scroll: usize,
    pub adblock_enabled: bool,
    pub ads_blocked: usize,
    pub upscale_mode: UpscaleMode,
    pub dithering_enabled: bool,
}

impl NativeEngine {
    pub fn new() -> Self {
        Self {
            current_url: "about:home".to_string(),
            raw_html: String::new(),
            title: "termblitz home".to_string(),
            body_bg: Color::new(246, 246, 239),
            lines: Vec::new(),
            links: Vec::new(),
            scroll_y: 0,
            max_scroll: 0,
            adblock_enabled: true,
            ads_blocked: 0,
            upscale_mode: UpscaleMode::FsrOnly,
            dithering_enabled: true,
        }
    }

    pub fn load_url(&mut self, url: &str, cols: u16) {
        self.current_url = url.to_string();
        self.scroll_y = 0;

        if url == "about:home" || url.is_empty() {
            self.raw_html = Self::default_home_html();
            self.parse_and_layout(cols);
            return;
        }

        let output = Command::new("curl")
            .arg("-sL")
            .arg("-A")
            .arg("Mozilla/5.0 (Linux; Android 14; Mobile) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36")
            .arg("--max-time")
            .arg("10")
            .arg(url)
            .output();

        match output {
            Ok(res) if res.status.success() => {
                self.raw_html = String::from_utf8_lossy(&res.stdout).to_string();
            }
            Ok(res) => {
                self.raw_html = format!(
                    "<html><body><h1>Network Error</h1><p>Failed to load {}: exit code {}</p></body></html>",
                    url, res.status
                );
            }
            Err(e) => {
                self.raw_html = format!(
                    "<html><body><h1>Fetch Error</h1><p>Failed to execute curl: {}</p></body></html>",
                    e
                );
            }
        }

        self.parse_and_layout(cols);
    }

    fn default_home_html() -> String {
        r##"<!DOCTYPE html>
<html>
<head><title>termblitz :: Native Visual Terminal Browser</title></head>
<body bgcolor="#f6f6ef">
    <table width="100%" bgcolor="#ff6600">
        <tr><td><b>&nbsp;⚡ termblitz :: Ultra-Fast Terminal Visual Web Browser</b></td></tr>
    </table>
    <br>
    <p>Welcome to <b>termblitz</b> &mdash; engineered for maximum performance on Android Termux, Linux, macOS &amp; Windows.</p>
    <hr>
    <h2>Quick Jump Links:</h2>
    <ul>
        <li><a href="https://www.google.com">Google Search (Hybrid Visual Mode)</a></li>
        <li><a href="https://news.ycombinator.com">Hacker News (Y Combinator)</a></li>
        <li><a href="https://en.wikipedia.org/wiki/Main_Page">Wikipedia: The Free Encyclopedia</a></li>
        <li><a href="https://lobste.rs">Lobsters Tech Community</a></li>
        <li><a href="https://lite.cnn.com">CNN Lite News</a></li>
        <li><a href="https://text.npr.org">NPR Text-Only News</a></li>
    </ul>
    <hr>
    <h2>Touch Gestures &amp; Keyboard Controls:</h2>
    <p>&bull; <b>Mode Toggle [m / Tap Top-Left Badge]:</b> Cycle <i>Native</i> &rarr; <i>Hybrid</i> &rarr; <i>Raw</i>.</p>
    <p>&bull; <b>Search &amp; URL [u / / / Tap Top Bar]:</b> Enter any URL or Google search query.</p>
    <p>&bull; <b>Touch Gestures:</b> Swipe Up/Down to scroll, Swipe Left/Right for Back/Forward, Tap links or search boxes directly.</p>
    <p>&bull; <b>Brave Shield [a]:</b> Toggle integrated network &amp; cosmetic ad/tracker blocker.</p>
</body>
</html>"##
            .to_string()
    }

    /// Pre-strips `<script>...</script>`, `<style>...</style>`, `<svg>...</svg>`, `<head>...</head>`, and `<!--...-->`
    /// so `<` characters inside inline JavaScript (e.g. `for (i=0; i<10; i++)`) never corrupt tag parsing.
    fn strip_non_content_blocks(html: &str) -> String {
        let mut out = String::with_capacity(html.len());
        let bytes = html.as_bytes();
        let len = bytes.len();
        let mut i = 0;

        let starts_with_ci = |pos: usize, needle: &[u8]| -> bool {
            if pos + needle.len() > len {
                return false;
            }
            bytes[pos..pos + needle.len()]
                .iter()
                .zip(needle.iter())
                .all(|(a, b)| a.eq_ignore_ascii_case(b))
        };

        while i < len {
            if starts_with_ci(i, b"<!--") {
                i += 4;
                while i < len && !starts_with_ci(i, b"-->") {
                    i += 1;
                }
                if i < len {
                    i += 3;
                }
                continue;
            }

            let skip_tag: Option<&[u8]> = if starts_with_ci(i, b"<script") {
                Some(b"</script>")
            } else if starts_with_ci(i, b"<style") {
                Some(b"</style>")
            } else if starts_with_ci(i, b"<svg") {
                Some(b"</svg>")
            } else if starts_with_ci(i, b"<noscript") {
                Some(b"</noscript>")
            } else if starts_with_ci(i, b"<head") {
                Some(b"</head>")
            } else {
                None
            };

            if let Some(close_tag) = skip_tag {
                let next_ch = bytes.get(i + close_tag.len() - 2).copied().unwrap_or(b' ');
                if next_ch == b'>' || next_ch == b' ' || next_ch == b'\t' || next_ch == b'\n' || next_ch == b'\r' {
                    i += close_tag.len() - 2;
                    while i < len && !starts_with_ci(i, close_tag) {
                        i += 1;
                    }
                    if i < len {
                        i += close_tag.len();
                    }
                    continue;
                }
            }

            // Copy next valid UTF-8 char
            if let Some(ch) = html[i..].chars().next() {
                out.push(ch);
                i += ch.len_utf8();
            } else {
                i += 1;
            }
        }
        out
    }

    /// Pure Rust streaming HTML/CSS layout engine with Table Row Horizontal Flow & Color Palettes
    pub fn parse_and_layout(&mut self, cols: u16) {
        self.lines.clear();
        self.links.clear();

        let lower_html = self.raw_html.to_lowercase();
        if lower_html.contains("bgcolor=\"#f6f6ef\"")
            || lower_html.contains("bgcolor='#f6f6ef'")
            || lower_html.contains("bgcolor=#f6f6ef")
            || self.current_url.contains("news.ycombinator.com")
        {
            self.body_bg = Color::new(246, 246, 239);
        } else if lower_html.contains("bgcolor=\"#ffffff\"")
            || lower_html.contains("bgcolor='#ffffff'")
            || lower_html.contains("bgcolor=#ffffff")
            || lower_html.contains("bgcolor=\"white\"")
            || self.current_url.contains("wikipedia.org")
            || self.current_url.contains("google.com")
        {
            self.body_bg = Color::WHITE;
        } else if let Some(body_pos) = lower_html.find("<body") {
            let rest = &self.raw_html[body_pos..];
            if let Some(end_body) = rest.find('>') {
                let body_tag = &rest[..end_body];
                if let Some(bg_str) = extract_attr(body_tag, "bgcolor") {
                    if let Some(c) = parse_color_from_str(bg_str) {
                        self.body_bg = c;
                    }
                }
            }
        }

        let is_light_theme = self.body_bg.luma_fast() > 128;
        let default_fg = if is_light_theme {
            Color::new(20, 20, 20)
        } else {
            Color::new(220, 220, 220)
        };
        let link_color = if is_light_theme {
            Color::new(10, 40, 180)
        } else {
            Color::new(80, 160, 255)
        };
        let subtext_color = Color::new(130, 130, 130);

        let wrap_width = (cols as usize).saturating_sub(2).max(20);

        let cleaned_html = Self::strip_non_content_blocks(&self.raw_html);

        let mut in_tag = false;
        let mut in_table = false;
        let mut in_table_row = false;
        let mut current_tag = String::new();
        let mut text_acc = String::new();

        let mut current_fg = default_fg;
        let mut current_bg: Option<Color> = None;
        let mut current_row_bg: Option<Color> = None;
        let mut is_full_width_row = false;
        let mut is_bold = false;
        let mut is_underline = false;
        let mut active_link: Option<String> = None;

        let mut block_lines: Vec<RenderedLine> = Vec::new();
        let mut current_spans: Vec<TextSpan> = Vec::new();
        let mut current_line_len = 0;

        let flush_current_line = |spans: &mut Vec<TextSpan>,
                                  block: &mut Vec<RenderedLine>,
                                  line_len: &mut usize,
                                  row_bg: Option<Color>,
                                  is_full_width: bool| {
            if !spans.is_empty() {
                let line_spans = std::mem::take(spans);
                block.push(RenderedLine {
                    spans: line_spans,
                    row_bg,
                    is_full_width_bg: is_full_width,
                });
                *line_len = 0;
            }
        };

        let flush_word = |spans: &mut Vec<TextSpan>,
                          block: &mut Vec<RenderedLine>,
                          line_len: &mut usize,
                          word: &str,
                          fg: Color,
                          bg: Option<Color>,
                          row_bg: Option<Color>,
                          is_full_width: bool,
                          bold: bool,
                          underline: bool,
                          link: Option<String>,
                          max_w: usize| {
            if word.is_empty() {
                return;
            }
            let word_w = display_width(word);
            if *line_len + word_w > max_w && *line_len > 0 {
                let line_spans = std::mem::take(spans);
                block.push(RenderedLine {
                    spans: line_spans,
                    row_bg,
                    is_full_width_bg: is_full_width,
                });
                *line_len = 0;
            }

            spans.push(TextSpan {
                text: word.to_string(),
                fg,
                bg,
                bold,
                underline,
                link_url: link,
            });
            *line_len += word_w;
        };

        let chars: Vec<char> = cleaned_html.chars().collect();
        let mut i = 0;
        let len = chars.len();

        while i < len {
            let c = chars[i];

            if c == '<' {
                if !text_acc.is_empty() {
                    let decoded = decode_html_entities(&text_acc);
                    let cleaned = decoded.replace('\n', " ").replace('\r', " ").replace('\t', " ");
                    for word in cleaned.split_whitespace() {
                        flush_word(
                            &mut current_spans,
                            &mut block_lines,
                            &mut current_line_len,
                            &format!("{} ", word),
                            current_fg,
                            current_bg,
                            current_row_bg,
                            is_full_width_row,
                            is_bold,
                            is_underline,
                            active_link.clone(),
                            wrap_width,
                        );
                    }
                    text_acc.clear();
                }

                in_tag = true;
                current_tag.clear();
                i += 1;
                continue;
            } else if c == '>' && in_tag {
                in_tag = false;
                let tag_raw = current_tag.trim();
                let tag_str = tag_raw.to_lowercase();
                let tag_name = tag_str.split_whitespace().next().unwrap_or("");

                if let Some(bg_str) = extract_attr(tag_raw, "bgcolor") {
                    if let Some(color) = parse_color_from_str(bg_str) {
                        current_row_bg = Some(color);
                        is_full_width_row = true;
                        if color == Color::new(255, 102, 0) {
                            current_fg = Color::BLACK;
                        }
                    }
                }

                if let Some(style_str) = extract_attr(tag_raw, "style") {
                    if let Some(fg) = extract_style_color(style_str) {
                        current_fg = fg;
                    }
                    if let Some(bg) = extract_style_bg(style_str) {
                        current_bg = Some(bg);
                    }
                }

                if tag_str.contains("class=\"subtext\"")
                    || tag_str.contains("class='subtext'")
                    || tag_str.contains("class=\"comhead\"")
                    || tag_str.contains("class=\"score\"")
                    || tag_str.contains("class=\"age\"")
                {
                    current_fg = subtext_color;
                } else if tag_str.contains("class=\"titleline\"") || tag_str.contains("class=\"title\"") {
                    current_fg = default_fg;
                }

                match tag_name {
                    "table" => {
                        in_table = true;
                    }
                    "/table" => {
                        in_table = false;
                        flush_current_line(
                            &mut current_spans,
                            &mut block_lines,
                            &mut current_line_len,
                            current_row_bg,
                            is_full_width_row,
                        );
                        current_row_bg = None;
                        is_full_width_row = false;
                        current_fg = default_fg;
                    }
                    "tr" => {
                        in_table_row = true;
                        flush_current_line(
                            &mut current_spans,
                            &mut block_lines,
                            &mut current_line_len,
                            current_row_bg,
                            is_full_width_row,
                        );
                        if tag_str.contains("bgcolor=\"#ff6600\"") || tag_str.contains("bgcolor='#ff6600'") {
                            current_row_bg = Some(Color::new(255, 102, 0));
                            is_full_width_row = true;
                            current_fg = Color::BLACK;
                        } else {
                            current_row_bg = None;
                            is_full_width_row = false;
                            current_fg = default_fg;
                        }
                    }
                    "/tr" => {
                        flush_current_line(
                            &mut current_spans,
                            &mut block_lines,
                            &mut current_line_len,
                            current_row_bg,
                            is_full_width_row,
                        );
                        in_table_row = false;
                        current_row_bg = None;
                        is_full_width_row = false;
                        current_fg = default_fg;
                    }
                    "td" | "th" => {
                        if in_table_row {
                            if !current_spans.is_empty() {
                                current_spans.push(TextSpan {
                                    text: "  ".to_string(),
                                    fg: default_fg,
                                    bg: current_row_bg,
                                    bold: false,
                                    underline: false,
                                    link_url: None,
                                });
                                current_line_len += 2;
                            } else if tag_str.contains("colspan=\"2\"") || tag_str.contains("colspan='2'") {
                                current_spans.push(TextSpan {
                                    text: "    ".to_string(),
                                    fg: default_fg,
                                    bg: current_row_bg,
                                    bold: false,
                                    underline: false,
                                    link_url: None,
                                });
                                current_line_len += 4;
                            }
                        }
                    }
                    "/td" | "/th" => {}
                    "h1" => {
                        flush_current_line(
                            &mut current_spans,
                            &mut block_lines,
                            &mut current_line_len,
                            current_row_bg,
                            is_full_width_row,
                        );
                        current_fg = if is_light_theme { Color::new(10, 50, 120) } else { Color::CYAN };
                        is_bold = true;
                    }
                    "/h1" => {
                        flush_current_line(
                            &mut current_spans,
                            &mut block_lines,
                            &mut current_line_len,
                            current_row_bg,
                            is_full_width_row,
                        );
                        current_fg = default_fg;
                        is_bold = false;
                    }
                    "h2" => {
                        flush_current_line(
                            &mut current_spans,
                            &mut block_lines,
                            &mut current_line_len,
                            current_row_bg,
                            is_full_width_row,
                        );
                        current_fg = if is_light_theme { Color::new(120, 60, 10) } else { Color::YELLOW };
                        is_bold = true;
                    }
                    "/h2" => {
                        flush_current_line(
                            &mut current_spans,
                            &mut block_lines,
                            &mut current_line_len,
                            current_row_bg,
                            is_full_width_row,
                        );
                        current_fg = default_fg;
                        is_bold = false;
                    }
                    "h3" | "h4" | "h5" | "h6" => {
                        flush_current_line(
                            &mut current_spans,
                            &mut block_lines,
                            &mut current_line_len,
                            current_row_bg,
                            is_full_width_row,
                        );
                        current_fg = if is_light_theme { Color::new(10, 100, 40) } else { Color::GREEN };
                        is_bold = true;
                    }
                    "/h3" | "/h4" | "/h5" | "/h6" => {
                        flush_current_line(
                            &mut current_spans,
                            &mut block_lines,
                            &mut current_line_len,
                            current_row_bg,
                            is_full_width_row,
                        );
                        current_fg = default_fg;
                        is_bold = false;
                    }
                    "p" | "div" | "/p" | "/div" => {
                        if !in_table && !in_table_row && !current_spans.is_empty() {
                            flush_current_line(
                                &mut current_spans,
                                &mut block_lines,
                                &mut current_line_len,
                                current_row_bg,
                                is_full_width_row,
                            );
                        }
                    }
                    "br" => {
                        flush_current_line(
                            &mut current_spans,
                            &mut block_lines,
                            &mut current_line_len,
                            current_row_bg,
                            is_full_width_row,
                        );
                    }
                    "hr" => {
                        flush_current_line(
                            &mut current_spans,
                            &mut block_lines,
                            &mut current_line_len,
                            current_row_bg,
                            is_full_width_row,
                        );
                        block_lines.push(RenderedLine {
                            spans: vec![TextSpan {
                                text: "─".repeat(wrap_width),
                                fg: Color::GRAY,
                                bg: None,
                                bold: false,
                                underline: false,
                                link_url: None,
                            }],
                            row_bg: None,
                            is_full_width_bg: false,
                        });
                        current_line_len = 0;
                    }
                    "li" => {
                        flush_current_line(
                            &mut current_spans,
                            &mut block_lines,
                            &mut current_line_len,
                            current_row_bg,
                            is_full_width_row,
                        );
                        current_spans.push(TextSpan {
                            text: "  • ".to_string(),
                            fg: if is_light_theme { Color::new(160, 100, 0) } else { Color::YELLOW },
                            bg: None,
                            bold: true,
                            underline: false,
                            link_url: None,
                        });
                        current_line_len = 4;
                    }
                    "/li" => {
                        flush_current_line(
                            &mut current_spans,
                            &mut block_lines,
                            &mut current_line_len,
                            current_row_bg,
                            is_full_width_row,
                        );
                    }
                    "blockquote" => {
                        flush_current_line(
                            &mut current_spans,
                            &mut block_lines,
                            &mut current_line_len,
                            current_row_bg,
                            is_full_width_row,
                        );
                        current_spans.push(TextSpan {
                            text: "│ ".to_string(),
                            fg: Color::CYAN,
                            bg: None,
                            bold: true,
                            underline: false,
                            link_url: None,
                        });
                        current_line_len = 2;
                    }
                    "/blockquote" => {
                        flush_current_line(
                            &mut current_spans,
                            &mut block_lines,
                            &mut current_line_len,
                            current_row_bg,
                            is_full_width_row,
                        );
                    }
                    "code" | "pre" => {
                        current_fg = if is_light_theme { Color::new(0, 100, 0) } else { Color::GREEN };
                        current_bg = Some(if is_light_theme { Color::new(230, 230, 230) } else { Color::DARK_GRAY });
                    }
                    "/code" | "/pre" => {
                        current_fg = default_fg;
                        current_bg = None;
                    }
                    "b" | "strong" => {
                        is_bold = true;
                    }
                    "/b" | "/strong" => {
                        is_bold = false;
                    }
                    "i" | "em" => {
                        is_underline = true;
                    }
                    "/i" | "/em" => {
                        is_underline = false;
                    }
                    _ if tag_name == "a" || tag_name.starts_with("a ") => {
                        if let Some(href) = extract_attr(tag_raw, "href") {
                            let decoded_href = decode_html_entities(href);
                            if self.adblock_enabled && is_ad_or_tracker_url(&decoded_href) {
                                self.ads_blocked += 1;
                                active_link = None;
                            } else {
                                active_link = Some(decoded_href);
                            }
                        }
                        current_fg = link_color;
                        is_underline = true;
                    }
                    "/a" => {
                        active_link = None;
                        current_fg = default_fg;
                        is_underline = false;
                    }
                    _ => {}
                }

                i += 1;
                continue;
            }

            if in_tag {
                current_tag.push(c);
            } else {
                text_acc.push(c);
            }

            i += 1;
        }

        flush_current_line(
            &mut current_spans,
            &mut block_lines,
            &mut current_line_len,
            current_row_bg,
            is_full_width_row,
        );

        self.lines = block_lines;
        self.max_scroll = self.lines.len().saturating_sub(10);
    }

    pub fn render_to(&mut self, renderer: &mut DiffRenderer, start_y: u16, max_y: u16) {
        self.links.clear();
        let view_height = (max_y - start_y) as usize;

        for (row_idx, line_idx) in (self.scroll_y..(self.scroll_y + view_height)).enumerate() {
            let term_y = start_y + (row_idx as u16);
            if term_y >= max_y {
                break;
            }

            if let Some(line) = self.lines.get(line_idx) {
                let row_bg = line.row_bg.unwrap_or(self.body_bg);

                for x in 0..renderer.width {
                    renderer.set_cell(x, term_y, Cell::new(' ', Color::WHITE, row_bg));
                }

                let mut term_x = if line.is_full_width_bg { 0 } else { 1 };
                for span in &line.spans {
                    let span_len = display_width(&span.text) as u16;
                    let span_bg = span.bg.unwrap_or(row_bg);

                    if let Some(url) = &span.link_url {
                        self.links.push(InteractiveLink {
                            x: term_x,
                            y: term_y,
                            width: span_len,
                            url: url.clone(),
                        });
                    }

                    renderer.put_str(
                        term_x,
                        term_y,
                        &span.text,
                        span.fg,
                        span_bg,
                        span.bold,
                        span.underline,
                    );
                    term_x += span_len;
                }
            } else {
                for x in 0..renderer.width {
                    renderer.set_cell(x, term_y, Cell::new(' ', Color::WHITE, self.body_bg));
                }
            }
        }
    }

    pub fn hit_test(&self, click_x: u16, click_y: u16) -> Option<String> {
        for link in &self.links {
            if click_y == link.y && click_x >= link.x && click_x < (link.x + link.width) {
                return Some(link.url.clone());
            }
        }
        None
    }
}

/// Base64 decoder for CDP screenshot payloads
pub fn base64_decode(input: &str) -> Vec<u8> {
    let mut table = [255u8; 256];
    for (i, &b) in b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/".iter().enumerate() {
        table[b as usize] = i as u8;
    }
    let mut out = Vec::with_capacity(input.len() * 3 / 4);
    let mut buf = 0u32;
    let mut bits = 0;
    for &b in input.as_bytes() {
        if b == b'=' || b == b'\r' || b == b'\n' || b == b' ' || b == b'"' {
            continue;
        }
        let val = table[b as usize];
        if val == 255 {
            continue;
        }
        buf = (buf << 6) | (val as u32);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
        }
    }
    out
}

/// Unescapes JSON string escape sequences (\n, \r, \t, \", \\, \uXXXX)
pub fn json_unescape(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('t') => out.push('\t'),
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('/') => out.push('/'),
                Some('u') => {
                    let mut hex = String::with_capacity(4);
                    for _ in 0..4 {
                        if let Some(&hc) = chars.peek() {
                            if hc.is_ascii_hexdigit() {
                                hex.push(hc);
                                chars.next();
                            }
                        }
                    }
                    if let Ok(code) = u32::from_str_radix(&hex, 16) {
                        if let Some(ch) = std::char::from_u32(code) {
                            out.push(ch);
                            continue;
                        }
                    }
                    out.push_str("\\u");
                    out.push_str(&hex);
                }
                Some(other) => out.push(other),
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

pub fn json_escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len() + 16);
    for c in input.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out
}

pub fn find_unescaped_quote(s: &str) -> Option<usize> {
    let mut escape = false;
    for (i, c) in s.char_indices() {
        if escape {
            escape = false;
        } else if c == '\\' {
            escape = true;
        } else if c == '"' {
            return Some(i);
        }
    }
    None
}

#[derive(Debug)]
pub enum CdpWorkerCommand {
    Navigate(String),
    Resize { cols: u16, rows: u16 },
    Scroll(i32),
    Click(u16, u16),
    SetZoom(f32),
    SetHideText(bool),
    SetAdBlock(bool),
    HistoryBack,
    HistoryForward,
    Stop,
}

struct WsConnection {
    stream: TcpStream,
    buf: Vec<u8>,
}

impl WsConnection {
    fn connect(port: u16, path: &str) -> Result<Self, String> {
        let mut stream = TcpStream::connect(("127.0.0.1", port))
            .map_err(|e| format!("Connect failed: {}", e))?;
        stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
        stream.set_write_timeout(Some(Duration::from_secs(5))).ok();

        let req = format!(
            "GET {} HTTP/1.1\r\n\
             Host: 127.0.0.1:{}\r\n\
             Upgrade: websocket\r\n\
             Connection: Upgrade\r\n\
             Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\
             Sec-WebSocket-Version: 13\r\n\r\n",
            path, port
        );
        stream.write_all(req.as_bytes()).map_err(|e| e.to_string())?;

        let mut resp = Vec::new();
        let mut buf = [0u8; 4096];
        while let Ok(n) = stream.read(&mut buf) {
            if n == 0 {
                break;
            }
            resp.extend_from_slice(&buf[..n]);
            if resp.windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
            if resp.len() > 16384 {
                return Err("WebSocket handshake response too large".to_string());
            }
        }

        if !String::from_utf8_lossy(&resp).contains("101") {
            return Err("WebSocket upgrade failed: HTTP 101 missing".to_string());
        }

        stream.set_read_timeout(Some(Duration::from_millis(12))).ok();

        Ok(Self {
            stream,
            buf: Vec::with_capacity(262144),
        })
    }

    fn send_text(&mut self, text: &str) -> Result<(), String> {
        self.send_frame(0x81, text.as_bytes())
    }

    fn send_frame(&mut self, opcode_byte: u8, payload: &[u8]) -> Result<(), String> {
        let len = payload.len();
        let mut frame = Vec::with_capacity(len + 14);

        frame.push(opcode_byte);

        if len <= 125 {
            frame.push(0x80 | (len as u8));
        } else if len <= 65535 {
            frame.push(0x80 | 126);
            frame.extend_from_slice(&(len as u16).to_be_bytes());
        } else {
            frame.push(0x80 | 127);
            frame.extend_from_slice(&(len as u64).to_be_bytes());
        }

        let mask = [0x12, 0x34, 0x56, 0x78];
        frame.extend_from_slice(&mask);

        for (i, &b) in payload.iter().enumerate() {
            frame.push(b ^ mask[i % 4]);
        }

        self.stream.write_all(&frame).map_err(|e| e.to_string())?;
        self.stream.flush().map_err(|e| e.to_string())?;
        Ok(())
    }

    fn try_extract_frame(&mut self) -> Option<String> {
        loop {
            if self.buf.len() < 2 {
                return None;
            }

            let opcode = self.buf[0] & 0x0F;
            let pay_len = (self.buf[1] & 0x7F) as usize;
            let mut idx = 2;
            let actual_len = if pay_len == 126 {
                if self.buf.len() < 4 {
                    return None;
                }
                let l = u16::from_be_bytes([self.buf[2], self.buf[3]]) as usize;
                idx = 4;
                l
            } else if pay_len == 127 {
                if self.buf.len() < 10 {
                    return None;
                }
                let mut arr = [0u8; 8];
                arr.copy_from_slice(&self.buf[2..10]);
                let l = u64::from_be_bytes(arr) as usize;
                idx = 10;
                l
            } else {
                pay_len
            };

            if self.buf.len() < idx + actual_len {
                return None;
            }

            let payload_bytes = self.buf[idx..idx + actual_len].to_vec();
            self.buf.drain(..idx + actual_len);

            if opcode == 0x9 {
                let _ = self.send_frame(0x8A, &payload_bytes);
                continue;
            }
            if opcode == 0x8 {
                return None;
            }
            return Some(String::from_utf8_lossy(&payload_bytes).into_owned());
        }
    }

    /// Reads all available socket chunks until a full WebSocket frame is assembled or socket times out
    fn poll_message(&mut self) -> Option<String> {
        loop {
            if let Some(msg) = self.try_extract_frame() {
                return Some(msg);
            }

            let mut chunk = [0u8; 65536];
            match self.stream.read(&mut chunk) {
                Ok(n) if n > 0 => {
                    self.buf.extend_from_slice(&chunk[..n]);
                }
                _ => break,
            }
        }

        self.try_extract_frame()
    }
}

const TEXT_EXTRACT_ID: u64 = 55555;

/// Injects mobile reflow CSS, cosmetic ad-blocking rules, and non-destructive text transparency (`-webkit-text-fill-color: transparent`).
/// Using `-webkit-text-fill-color: transparent` hides Chromium's raster glyphs from the screencast while preserving `getComputedStyle(p).color`!
fn build_page_style_js(hide_text: bool, adblock: bool) -> String {
    let raw_flag = if hide_text { "no" } else { "yes" };
    let ad_flag = if adblock { "yes" } else { "no" };
    format!(
        "(function(){{\
            var d=document.documentElement;\
            if(!d)return;\
            d.style.zoom='1';\
            d.setAttribute('data-termblitz-raw','{}');\
            d.setAttribute('data-termblitz-adblock','{}');\
            var head=document.head||d;\
            var s=document.getElementById('termblitz-core-style');\
            if(!s){{\
                s=document.createElement('style');\
                s.id='termblitz-core-style';\
                s.textContent='\
                    html,body{{max-width:100vw!important;overflow-x:hidden!important;box-sizing:border-box!important;word-break:break-word!important;-webkit-text-size-adjust:100%!important;}}\
                    img,video,canvas,svg{{max-width:100%!important;height:auto!important;}}\
                    html:not([data-termblitz-raw=yes]) *,html:not([data-termblitz-raw=yes]) *::before,html:not([data-termblitz-raw=yes]) *::after{{-webkit-text-fill-color:transparent!important;text-shadow:none!important;-webkit-text-stroke:0!important;}}\
                    html[data-termblitz-adblock=yes] ins.adsbygoogle,html[data-termblitz-adblock=yes] [id^=google_ads],html[data-termblitz-adblock=yes] [id^=div-gpt-ad],html[data-termblitz-adblock=yes] [data-text-ad],html[data-termblitz-adblock=yes] [aria-label=Ads],html[data-termblitz-adblock=yes] iframe[src*=doubleclick],html[data-termblitz-adblock=yes] iframe[id*=google_ads]{{display:none!important;}}\
                ';\
                head.appendChild(s);\
            }}\
        }})()",
        raw_flag, ad_flag
    )
}

fn request_text_extraction(ws: &mut WsConnection, seq: &mut u64) {
    *seq += 1;
    let js = "(function(){\
        var T=String.fromCharCode(9),N=String.fromCharCode(10),r=[],items=[];\
        if(!document.body)return '';\
        var vw=window.innerWidth||360,vh=window.innerHeight||640;\
        var docH=Math.max(document.body.scrollHeight||0,document.documentElement.scrollHeight||0,vh);\
        var maxS=Math.max(1,docH-vh);\
        var pct=Math.min(100,Math.max(0,Math.round(((window.scrollY||0)*100)/maxS)));\
        r.push('META'+T+vw+T+vh+T+pct+T+(location.href||''));\
        function getBg(el){\
            var bp=el;\
            while(bp&&bp!==document.documentElement){\
                var bc=getComputedStyle(bp).backgroundColor;\
                if(bc&&bc!=='rgba(0, 0, 0, 0)'&&bc!=='transparent')return bc;\
                bp=bp.parentElement;\
            }\
            return '';\
        }\
        var skipTags={SCRIPT:1,STYLE:1,NOSCRIPT:1,TEMPLATE:1,SVG:1,PATH:1,HEAD:1,META:1,TITLE:1,OBJECT:1};\
        var tw=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);\
        var n;\
        while(n=tw.nextNode()){\
            var t=n.textContent;\
            if(!t)continue;\
            var txt=t.split(T).join(' ').split(N).join(' ').trim();\
            if(!txt)continue;\
            var p=n.parentElement;\
            if(!p||skipTags[p.tagName])continue;\
            var s=getComputedStyle(p);\
            if(s.display==='none'||s.visibility==='hidden'||parseFloat(s.opacity)<0.05||parseFloat(s.fontSize)<2)continue;\
            if(s.clip==='rect(1px, 1px, 1px, 1px)'||s.clipPath==='inset(50%)'||((p.offsetWidth<=2||p.offsetHeight<=2)&&s.overflow==='hidden'))continue;\
            if(p.closest('[aria-hidden=true],[hidden],ins.adsbygoogle,[data-text-ad],[id^=google_ads]'))continue;\
            var rng=document.createRange();\
            rng.selectNodeContents(n);\
            var rects=rng.getClientRects();\
            if(!rects||!rects.length)continue;\
            var minL=99999,minT=99999,maxR=-99999,maxB=-99999,cnt=0;\
            for(var i=0;i<rects.length;i++){\
                var rc=rects[i];\
                if(rc.width>=1&&rc.height>=1){\
                    if(rc.left<minL)minL=rc.left;\
                    if(rc.top<minT)minT=rc.top;\
                    if(rc.right>maxR)maxR=rc.right;\
                    if(rc.bottom>maxB)maxB=rc.bottom;\
                    cnt++;\
                }\
            }\
            if(!cnt)continue;\
            var w=~~(maxR-minL),h=~~(maxB-minT),l=~~minL,top=~~minT;\
            if(w<2||h<2||maxB<0||minT>vh+20||maxR<0||minL>vw+20)continue;\
            var a=p.closest('a');\
            var href=a?(a.href||''):'';\
            var bg=getBg(p);\
            items.push({txt:txt,l:l,top:top,w:w,h:h,col:s.color,bold:(s.fontWeight==='bold'||+s.fontWeight>=600?1:0),link:(a?1:0),bg:bg,inp:0,href:href});\
            if(items.length>450)break;\
        }\
        var inputs=document.querySelectorAll('input:not([type=hidden]):not([type=checkbox]):not([type=radio]):not([type=submit]):not([type=button]),textarea,[role=searchbox],[role=combobox]');\
        for(var j=0;j<inputs.length;j++){\
            var el=inputs[j];\
            var es=getComputedStyle(el);\
            if(es.display==='none'||es.visibility==='hidden')continue;\
            var erc=el.getBoundingClientRect();\
            if(!erc||erc.width<24||erc.height<10||erc.bottom<0||erc.top>vh)continue;\
            var ity=~~(erc.top+Math.max(0,(erc.height-16)/2));\
            var matched=false;\
            for(var k=0;k<items.length;k++){\
                var it=items[k];\
                if(it.l>=erc.left-6&&it.l<=erc.right&&it.top>=erc.top-6&&it.top<=erc.bottom){\
                    it.top=ity;\
                    it.inp=1;\
                    if(el.value&&el.value.trim())it.txt=el.value.trim();\
                    matched=true;\
                }\
            }\
            if(!matched){\
                var val=(el.value||el.placeholder||el.getAttribute('aria-label')||el.getAttribute('title')||'Search or type...').split(T).join(' ').split(N).join(' ').trim();\
                if(!val)val='Search or type...';\
                items.push({txt:val,l:~~(erc.left+8),top:ity,w:~~Math.max(16,erc.width-16),h:18,col:es.color,bold:0,link:0,bg:getBg(el),inp:1,href:''});\
            }\
        }\
        for(var m=0;m<items.length;m++){\
            var x=items[m];\
            r.push(x.txt+T+x.l+T+x.top+T+x.w+T+x.h+T+x.col+T+x.bold+T+x.link+T+x.bg+T+x.inp+T+x.href);\
        }\
        return r.join(N);\
    })()";
    let cmd = format!(
        r#"{{"id":{},"method":"Runtime.evaluate","params":{{"expression":"{}","returnByValue":true}}}}"#,
        TEXT_EXTRACT_ID,
        json_escape(js)
    );
    let _ = ws.send_text(&cmd);
}

fn configure_cdp_adblock(ws: &mut WsConnection, seq: &mut u64, enabled: bool) {
    *seq += 1;
    let _ = ws.send_text(&format!(r#"{{"id":{},"method":"Network.enable"}}"#, *seq));
    *seq += 1;
    if enabled {
        let blocked_cmd = format!(
            r#"{{"id":{},"method":"Network.setBlockedURLs","params":{{"urls":["*://*.doubleclick.net/*","*://*.googlesyndication.com/*","*://*.googleadservices.com/*","*://*.adservice.google.*/*","*://*.moatads.com/*","*://*.taboola.com/*","*://*.outbrain.com/*","*://*.criteo.com/*","*://*.amazon-adsystem.com/*","*://*.adsafeprotected.com/*","*://*.scorecardresearch.com/*","*://*.quantserve.com/*","*://*.adnxs.com/*","*://*.rubiconproject.com/*","*://*.pubmatic.com/*","*://*.openx.net/*","*://*.popads.net/*","*://*.google-analytics.com/*","*://*.googletagmanager.com/*"]}}}}"#,
            *seq
        );
        let _ = ws.send_text(&blocked_cmd);
    } else {
        let unblock_cmd = format!(
            r#"{{"id":{},"method":"Network.setBlockedURLs","params":{{"urls":[]}}}}"#,
            *seq
        );
        let _ = ws.send_text(&unblock_cmd);
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_cdp_worker(
    port: u16,
    initial_url: String,
    cols: u16,
    rows: u16,
    shared_buf: Arc<Mutex<PixelBuffer>>,
    shared_overlays: Arc<Mutex<Vec<TextOverlaySpan>>>,
    shared_viewport: Arc<Mutex<(u32, u32)>>,
    shared_scroll_pct: Arc<AtomicUsize>,
    shared_live_url: Arc<Mutex<String>>,
    frame_counter: Arc<AtomicUsize>,
    shared_error: Arc<Mutex<Option<String>>>,
    is_loading: Arc<AtomicBool>,
    command_rx: Receiver<CdpWorkerCommand>,
) {
    std::thread::spawn(move || {
        let mut ws_path = String::new();
        for _ in 0..50 {
            if let Ok(mut stream) = TcpStream::connect(("127.0.0.1", port)) {
                stream.set_read_timeout(Some(Duration::from_millis(500))).ok();
                stream.set_write_timeout(Some(Duration::from_millis(500))).ok();
                let _ = stream.write_all(b"GET /json/list HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
                let mut resp = Vec::new();
                let mut buf = [0u8; 8192];
                if let Ok(n) = stream.read(&mut buf) {
                    resp.extend_from_slice(&buf[..n]);
                    let resp_str = String::from_utf8_lossy(&resp);

                    let mut page_path = String::new();
                    for block in resp_str.split('{') {
                        if block.contains(r#""type": "page""#) || block.contains(r#""type":"page""#) {
                            if let Some(pos) = block.find("/devtools/page/") {
                                let rest = &block[pos..];
                                if let Some(end) = rest.find('"') {
                                    page_path = rest[..end].to_string();
                                    break;
                                }
                            }
                        }
                    }

                    if !page_path.is_empty() {
                        ws_path = page_path;
                        break;
                    } else if let Some(pos) = resp_str.find("/devtools/page/") {
                        let rest = &resp_str[pos..];
                        if let Some(end) = rest.find('"') {
                            ws_path = rest[..end].to_string();
                            break;
                        }
                    }
                }
            }
            std::thread::sleep(Duration::from_millis(150));
        }

        if ws_path.is_empty() {
            *shared_error.lock().unwrap() = Some("Chromium started but DevTools page endpoint was not found".to_string());
            return;
        }

        let mut ws = match WsConnection::connect(port, &ws_path) {
            Ok(w) => w,
            Err(e) => {
                *shared_error.lock().unwrap() = Some(format!("CDP WebSocket connection failed: {}", e));
                return;
            }
        };

        let mut seq: u64 = 10;

        seq += 1;
        let _ = ws.send_text(&format!(r#"{{"id":{},"method":"Page.enable"}}"#, seq));

        let mut curr_cols = cols;
        let mut curr_rows = rows;
        let mut curr_zoom: f32 = 1.0;
        let mut curr_hide_text: bool = true;
        let mut curr_adblock: bool = true;

        configure_cdp_adblock(&mut ws, &mut seq, curr_adblock);

        seq += 1;
        let preload_js = build_page_style_js(curr_hide_text, curr_adblock);
        let preload_src = format!(
            "document.addEventListener('DOMContentLoaded',function(){{{}}});{}",
            preload_js, preload_js
        );
        let add_script_cmd = format!(
            r#"{{"id":{},"method":"Page.addScriptToEvaluateOnNewDocument","params":{{"source":"{}"}}}}"#,
            seq,
            json_escape(&preload_src)
        );
        let _ = ws.send_text(&add_script_cmd);

        seq += 1;
        let ua_cmd = format!(
            r#"{{"id":{},"method":"Emulation.setUserAgentOverride","params":{{"userAgent":"Mozilla/5.0 (Linux; Android 14; Mobile) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36"}}}}"#,
            seq
        );
        let _ = ws.send_text(&ua_cmd);

        let (mut width_px, mut height_px) = compute_viewport_size(curr_cols, curr_rows, curr_zoom);
        *shared_viewport.lock().unwrap() = (width_px, height_px);
        seq += 1;
        let emu_cmd = format!(
            r#"{{"id":{},"method":"Emulation.setDeviceMetricsOverride","params":{{"width":{},"height":{},"deviceScaleFactor":1.0,"mobile":true,"fitWindow":false}}}}"#,
            seq, width_px, height_px
        );
        let _ = ws.send_text(&emu_cmd);

        seq += 1;
        let nav_cmd = format!(
            r#"{{"id":{},"method":"Page.navigate","params":{{"url":"{}"}}}}"#,
            seq,
            json_escape(&initial_url)
        );
        let _ = ws.send_text(&nav_cmd);

        seq += 1;
        let sc_cmd = format!(
            r#"{{"id":{},"method":"Page.startScreencast","params":{{"format":"jpeg","quality":85,"maxWidth":{},"maxHeight":{},"everyNthFrame":1}}}}"#,
            seq, width_px, height_px
        );
        let _ = ws.send_text(&sc_cmd);

        let mut last_text_extract = Instant::now() - Duration::from_secs(1);
        let mut pending_extract = true;
        let mut active_until = Instant::now() + Duration::from_secs(3);

        loop {
            while let Ok(cmd) = command_rx.try_recv() {
                match cmd {
                    CdpWorkerCommand::Navigate(new_url) => {
                        is_loading.store(true, Ordering::Relaxed);
                        frame_counter.store(0, Ordering::Relaxed);
                        shared_overlays.lock().unwrap().clear();
                        *shared_live_url.lock().unwrap() = new_url.clone();
                        let (w, h) = compute_viewport_size(curr_cols, curr_rows, curr_zoom);
                        width_px = w;
                        height_px = h;
                        *shared_viewport.lock().unwrap() = (width_px, height_px);
                        seq += 1;
                        let emu = format!(
                            r#"{{"id":{},"method":"Emulation.setDeviceMetricsOverride","params":{{"width":{},"height":{},"deviceScaleFactor":1.0,"mobile":true,"fitWindow":false}}}}"#,
                            seq, width_px, height_px
                        );
                        let _ = ws.send_text(&emu);
                        seq += 1;
                        let msg = format!(
                            r#"{{"id":{},"method":"Page.navigate","params":{{"url":"{}"}}}}"#,
                            seq,
                            json_escape(&new_url)
                        );
                        let _ = ws.send_text(&msg);
                        pending_extract = true;
                        active_until = Instant::now() + Duration::from_secs(3);
                    }
                    CdpWorkerCommand::SetHideText(hide) => {
                        curr_hide_text = hide;
                        seq += 1;
                        let js = build_page_style_js(curr_hide_text, curr_adblock);
                        let eval_cmd = format!(
                            r#"{{"id":{},"method":"Runtime.evaluate","params":{{"expression":"{}"}}}}"#,
                            seq,
                            json_escape(&js)
                        );
                        let _ = ws.send_text(&eval_cmd);
                        if hide {
                            request_text_extraction(&mut ws, &mut seq);
                            last_text_extract = Instant::now();
                        }
                        active_until = Instant::now() + Duration::from_millis(800);
                    }
                    CdpWorkerCommand::SetAdBlock(enabled) => {
                        curr_adblock = enabled;
                        configure_cdp_adblock(&mut ws, &mut seq, curr_adblock);
                        seq += 1;
                        let js = build_page_style_js(curr_hide_text, curr_adblock);
                        let eval_cmd = format!(
                            r#"{{"id":{},"method":"Runtime.evaluate","params":{{"expression":"{}"}}}}"#,
                            seq,
                            json_escape(&js)
                        );
                        let _ = ws.send_text(&eval_cmd);
                        pending_extract = true;
                    }
                    CdpWorkerCommand::Resize { cols: new_cols, rows: new_rows } => {
                        curr_cols = new_cols;
                        curr_rows = new_rows;
                        let (w, h) = compute_viewport_size(curr_cols, curr_rows, curr_zoom);
                        width_px = w;
                        height_px = h;
                        *shared_viewport.lock().unwrap() = (width_px, height_px);
                        seq += 1;
                        let emu = format!(
                            r#"{{"id":{},"method":"Emulation.setDeviceMetricsOverride","params":{{"width":{},"height":{},"deviceScaleFactor":1.0,"mobile":true,"fitWindow":false}}}}"#,
                            seq, width_px, height_px
                        );
                        let _ = ws.send_text(&emu);
                        seq += 1;
                        let sc = format!(
                            r#"{{"id":{},"method":"Page.startScreencast","params":{{"format":"jpeg","quality":85,"maxWidth":{},"maxHeight":{},"everyNthFrame":1}}}}"#,
                            seq, width_px, height_px
                        );
                        let _ = ws.send_text(&sc);
                        pending_extract = true;
                        active_until = Instant::now() + Duration::from_millis(1000);
                    }
                    CdpWorkerCommand::Scroll(delta) => {
                        let center_x = width_px / 2;
                        let center_y = height_px / 2;
                        seq += 1;
                        let scroll_msg = format!(
                            r#"{{"id":{},"method":"Input.dispatchMouseEvent","params":{{"type":"mouseWheel","x":{},"y":{},"deltaX":0,"deltaY":{}}}}}"#,
                            seq, center_x, center_y, delta * 90
                        );
                        let _ = ws.send_text(&scroll_msg);
                        pending_extract = true;
                        active_until = Instant::now() + Duration::from_millis(900);
                    }
                    CdpWorkerCommand::Click(col, row) => {
                        let px_x = (((col as f32 + 0.5) * width_px as f32) / curr_cols.max(1) as f32) as u32;
                        let px_y = ((((row.saturating_sub(1) as f32) + 0.5) * height_px as f32)
                            / (curr_rows.saturating_sub(1)).max(1) as f32) as u32;
                        seq += 1;
                        let press = format!(
                            r#"{{"id":{},"method":"Input.dispatchMouseEvent","params":{{"type":"mousePressed","x":{},"y":{},"button":"left","clickCount":1}}}}"#,
                            seq, px_x, px_y
                        );
                        let _ = ws.send_text(&press);
                        seq += 1;
                        let release = format!(
                            r#"{{"id":{},"method":"Input.dispatchMouseEvent","params":{{"type":"mouseReleased","x":{},"y":{},"button":"left","clickCount":1}}}}"#,
                            seq, px_x, px_y
                        );
                        let _ = ws.send_text(&release);
                        pending_extract = true;
                        active_until = Instant::now() + Duration::from_secs(2);
                    }
                    CdpWorkerCommand::SetZoom(new_zoom) => {
                        curr_zoom = new_zoom.clamp(0.5, 3.0);
                        let (w, h) = compute_viewport_size(curr_cols, curr_rows, curr_zoom);
                        width_px = w;
                        height_px = h;
                        *shared_viewport.lock().unwrap() = (width_px, height_px);
                        seq += 1;
                        let emu = format!(
                            r#"{{"id":{},"method":"Emulation.setDeviceMetricsOverride","params":{{"width":{},"height":{},"deviceScaleFactor":1.0,"mobile":true,"fitWindow":false}}}}"#,
                            seq, width_px, height_px
                        );
                        let _ = ws.send_text(&emu);
                        seq += 1;
                        let sc = format!(
                            r#"{{"id":{},"method":"Page.startScreencast","params":{{"format":"jpeg","quality":85,"maxWidth":{},"maxHeight":{},"everyNthFrame":1}}}}"#,
                            seq, width_px, height_px
                        );
                        let _ = ws.send_text(&sc);
                        pending_extract = true;
                        active_until = Instant::now() + Duration::from_millis(1000);
                    }
                    CdpWorkerCommand::HistoryBack => {
                        seq += 1;
                        let eval_cmd = format!(
                            r#"{{"id":{},"method":"Runtime.evaluate","params":{{"expression":"history.back()"}}}}"#,
                            seq
                        );
                        let _ = ws.send_text(&eval_cmd);
                        pending_extract = true;
                        active_until = Instant::now() + Duration::from_secs(2);
                    }
                    CdpWorkerCommand::HistoryForward => {
                        seq += 1;
                        let eval_cmd = format!(
                            r#"{{"id":{},"method":"Runtime.evaluate","params":{{"expression":"history.forward()"}}}}"#,
                            seq
                        );
                        let _ = ws.send_text(&eval_cmd);
                        pending_extract = true;
                        active_until = Instant::now() + Duration::from_secs(2);
                    }
                    CdpWorkerCommand::Stop => {
                        let _ = ws.send_text(r#"{"id":999,"method":"Page.stopScreencast"}"#);
                        return;
                    }
                }
            }

            while let Some(msg) = ws.poll_message() {
                if msg.contains(r#""Page.loadEventFired""#) || msg.contains(r#""Page.frameNavigated""#) {
                    if msg.contains(r#""Page.loadEventFired""#) {
                        is_loading.store(false, Ordering::Relaxed);
                    }
                    seq += 1;
                    let setup_js = build_page_style_js(curr_hide_text, curr_adblock);
                    let eval_cmd = format!(
                        r#"{{"id":{},"method":"Runtime.evaluate","params":{{"expression":"{}"}}}}"#,
                        seq,
                        json_escape(&setup_js)
                    );
                    let _ = ws.send_text(&eval_cmd);
                    pending_extract = true;
                    active_until = Instant::now() + Duration::from_secs(2);
                }

                // Handle text extraction response
                if msg.contains(r#""id":55555"#) || msg.contains(r#""id": 55555"#) {
                    if let Some(pos) = msg.find(r#""value":""#) {
                        let rest = &msg[pos + 9..];
                        if let Some(q_end) = find_unescaped_quote(rest) {
                            let raw_val = &rest[..q_end];
                            let unescaped = json_unescape(raw_val);
                            let mut spans = Vec::new();
                            for line in unescaped.lines() {
                                let parts: Vec<&str> = line.split('\t').collect();
                                if parts.first() == Some(&"META") && parts.len() >= 5 {
                                    let vw = parts[1].parse::<u32>().unwrap_or(width_px).max(100);
                                    let vh = parts[2].parse::<u32>().unwrap_or(height_px).max(100);
                                    let pct = parts[3].parse::<usize>().unwrap_or(0);
                                    let url_str = parts[4].trim();
                                    *shared_viewport.lock().unwrap() = (vw, vh);
                                    shared_scroll_pct.store(pct, Ordering::Relaxed);
                                    if !url_str.is_empty() && url_str != "about:blank" {
                                        *shared_live_url.lock().unwrap() = url_str.to_string();
                                    }
                                    continue;
                                }
                                if parts.len() >= 8 {
                                    let text = parts[0].to_string();
                                    let px_x = parts[1].parse::<i32>().unwrap_or(0);
                                    let px_y = parts[2].parse::<i32>().unwrap_or(0);
                                    let px_w = parts[3].parse::<i32>().unwrap_or(0);
                                    let px_h = parts[4].parse::<i32>().unwrap_or(0);
                                    let fg = parse_color_from_str(parts[5]).unwrap_or(Color::BLACK);
                                    let bold = parts[6] == "1";
                                    let is_link = parts[7] == "1";
                                    let bg = if parts.len() >= 9 {
                                        parse_color_from_str(parts[8])
                                    } else {
                                        None
                                    };
                                    let is_input = parts.get(9).copied() == Some("1");
                                    let link_url = parts
                                        .get(10)
                                        .map(|s| s.trim())
                                        .filter(|s| !s.is_empty())
                                        .map(|s| s.to_string());
                                    spans.push(TextOverlaySpan {
                                        text,
                                        px_x,
                                        px_y,
                                        px_w,
                                        px_h,
                                        fg,
                                        bg,
                                        bold,
                                        is_link,
                                        is_input,
                                        link_url,
                                    });
                                }
                            }
                            *shared_overlays.lock().unwrap() = spans;
                        }
                    }
                }

                if msg.contains(r#""Page.screencastFrame""#) {
                    let session_id = if let Some(pos) = msg.find(r#""sessionId""#) {
                        let rest = &msg[pos + 11..];
                        let digits: String = rest
                            .chars()
                            .skip_while(|c| *c != ':' && !c.is_ascii_digit())
                            .skip_while(|c| *c == ':' || c.is_whitespace())
                            .take_while(|c| c.is_ascii_digit())
                            .collect();
                        digits.parse::<u64>().unwrap_or(0)
                    } else {
                        0
                    };

                    if let Some(pos) = msg.find(r#""data""#) {
                        let rest = &msg[pos + 6..];
                        if let Some(q1) = rest.find('"') {
                            let after = &rest[q1 + 1..];
                            if let Some(q2) = after.find('"') {
                                let b64 = &after[..q2];
                                let jpeg_bytes = base64_decode(b64);
                                if let Ok(img) = image::load_from_memory(&jpeg_bytes) {
                                    let new_buf = PixelBuffer::from_image(&img);
                                    *shared_buf.lock().unwrap() = new_buf;
                                    let c = frame_counter.fetch_add(1, Ordering::Relaxed);
                                    if c >= 1 {
                                        is_loading.store(false, Ordering::Relaxed);
                                    }
                                    pending_extract = true;
                                }
                            }
                        }
                    }

                    // Always acknowledge frame immediately
                    seq += 1;
                    let ack = format!(
                        r#"{{"id":{},"method":"Page.screencastFrameAck","params":{{"sessionId":{}}}}}"#,
                        seq, session_id
                    );
                    let _ = ws.send_text(&ack);
                }
            }

            // Throttled real-time DOM text & coordinate extraction during active updates
            let now = Instant::now();
            if (pending_extract || now < active_until)
                && last_text_extract.elapsed() >= Duration::from_millis(90)
            {
                let setup_js = build_page_style_js(curr_hide_text, curr_adblock);
                seq += 1;
                let eval_cmd = format!(
                    r#"{{"id":{},"method":"Runtime.evaluate","params":{{"expression":"{}"}}}}"#,
                    seq,
                    json_escape(&setup_js)
                );
                let _ = ws.send_text(&eval_cmd);
                request_text_extraction(&mut ws, &mut seq);
                last_text_extract = now;
                pending_extract = false;
            }
        }
    });
}

pub struct CdpEngine {
    pub process: Option<Child>,
    pub port: u16,
    pub is_connected: bool,
    pub current_url: String,
    pub error_message: Arc<Mutex<Option<String>>>,
    pub chromium_bin: Option<String>,
    pub frame_buffer: Arc<Mutex<PixelBuffer>>,
    pub text_overlays: Arc<Mutex<Vec<TextOverlaySpan>>>,
    pub viewport_dims: Arc<Mutex<(u32, u32)>>,
    pub scroll_pct: Arc<AtomicUsize>,
    pub live_url: Arc<Mutex<String>>,
    pub frame_counter: Arc<AtomicUsize>,
    pub is_loading: Arc<AtomicBool>,
    pub zoom: f32,
    pub adblock_enabled: bool,
    pub upscale_mode: UpscaleMode,
    pub dithering_enabled: bool,
    command_tx: Option<Sender<CdpWorkerCommand>>,
    cols: u16,
    rows: u16,
}

impl CdpEngine {
    pub fn new(port: u16) -> Self {
        let chromium_bin = Self::find_chromium_binary();
        Self {
            process: None,
            port,
            is_connected: false,
            current_url: "about:blank".to_string(),
            error_message: Arc::new(Mutex::new(None)),
            frame_buffer: Arc::new(Mutex::new(PixelBuffer::new(80, 48, Color::new(246, 246, 239)))),
            text_overlays: Arc::new(Mutex::new(Vec::new())),
            viewport_dims: Arc::new(Mutex::new((360, 640))),
            scroll_pct: Arc::new(AtomicUsize::new(0)),
            live_url: Arc::new(Mutex::new(String::new())),
            frame_counter: Arc::new(AtomicUsize::new(0)),
            is_loading: Arc::new(AtomicBool::new(true)),
            chromium_bin,
            command_tx: None,
            cols: 80,
            rows: 24,
            zoom: 1.0,
            adblock_enabled: true,
            upscale_mode: UpscaleMode::FsrOnly,
            dithering_enabled: true,
        }
    }

    pub fn get_text_overlays(&self) -> Vec<TextOverlaySpan> {
        self.text_overlays.lock().unwrap().clone()
    }

    pub fn get_viewport_dims(&self) -> (u32, u32) {
        *self.viewport_dims.lock().unwrap()
    }

    pub fn get_scroll_pct(&self) -> usize {
        self.scroll_pct.load(Ordering::Relaxed)
    }

    pub fn get_live_url(&self) -> String {
        self.live_url.lock().unwrap().clone()
    }

    pub fn hit_test_overlay(&self, click_x: u16, click_y: u16, cols: u16, rows: u16) -> (Option<String>, bool) {
        let overlays = self.get_text_overlays();
        let (vw, vh) = self.get_viewport_dims();
        let pw = vw.max(1) as f32;
        let ph = vh.max(1) as f32;
        let view_start_y = 1u16;
        let view_rows = (rows.saturating_sub(view_start_y)).max(1) as f32;

        for span in overlays.iter().rev() {
            let start_col = ((span.px_x.max(0) as f32 * cols as f32) / pw).round() as u16;
            let span_cols = ((span.px_w.max(8) as f32 * cols as f32) / pw).round().max(2.0) as u16;
            let term_row = view_start_y + ((span.px_y.max(0) as f32 * view_rows) / ph).round() as u16;
            let box_rows = ((span.px_h.max(14) as f32 * view_rows) / ph).ceil().max(1.0) as u16;

            if click_y >= term_row
                && click_y < term_row.saturating_add(box_rows)
                && click_x >= start_col
                && click_x <= start_col.saturating_add(span_cols)
            {
                if span.is_input {
                    return (None, true);
                }
                if let Some(ref u) = span.link_url {
                    return (Some(u.clone()), false);
                }
            }
        }
        (None, false)
    }

    pub fn set_hide_text(&mut self, hide: bool) {
        if let Some(ref tx) = self.command_tx {
            let _ = tx.send(CdpWorkerCommand::SetHideText(hide));
        }
    }

    pub fn set_adblock(&mut self, enabled: bool) {
        self.adblock_enabled = enabled;
        if let Some(ref tx) = self.command_tx {
            let _ = tx.send(CdpWorkerCommand::SetAdBlock(enabled));
        }
    }

    pub fn history_back(&mut self) {
        if let Some(ref tx) = self.command_tx {
            let _ = tx.send(CdpWorkerCommand::HistoryBack);
        }
    }

    pub fn history_forward(&mut self) {
        if let Some(ref tx) = self.command_tx {
            let _ = tx.send(CdpWorkerCommand::HistoryForward);
        }
    }

    pub fn zoom_in(&mut self) {
        self.zoom = (self.zoom + 0.25).min(2.5);
        if let Some(ref tx) = self.command_tx {
            let _ = tx.send(CdpWorkerCommand::SetZoom(self.zoom));
        }
    }

    pub fn zoom_out(&mut self) {
        self.zoom = (self.zoom - 0.25).max(0.5);
        if let Some(ref tx) = self.command_tx {
            let _ = tx.send(CdpWorkerCommand::SetZoom(self.zoom));
        }
    }

    pub fn zoom_reset(&mut self) {
        self.zoom = 1.0;
        if let Some(ref tx) = self.command_tx {
            let _ = tx.send(CdpWorkerCommand::SetZoom(self.zoom));
        }
    }

    pub fn get_error(&self) -> Option<String> {
        self.error_message.lock().unwrap().clone()
    }

    pub fn get_frame_count(&self) -> usize {
        self.frame_counter.load(Ordering::Relaxed)
    }

    pub fn is_loading(&self) -> bool {
        self.is_loading.load(Ordering::Relaxed)
    }

    /// Automatically searches for headless Chromium across Termux ($PREFIX), Debian/Linux (/usr/bin), macOS, Windows, and $PATH
    pub fn find_chromium_binary() -> Option<String> {
        if let Ok(prefix) = std::env::var("PREFIX") {
            let termux_candidates = [
                format!("{}/bin/chromium", prefix),
                format!("{}/bin/chromium-browser", prefix),
            ];
            for p in &termux_candidates {
                if std::path::Path::new(p).exists() {
                    return Some(p.clone());
                }
            }
        }

        let standard_candidates = [
            "/usr/lib/chromium/chromium",
            "/data/data/com.termux/files/usr/var/lib/proot-distro/containers/debian/rootfs/usr/lib/chromium/chromium",
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
            "/usr/local/bin/chromium",
            "/usr/local/bin/chromium-browser",
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/snap/bin/chromium",
            "/var/lib/flatpak/exports/bin/org.chromium.Chromium",
            "/data/data/com.termux/files/usr/var/lib/proot-distro/containers/debian/rootfs/usr/bin/chromium",
            "/data/data/com.termux/files/usr/var/lib/proot-distro/installed-rootfs/debian/usr/bin/chromium",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
            "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
            "C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe",
        ];

        for p in &standard_candidates {
            if std::path::Path::new(p).exists() {
                return Some(p.to_string());
            }
        }

        let binary_names = [
            "chromium",
            "chromium-browser",
            "google-chrome",
            "google-chrome-stable",
            "chrome",
            "brave-browser",
            "microsoft-edge-stable",
            "msedge",
        ];

        if let Ok(path_var) = std::env::var("PATH") {
            for dir in std::env::split_paths(&path_var) {
                for name in &binary_names {
                    let full_path = dir.join(name);
                    if full_path.is_file() {
                        return Some(full_path.to_string_lossy().to_string());
                    }
                }
            }
        }

        for name in &binary_names {
            if let Ok(res) = Command::new(name).arg("--version").output() {
                if res.status.success() {
                    return Some(name.to_string());
                }
            }
        }

        None
    }

    pub fn ensure_browser_running(&mut self, window_w: u16, window_h: u16) -> Result<(), String> {
        if self.process.is_some() {
            return Ok(());
        }

        if let Ok(mut stream) = TcpStream::connect(("127.0.0.1", self.port)) {
            stream.set_read_timeout(Some(Duration::from_millis(300))).ok();
            stream.set_write_timeout(Some(Duration::from_millis(300))).ok();
            let _ = stream.write_all(b"GET /json/version HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
            let mut resp = [0u8; 64];
            if let Ok(n) = stream.read(&mut resp) {
                if n > 0 && String::from_utf8_lossy(&resp).contains("200") {
                    self.is_connected = true;
                    return Ok(());
                }
            }
        }

        let bin = match &self.chromium_bin {
            Some(b) => b.clone(),
            None => {
                let err = "Chromium binary not found on system.\n\nInstall via: apt install -y chromium\nOr press [m] for Native Mode.".to_string();
                *self.error_message.lock().unwrap() = Some(err.clone());
                return Err(err);
            }
        };

        let win_arg = format!("--window-size={},{}", window_w, window_h);

        let child = Command::new(&bin)
            .arg("--headless")
            .arg(format!("--remote-debugging-port={}", self.port))
            .arg("--no-sandbox")
            .arg("--no-zygote")
            .arg("--disable-gpu")
            .arg("--disable-software-rasterizer")
            .arg("--disable-dev-shm-usage")
            .arg("--disable-logging")
            .arg("--log-level=3")
            .arg("--silent-debugger-extension-api")
            .arg("--disable-breakpad")
            .arg("--disable-component-update")
            .arg("--disable-background-networking")
            .arg("--disable-popup-blocking")
            .arg("--disable-features=DevicePosture,MediaStream,AudioServiceOutOfProcess,Translate,BackForwardCache")
            .arg("--mute-audio")
            .arg("--no-first-run")
            .arg("--no-default-browser-check")
            .arg("--hide-scrollbars")
            .arg(&win_arg)
            .arg("about:blank")
            .env("CHROME_LOG_FILE", "/dev/null")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();

        match child {
            Ok(c) => {
                self.process = Some(c);
                *self.error_message.lock().unwrap() = None;

                let mut ready = false;
                for _ in 0..40 {
                    std::thread::sleep(Duration::from_millis(150));
                    if let Ok(mut stream) = TcpStream::connect(("127.0.0.1", self.port)) {
                        stream.set_read_timeout(Some(Duration::from_millis(500))).ok();
                        stream.set_write_timeout(Some(Duration::from_millis(500))).ok();
                        let _ = stream.write_all(b"GET /json/version HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
                        let mut resp = [0u8; 64];
                        if let Ok(n) = stream.read(&mut resp) {
                            if n > 0 && String::from_utf8_lossy(&resp).contains("200") {
                                ready = true;
                                break;
                            }
                        }
                    }
                }

                if ready {
                    self.is_connected = true;
                    Ok(())
                } else {
                    let err = format!(
                        "Chromium started (PID: {:?}) but failed to bind port {}",
                        self.process.as_ref().map(|p| p.id()),
                        self.port
                    );
                    *self.error_message.lock().unwrap() = Some(err.clone());
                    Err(err)
                }
            }
            Err(e) => {
                let err = format!("Failed to spawn {}: {}", bin, e);
                *self.error_message.lock().unwrap() = Some(err.clone());
                Err(err)
            }
        }
    }

    pub fn navigate(&mut self, url: &str, cols: u16, rows: u16) {
        self.current_url = url.to_string();
        *self.live_url.lock().unwrap() = url.to_string();
        self.cols = cols;
        self.rows = rows;
        self.is_loading.store(true, Ordering::Relaxed);

        let (w_px, h_px) = compute_viewport_size(cols, rows, self.zoom);

        if let Err(e) = self.ensure_browser_running(w_px as u16, h_px as u16) {
            *self.error_message.lock().unwrap() = Some(e);
            return;
        }

        if let Some(ref tx) = self.command_tx {
            let _ = tx.send(CdpWorkerCommand::Navigate(url.to_string()));
        } else {
            let (tx, rx) = channel();
            self.command_tx = Some(tx);
            spawn_cdp_worker(
                self.port,
                url.to_string(),
                cols,
                rows,
                Arc::clone(&self.frame_buffer),
                Arc::clone(&self.text_overlays),
                Arc::clone(&self.viewport_dims),
                Arc::clone(&self.scroll_pct),
                Arc::clone(&self.live_url),
                Arc::clone(&self.frame_counter),
                Arc::clone(&self.error_message),
                Arc::clone(&self.is_loading),
                rx,
            );
        }
    }

    pub fn scroll(&mut self, delta: i32) {
        if let Some(ref tx) = self.command_tx {
            let _ = tx.send(CdpWorkerCommand::Scroll(delta));
        }
    }

    pub fn click(&mut self, col: u16, row: u16) {
        if let Some(ref tx) = self.command_tx {
            let _ = tx.send(CdpWorkerCommand::Click(col, row));
        }
    }

    pub fn resize(&mut self, cols: u16, rows: u16) {
        self.cols = cols;
        self.rows = rows;
        if let Some(ref tx) = self.command_tx {
            let _ = tx.send(CdpWorkerCommand::Resize { cols, rows });
        }
    }

    pub fn get_frame(&self) -> PixelBuffer {
        self.frame_buffer.lock().unwrap().clone()
    }

    pub fn stop(&mut self) {
        if let Some(tx) = self.command_tx.take() {
            let _ = tx.send(CdpWorkerCommand::Stop);
        }
        if let Some(mut child) = self.process.take() {
            let pid = child.id();
            let _ = child.kill();
            let _ = child.wait();
            #[cfg(unix)]
            {
                let _ = Command::new("pkill")
                    .arg("-9")
                    .arg("-P")
                    .arg(pid.to_string())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .output();
            }
        }
    }
}

impl Drop for CdpEngine {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_decoding() {
        assert_eq!(decode_html_entities("473&nbsp;comments"), "473 comments");
        assert_eq!(decode_html_entities("Go&#x27;s"), "Go's");
        assert_eq!(decode_html_entities("Go&#39;s"), "Go's");
        assert_eq!(decode_html_entities("&quot;hello&quot;"), "\"hello\"");
        assert_eq!(decode_html_entities("foo &amp; bar"), "foo & bar");
        assert_eq!(decode_html_entities("&lt;div&gt;"), "<div>");
        assert_eq!(decode_html_entities("a &mdash; b"), "a — b");
    }

    #[test]
    fn test_color_and_bg_parsing() {
        assert_eq!(parse_color_from_str("#ff6600"), Some(Color::new(255, 102, 0)));
        assert_eq!(parse_color_from_str("#f6f6ef"), Some(Color::new(246, 246, 239)));
        assert_eq!(parse_color_from_str("white"), Some(Color::WHITE));
        assert_eq!(parse_color_from_str("black"), Some(Color::BLACK));
    }

    #[test]
    fn test_script_with_less_than_does_not_swallow_body() {
        let mut engine = NativeEngine::new();
        engine.raw_html = r##"
        <html>
        <head><script>for (var i = 0; i < 10; i++) { console.log(i); }</script></head>
        <body><p>Visible Body Content</p></body>
        </html>
        "##.to_string();
        engine.parse_and_layout(80);
        let all_text: String = engine
            .lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.text.as_str()))
            .collect();
        assert!(all_text.contains("Visible Body Content"));
        assert!(!all_text.contains("console.log"));
    }

    #[test]
    fn test_table_horizontal_layout() {
        let mut engine = NativeEngine::new();
        engine.raw_html = r##"
        <table bgcolor="#f6f6ef">
            <tr bgcolor="#ff6600">
                <td><b>Hacker News</b></td>
                <td>new | past | comments</td>
            </tr>
            <tr>
                <td align="right">1.</td>
                <td><a href="https://example.com">AI is forcing us to pay debt</a> (ieee.org)</td>
            </tr>
            <tr>
                <td colspan="2"></td>
                <td class="subtext">771 points by user 7 hours ago | 473&nbsp;comments</td>
            </tr>
        </table>
        "##.to_string();

        engine.parse_and_layout(100);

        assert_eq!(engine.lines[0].row_bg, Some(Color::new(255, 102, 0)));
        let header_text: String = engine.lines[0].spans.iter().map(|s| s.text.as_str()).collect();
        assert!(header_text.contains("Hacker News"));
        assert!(header_text.contains("new | past | comments"));

        let story_text: String = engine.lines[1].spans.iter().map(|s| s.text.as_str()).collect();
        assert!(story_text.contains("1."));
        assert!(story_text.contains("AI is forcing us to pay debt"));

        let subtext: String = engine.lines[2].spans.iter().map(|s| s.text.as_str()).collect();
        assert!(subtext.contains("473 comments"));
        assert!(!subtext.contains("&nbsp;"));
    }

    #[test]
    fn test_render_mode_cycle() {
        assert_eq!(RenderMode::Native.next(true), RenderMode::CdpHybrid);
        assert_eq!(RenderMode::CdpHybrid.next(true), RenderMode::CdpRaw);
        assert_eq!(RenderMode::CdpRaw.next(true), RenderMode::Native);

        assert_eq!(RenderMode::Native.next(false), RenderMode::Native);
        assert_eq!(RenderMode::CdpHybrid.next(false), RenderMode::Native);
        assert_eq!(RenderMode::CdpRaw.next(false), RenderMode::Native);

        assert!(!RenderMode::Native.is_cdp());
        assert!(RenderMode::CdpHybrid.is_cdp());
        assert!(RenderMode::CdpRaw.is_cdp());
    }

    #[test]
    fn test_rgba_color_parsing() {
        assert_eq!(parse_color_from_str("rgba(255, 102, 0, 1.0)"), Some(Color::new(255, 102, 0)));
        assert_eq!(parse_color_from_str("rgba(0, 0, 0, 0)"), None);
        assert_eq!(parse_color_from_str("rgb(10, 20, 30)"), Some(Color::new(10, 20, 30)));
    }

    #[test]
    fn test_json_unescape() {
        assert_eq!(json_unescape("hello\\nworld\\ttab"), "hello\nworld\ttab");
        assert_eq!(json_unescape("say \\\"hello\\\""), "say \"hello\"");
        assert_eq!(json_unescape("unicode \\u0041"), "unicode A");
    }

    #[test]
    fn test_normalized_coordinate_mapping() {
        let span = TextOverlaySpan {
            text: "Google offered in:".to_string(),
            px_x: 60,
            px_y: 260,
            px_w: 120,
            px_h: 18,
            fg: Color::BLACK,
            bg: None,
            bold: false,
            is_link: false,
            is_input: false,
            link_url: None,
        };

        let cols = 45u16;
        let rows = 42u16;
        let view_start_y = 1u16;
        let view_max_y = rows;
        let pw = 360.0f32;
        let ph = 640.0f32;
        let view_rows = (view_max_y.saturating_sub(view_start_y)) as f32;

        let start_col = ((span.px_x as f32 * cols as f32) / pw).round() as u16;
        let term_row = view_start_y + ((span.px_y as f32 * view_rows) / ph) as u16;
        let span_cols = ((span.px_w as f32 * cols as f32) / pw).round().max(1.0) as usize;

        assert_eq!(start_col, 8);
        assert_eq!(term_row, 17);
        assert_eq!(span_cols, 15);
    }
}
