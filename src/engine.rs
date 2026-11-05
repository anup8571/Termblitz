#![allow(dead_code)]
use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::Duration;

use crate::renderer::{Color, DiffRenderer, PixelBuffer};

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
}

impl NativeEngine {
    pub fn new() -> Self {
        Self {
            current_url: "about:home".to_string(),
            raw_html: String::new(),
            title: "termblitz home".to_string(),
            body_bg: Color::new(246, 246, 239), // Classic HN beige palette default
            lines: Vec::new(),
            links: Vec::new(),
            scroll_y: 0,
            max_scroll: 0,
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

        // Fetch URL via curl child process (supports HTTP/2, HTTPS, redirects, Brotli, Zero Rust compile weight)
        let output = Command::new("curl")
            .arg("-sL")
            .arg("-A")
            .arg("Mozilla/5.0 (Linux; Android 14; Termux) termblitz/0.1.0 NativeEngine")
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
    <p>Welcome to <b>termblitz</b> &mdash; engineered for maximum performance on Android / Termux.</p>
    <hr>
    <h2>Quick Jump Links:</h2>
    <ul>
        <li><a href="https://news.ycombinator.com">Hacker News (Y Combinator)</a></li>
        <li><a href="https://en.wikipedia.org/wiki/Main_Page">Wikipedia: The Free Encyclopedia</a></li>
        <li><a href="https://lobste.rs">Lobsters Tech Community</a></li>
        <li><a href="https://wttr.in/?format=v2">Live Weather Forecast (wttr.in)</a></li>
        <li><a href="https://lite.cnn.com">CNN Lite News</a></li>
        <li><a href="https://text.npr.org">NPR Text-Only News</a></li>
    </ul>
    <hr>
    <h2>Features & Engine Controls:</h2>
    <p>&bull; <b>Mode Toggle [m]:</b> Switch between <i>Native Ultra-Light Mode</i> (&lt; 1MB RAM) and <i>Full Headless Chromium Mode</i>.</p>
    <p>&bull; <b>Navigation [u]:</b> Focus address bar to enter a new URL.</p>
    <p>&bull; <b>Scrolling [j / k / Touch]:</b> Scroll up/down with arrow keys, vim keys, or touch swipe.</p>
    <p>&bull; <b>Touch Hit-Testing:</b> Tap directly on links on your screen to navigate.</p>
</body>
</html>"##
            .to_string()
    }

    /// Pure Rust streaming HTML/CSS layout engine with Table Row Horizontal Flow & Color Palettes
    pub fn parse_and_layout(&mut self, cols: u16) {
        self.lines.clear();
        self.links.clear();

        // Detect body background color
        let lower_html = self.raw_html.to_lowercase();
        if lower_html.contains("bgcolor=\"#f6f6ef\"")
            || lower_html.contains("bgcolor='#f6f6ef'")
            || lower_html.contains("bgcolor=#f6f6ef")
            || self.current_url.contains("news.ycombinator.com")
        {
            self.body_bg = Color::new(246, 246, 239); // Hacker News beige
        } else if lower_html.contains("bgcolor=\"#ffffff\"")
            || lower_html.contains("bgcolor='#ffffff'")
            || lower_html.contains("bgcolor=#ffffff")
            || lower_html.contains("bgcolor=\"white\"")
            || self.current_url.contains("wikipedia.org")
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

        let mut in_tag = false;
        let mut in_script_or_style = false;
        let mut in_table = false;
        let mut in_table_row = false;
        let mut current_tag = String::new();
        let mut text_acc = String::new();

        // Style stack
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
            let word_w = word.chars().count();
            if *line_len + word_w > max_w && *line_len > 0 {
                // Line wrap
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

        let chars: Vec<char> = self.raw_html.chars().collect();
        let mut i = 0;
        let len = chars.len();

        while i < len {
            let c = chars[i];

            if c == '<' {
                // Process accumulated text with entity decoding
                if !in_script_or_style && !text_acc.is_empty() {
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

                // Handle script/style exclusion
                if tag_name == "script" || tag_name == "style" {
                    in_script_or_style = true;
                } else if tag_name == "/script" || tag_name == "/style" {
                    in_script_or_style = false;
                }

                // Check bgcolor attribute
                if let Some(bg_str) = extract_attr(tag_raw, "bgcolor") {
                    if let Some(color) = parse_color_from_str(bg_str) {
                        current_row_bg = Some(color);
                        is_full_width_row = true;
                        if color == Color::new(255, 102, 0) {
                            // HN orange bar
                            current_fg = Color::BLACK;
                        }
                    }
                }

                // Check inline style colors
                if let Some(style_str) = extract_attr(tag_raw, "style") {
                    if let Some(fg) = extract_style_color(style_str) {
                        current_fg = fg;
                    }
                    if let Some(bg) = extract_style_bg(style_str) {
                        current_bg = Some(bg);
                    }
                }

                // Check semantic class names for Hacker News
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

                // Handle Elements & Layout
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
                        // Reset line row styling unless table set it
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
                                // Add horizontal column separator instead of breaking line!
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
                                // Indent for Hacker News subtext line under title
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
                    "/td" | "/th" => {
                        // Do not break line on closing td! Keeps table columns flowing horizontally!
                    }
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
                    "p" | "div" => {
                        // Only break line if outside a table
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
                    "/p" | "/div" => {
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
                    _ if tag_name.starts_with('a') => {
                        if let Some(href) = extract_attr(tag_raw, "href") {
                            active_link = Some(decode_html_entities(href));
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
            } else if !in_script_or_style {
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

    /// Renders NativeDocument lines into the DiffRenderer filling entire row widths with the canvas color
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

                // Fill ENTIRE row width with row background color (No black/grey void bleeding!)
                for x in 0..renderer.width {
                    renderer.set_cell(
                        x,
                        term_y,
                        crate::renderer::Cell {
                            ch: ' ',
                            fg: Color::WHITE,
                            bg: row_bg,
                            bold: false,
                            underline: false,
                        },
                    );
                }

                let mut term_x = if line.is_full_width_bg { 0 } else { 1 }; // Left margin
                for span in &line.spans {
                    let span_len = span.text.chars().count() as u16;
                    let span_bg = span.bg.unwrap_or(row_bg);

                    // Register clickable hyperlink hit-box
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
                // Clear empty lines with body canvas background
                for x in 0..renderer.width {
                    renderer.set_cell(
                        x,
                        term_y,
                        crate::renderer::Cell {
                            ch: ' ',
                            fg: Color::WHITE,
                            bg: self.body_bg,
                            bold: false,
                            underline: false,
                        },
                    );
                }
            }
        }
    }

    /// Resolves clicked terminal coordinates to a target link URL
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

/// Finds the index of the next unescaped quote character in a JSON string slice
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
        // Handshake timeout: 5 seconds for reliable connection even under high mobile CPU load
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

        // Set snappy 15ms timeout for active event loop polling
        stream.set_read_timeout(Some(Duration::from_millis(15))).ok();

        Ok(Self {
            stream,
            buf: Vec::with_capacity(131072),
        })
    }

    fn send_text(&mut self, text: &str) -> Result<(), String> {
        let payload = text.as_bytes();
        let len = payload.len();
        let mut frame = Vec::with_capacity(len + 14);

        frame.push(0x81); // FIN + text opcode

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

    /// Tries to extract one full WebSocket frame from self.buf without reading
    fn try_extract_frame(&mut self) -> Option<String> {
        if self.buf.len() < 2 {
            return None;
        }

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

        String::from_utf8(payload_bytes).ok()
    }

    fn poll_message(&mut self) -> Option<String> {
        // First check if an existing complete frame is already present in self.buf
        if let Some(msg) = self.try_extract_frame() {
            return Some(msg);
        }

        let mut chunk = [0u8; 32768];
        match self.stream.read(&mut chunk) {
            Ok(n) if n > 0 => {
                self.buf.extend_from_slice(&chunk[..n]);
            }
            _ => {}
        }

        self.try_extract_frame()
    }
}

const TEXT_EXTRACT_ID: u64 = 55555;

fn request_text_extraction(ws: &mut WsConnection, seq: &mut u64) {
    *seq += 1;
    let js = "(function(){\
        var T=String.fromCharCode(9),N=String.fromCharCode(10),r=[];\
        if(!document.body)return '';\
        var z=parseFloat(document.documentElement.style.zoom)||1.0;\
        var maxH=(window.innerHeight/z)+60;\
        var h=document.getElementById('termblitz-hide-text');\
        var prevMedia=h?h.media:'';\
        if(h)h.media='not all';\
        var tw=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);\
        var n;\
        while(n=tw.nextNode()){\
            var t=n.textContent;\
            if(!t||!t.trim())continue;\
            var p=n.parentElement;\
            if(!p)continue;\
            var s=getComputedStyle(p);\
            if(s.display==='none'||s.visibility==='hidden')continue;\
            var rng=document.createRange();\
            rng.selectNodeContents(n);\
            var rects=rng.getClientRects();\
            var rc=rects.length>0?rects[0]:p.getBoundingClientRect();\
            if(!rc||rc.width<1||rc.height<1)continue;\
            var left=~~(rc.left/z);\
            var top=~~(rc.top/z);\
            var width=~~(rc.width/z);\
            var height=~~(rc.height/z);\
            if(top<-30||top>maxH)continue;\
            var txt=t.trim().split(T).join(' ').split(N).join(' ');\
            r.push(txt+T+left+T+top+T+width+T+height+T+s.color+T+(s.fontWeight==='bold'||+s.fontWeight>=700?1:0)+T+(p.closest('a')?1:0)+T+s.backgroundColor);\
            if(r.length>400)break;\
        }\
        if(h)h.media=prevMedia;\
        return r.join(N);\
    })()";
    let cmd = format!(
        r#"{{"id":{},"method":"Runtime.evaluate","params":{{"expression":"{}"}}}}"#,
        TEXT_EXTRACT_ID, js
    );
    let _ = ws.send_text(&cmd);
}

fn spawn_cdp_worker(
    port: u16,
    initial_url: String,
    cols: u16,
    rows: u16,
    shared_buf: Arc<Mutex<PixelBuffer>>,
    shared_overlays: Arc<Mutex<Vec<TextOverlaySpan>>>,
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

                    // Specifically prioritize target with "type": "page" to avoid connecting to omnibox popups
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

        // 1) Mobile User-Agent override so modern websites serve mobile layout with large, readable fonts
        seq += 1;
        let ua_cmd = format!(
            r#"{{"id":{},"method":"Emulation.setUserAgentOverride","params":{{"userAgent":"Mozilla/5.0 (Linux; Android 14; Mobile) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36"}}}}"#,
            seq
        );
        let _ = ws.send_text(&ua_cmd);

        // 2) Emulation.setDeviceMetricsOverride (Option C: mobile portrait viewport reflow)
        let mut curr_cols = cols;
        let mut curr_rows = rows;
        let mut curr_zoom: f32 = if curr_cols < 60 { 2.0 } else if curr_cols < 90 { 1.5 } else { 1.0 };
        let mut curr_hide_text: bool = true;
        let mut width_px = 360u32.max((curr_cols as u32) * 8);
        let mut height_px = ((curr_rows.saturating_sub(1) as u32) * 16).max(480);
        seq += 1;
        let emu_cmd = format!(
            r#"{{"id":{},"method":"Emulation.setDeviceMetricsOverride","params":{{"width":{},"height":{},"deviceScaleFactor":1.0,"mobile":true,"fitWindow":false}}}}"#,
            seq, width_px, height_px
        );
        let _ = ws.send_text(&emu_cmd);

        // 3) Page.navigate
        seq += 1;
        let nav_cmd = format!(r#"{{"id":{},"method":"Page.navigate","params":{{"url":"{}"}}}}"#, seq, initial_url);
        let _ = ws.send_text(&nav_cmd);

        // 4) Page.startScreencast (High quality JPEG capture for fast transfer and sharp rendering)
        seq += 1;
        let sc_cmd = format!(
            r#"{{"id":{},"method":"Page.startScreencast","params":{{"format":"jpeg","quality":85,"maxWidth":{},"maxHeight":{},"everyNthFrame":1}}}}"#,
            seq, width_px, height_px
        );
        let _ = ws.send_text(&sc_cmd);

        // 5) Event listener & ACK streaming loop
        loop {
            while let Ok(cmd) = command_rx.try_recv() {
                match cmd {
                    CdpWorkerCommand::Navigate(new_url) => {
                        is_loading.store(true, Ordering::Relaxed);
                        frame_counter.store(0, Ordering::Relaxed);
                        shared_overlays.lock().unwrap().clear();
                        width_px = 360u32.max((curr_cols as u32) * 8);
                        height_px = ((curr_rows.saturating_sub(1) as u32) * 16).max(480);
                        seq += 1;
                        let emu = format!(
                            r#"{{"id":{},"method":"Emulation.setDeviceMetricsOverride","params":{{"width":{},"height":{},"deviceScaleFactor":1.0,"mobile":true,"fitWindow":false}}}}"#,
                            seq, width_px, height_px
                        );
                        let _ = ws.send_text(&emu);
                        seq += 1;
                        let msg = format!(r#"{{"id":{},"method":"Page.navigate","params":{{"url":"{}"}}}}"#, seq, new_url);
                        let _ = ws.send_text(&msg);
                    }
                    CdpWorkerCommand::SetHideText(hide) => {
                        curr_hide_text = hide;
                        seq += 1;
                        let media_str = if hide { "" } else { "not all" };
                        let js = format!(
                            "var h=document.getElementById('termblitz-hide-text');if(!h){{h=document.createElement('style');h.id='termblitz-hide-text';h.textContent='*{{color:transparent!important;text-shadow:none!important;-webkit-text-stroke:0!important;}}';document.head.appendChild(h);}}h.media='{}';",
                            media_str
                        );
                        let eval_cmd = format!(
                            r#"{{"id":{},"method":"Runtime.evaluate","params":{{"expression":"{}"}}}}"#,
                            seq, js
                        );
                        let _ = ws.send_text(&eval_cmd);
                        if hide {
                            request_text_extraction(&mut ws, &mut seq);
                        }
                    }
                    CdpWorkerCommand::Resize { cols: new_cols, rows: new_rows } => {
                        curr_cols = new_cols;
                        curr_rows = new_rows;
                        width_px = 360u32.max((curr_cols as u32) * 8);
                        height_px = ((curr_rows.saturating_sub(1) as u32) * 16).max(480);
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
                        request_text_extraction(&mut ws, &mut seq);
                    }
                    CdpWorkerCommand::Scroll(delta) => {
                        let center_x = width_px / 2;
                        let center_y = height_px / 2;
                        seq += 1;
                        let scroll_msg = format!(
                            r#"{{"id":{},"method":"Input.dispatchMouseEvent","params":{{"type":"mouseWheel","x":{},"y":{},"deltaX":0,"deltaY":{}}}}}"#,
                            seq, center_x, center_y, delta * 120
                        );
                        let _ = ws.send_text(&scroll_msg);
                        request_text_extraction(&mut ws, &mut seq);
                    }
                    CdpWorkerCommand::Click(col, row) => {
                        let px_x = ((col as f32 * width_px as f32) / curr_cols.max(1) as f32) as u32;
                        let px_y = (((row.saturating_sub(1) as f32) * height_px as f32) / (curr_rows.saturating_sub(1)).max(1) as f32) as u32;
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
                    }
                    CdpWorkerCommand::SetZoom(new_zoom) => {
                        curr_zoom = new_zoom;
                        seq += 1;
                        let zoom_js = format!("document.documentElement.style.zoom = '{:.2}';", curr_zoom);
                        let eval_cmd = format!(
                            r#"{{"id":{},"method":"Runtime.evaluate","params":{{"expression":"{}"}}}}"#,
                            seq, zoom_js
                        );
                        let _ = ws.send_text(&eval_cmd);
                        request_text_extraction(&mut ws, &mut seq);
                    }
                    CdpWorkerCommand::Stop => {
                        let _ = ws.send_text(r#"{"id":999,"method":"Page.stopScreencast"}"#);
                        return;
                    }
                }
            }

            while let Some(msg) = ws.poll_message() {
                if msg.contains(r#""Page.loadEventFired""#) {
                    is_loading.store(false, Ordering::Relaxed);
                    // Inject Option C mobile reflow CSS, CSS zoom, and transparent text CSS on page load
                    seq += 1;
                    let media_str = if curr_hide_text { "" } else { "not all" };
                    let setup_js = format!(
                        "document.documentElement.style.zoom = '{:.2}'; \
                         var s=document.getElementById('termblitz-reflow'); \
                         if(!s){{ s=document.createElement('style'); s.id='termblitz-reflow'; s.textContent='html,body{{max-width:100vw!important;overflow-x:hidden!important;box-sizing:border-box!important;word-break:break-word!important;}} img,video,canvas,svg{{max-width:100%!important;height:auto!important;}}'; document.head.appendChild(s); }} \
                         var h=document.getElementById('termblitz-hide-text'); \
                         if(!h){{ h=document.createElement('style'); h.id='termblitz-hide-text'; h.textContent='*{{color:transparent!important;text-shadow:none!important;-webkit-text-stroke:0!important;}}'; document.head.appendChild(h); }} \
                         h.media='{}';",
                        curr_zoom, media_str
                    );
                    let eval_cmd = format!(
                        r#"{{"id":{},"method":"Runtime.evaluate","params":{{"expression":"{}"}}}}"#,
                        seq, setup_js
                    );
                    let _ = ws.send_text(&eval_cmd);

                    // Option A: Extract visible DOM text nodes
                    request_text_extraction(&mut ws, &mut seq);
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
                                    });
                                }
                            }
                            *shared_overlays.lock().unwrap() = spans;
                        }
                    }
                }

                if msg.contains(r#""Page.screencastFrame""#) {
                    // Robust whitespace-tolerant sessionId parser
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

                    // Robust whitespace-tolerant data payload parser
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
                                    if (c == 1 || c == 3 || c == 6 || c == 10) && curr_hide_text {
                                        request_text_extraction(&mut ws, &mut seq);
                                    }
                                }
                            }
                        }
                    }

                    // Always acknowledge frame immediately to maintain continuous pipeline
                    seq += 1;
                    let ack = format!(
                        r#"{{"id":{},"method":"Page.screencastFrameAck","params":{{"sessionId":{}}}}}"#,
                        seq, session_id
                    );
                    let _ = ws.send_text(&ack);
                }
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
    pub frame_counter: Arc<AtomicUsize>,
    pub is_loading: Arc<AtomicBool>,
    pub zoom: f32,
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
            frame_counter: Arc::new(AtomicUsize::new(0)),
            is_loading: Arc::new(AtomicBool::new(true)),
            chromium_bin,
            command_tx: None,
            cols: 80,
            rows: 24,
            zoom: 2.0, // Crisp 200% default zoom for mobile terminal readability
        }
    }

    pub fn get_text_overlays(&self) -> Vec<TextOverlaySpan> {
        self.text_overlays.lock().unwrap().clone()
    }

    pub fn set_hide_text(&mut self, hide: bool) {
        if let Some(ref tx) = self.command_tx {
            let _ = tx.send(CdpWorkerCommand::SetHideText(hide));
        }
    }

    pub fn zoom_in(&mut self) {
        self.zoom = (self.zoom + 0.25).min(3.5);
        if let Some(ref tx) = self.command_tx {
            let _ = tx.send(CdpWorkerCommand::SetZoom(self.zoom));
        }
    }

    pub fn zoom_out(&mut self) {
        self.zoom = (self.zoom - 0.25).max(0.75);
        if let Some(ref tx) = self.command_tx {
            let _ = tx.send(CdpWorkerCommand::SetZoom(self.zoom));
        }
    }

    pub fn zoom_reset(&mut self) {
        self.zoom = if self.cols < 60 { 2.0 } else { 1.0 };
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

    /// Automatically searches for headless Chromium across Termux ($PREFIX), Debian/Linux (/usr/bin), macOS, and $PATH
    pub fn find_chromium_binary() -> Option<String> {
        // 1. Explicit Termux environment path ($PREFIX/bin/chromium)
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

        // 2. Standard Linux, Desktop, and macOS locations (prefer direct binary over shell wrapper)
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
            // macOS Desktop Applications
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        ];

        for p in &standard_candidates {
            if std::path::Path::new(p).exists() {
                return Some(p.to_string());
            }
        }

        // 3. Search all directories in $PATH
        let binary_names = [
            "chromium",
            "chromium-browser",
            "google-chrome",
            "google-chrome-stable",
            "chrome",
            "brave-browser",
            "microsoft-edge-stable",
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

        // 4. Execution check fallback
        for name in &binary_names {
            if let Ok(res) = Command::new(name).arg("--version").output() {
                if res.status.success() {
                    return Some(name.to_string());
                }
            }
        }

        None
    }

    /// Spawns headless Chromium with remote debugging port and window sizing
    pub fn ensure_browser_running(&mut self, window_w: u16, window_h: u16) -> Result<(), String> {
        if self.process.is_some() {
            return Ok(());
        }

        // Check if an existing Chromium instance is already healthy on this port
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
                    let err = format!("Chromium started (PID: {:?}) but failed to bind port {}", self.process.as_ref().map(|p| p.id()), self.port);
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
        self.cols = cols;
        self.rows = rows;
        self.is_loading.store(true, Ordering::Relaxed);

        let w_px = (cols as u32) * 8;
        let h_px = (rows.saturating_sub(1) as u32) * 16;

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
            // Clean up any child worker processes
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

        // Header row should have orange background
        assert_eq!(engine.lines[0].row_bg, Some(Color::new(255, 102, 0)));
        let header_text: String = engine.lines[0].spans.iter().map(|s| s.text.as_str()).collect();
        assert!(header_text.contains("Hacker News"));
        assert!(header_text.contains("new | past | comments"));

        // Row 1 should have "1." and "AI is forcing us to pay debt" on the same line!
        let story_text: String = engine.lines[1].spans.iter().map(|s| s.text.as_str()).collect();
        assert!(story_text.contains("1."));
        assert!(story_text.contains("AI is forcing us to pay debt"));

        // Row 2 should have decoded comments (not raw &nbsp;)
        let subtext: String = engine.lines[2].spans.iter().map(|s| s.text.as_str()).collect();
        assert!(subtext.contains("473 comments"));
        assert!(!subtext.contains("&nbsp;"));
    }

    #[test]
    fn test_render_mode_cycle() {
        assert_eq!(RenderMode::Native.next(true), RenderMode::CdpHybrid);
        assert_eq!(RenderMode::CdpHybrid.next(true), RenderMode::CdpRaw);
        assert_eq!(RenderMode::CdpRaw.next(true), RenderMode::Native);

        // When Chromium is missing, lock to Native
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
        assert_eq!(parse_color_from_str("rgba(0, 0, 0, 0)"), None); // Fully transparent
        assert_eq!(parse_color_from_str("rgb(10, 20, 30)"), Some(Color::new(10, 20, 30)));
    }

    #[test]
    fn test_json_unescape() {
        assert_eq!(json_unescape("hello\\nworld\\ttab"), "hello\nworld\ttab");
        assert_eq!(json_unescape("say \\\"hello\\\""), "say \"hello\"");
        assert_eq!(json_unescape("unicode \\u0041"), "unicode A");
    }

    #[test]
    fn test_chromium_resolver() {
        let bin = CdpEngine::find_chromium_binary();
        // Verifies the resolver runs without panic and identifies chromium in Debian if present
        if let Some(path) = bin {
            assert!(!path.is_empty());
        }
    }

    #[test]
    fn test_parse_simulated_frame() {
        let dummy_jpeg = std::fs::read("/tmp/screencast_frame.jpg").unwrap_or_default();
        if dummy_jpeg.is_empty() {
            return;
        }
        let b64 = crate::renderer::base64_encode(&dummy_jpeg);
        let msg = format!(
            r#"{{"method":"Page.screencastFrame","params":{{"data":"{}","metadata":{{}},"sessionId": 1}}}}"#,
            b64
        );

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
        assert_eq!(session_id, 1);

        if let Some(pos) = msg.find(r#""data""#) {
            let rest = &msg[pos + 6..];
            if let Some(q1) = rest.find('"') {
                let after = &rest[q1 + 1..];
                if let Some(q2) = after.find('"') {
                    let b64_extracted = &after[..q2];
                    let jpeg_bytes = base64_decode(b64_extracted);
                    assert_eq!(jpeg_bytes.len(), dummy_jpeg.len());
                    let img = image::load_from_memory(&jpeg_bytes);
                    assert!(img.is_ok());
                }
            }
        }
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

        let available_cols = (cols - start_col) as usize;
        let max_len = span_cols.max(span.text.chars().count()).min(available_cols);
        let display_text: String = span.text.chars().take(max_len).collect();
        assert_eq!(display_text, "Google offered in:");
    }
}
