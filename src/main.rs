pub mod renderer;
pub mod engine;
pub mod input;
pub mod upscale;

use std::env;
use std::fs;
use std::io::{self, stdout, Write};
use std::time::{Duration, Instant};

use crossterm::{
    cursor::{Hide, Show},
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};

use engine::{CdpEngine, InlineStyleRun, NativeEngine, RenderMode, RenderedHitbox, TextOverlaySpan};
use input::{resolve_input_to_url, AppAction, InputHandler};
use renderer::{
    display_width, has_complex_script, is_combining_char, Color, DiffRenderer, GraphicsProtocol,
    PixelBuffer, SubCellMode,
};
use upscale::UpscaleMode;

struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let mut out = stdout();
        let _ = out.write_all(b"\x1b[?2026l\x1b[?1006l\x1b[?1002l\x1b[?1000l\x1b_Ga=d,d=A\x1b[0m\x1b[2J\x1b[H");
        let _ = execute!(out, LeaveAlternateScreen, DisableMouseCapture, Show);
        let _ = disable_raw_mode();
        let _ = out.flush();
    }
}

fn get_memory_mb() -> f32 {
    if let Ok(statm) = fs::read_to_string("/proc/self/statm") {
        if let Some(rss_pages) = statm.split_whitespace().nth(1) {
            if let Ok(pages) = rss_pages.parse::<f32>() {
                return (pages * 4.0) / 1024.0;
            }
        }
    }
    0.0
}

/// Truncates a string to at most `max_cols` terminal display columns while preserving
/// any zero-width Unicode combining marks attached to the included base characters.
fn truncate_by_display_width(s: &str, max_cols: usize) -> String {
    if max_cols == 0 {
        return String::new();
    }
    let mut out = String::with_capacity(s.len().min(max_cols * 4));
    let mut cols = 0usize;
    for ch in s.chars() {
        if is_combining_char(ch) {
            if cols > 0 {
                out.push(ch);
            }
            continue;
        }
        if cols >= max_cols {
            break;
        }
        out.push(ch);
        cols += 1;
    }
    out
}

/// Strips `https://`, `http://`, and leading `www.` for clean display in narrow mobile status bars.
fn format_clean_url(url: &str) -> String {
    let u = url.trim();
    let without_scheme = u
        .strip_prefix("https://")
        .or_else(|| u.strip_prefix("http://"))
        .unwrap_or(u);
    let without_www = without_scheme
        .strip_prefix("www.")
        .unwrap_or(without_scheme);
    let cleaned = without_www.trim_end_matches('/');
    if cleaned.is_empty() {
        u.to_string()
    } else {
        cleaned.to_string()
    }
}

/// Word-wraps multi-line DOM text overlays within their bounding box column width (`max_width`)
/// across up to `max_lines` rows using true Unicode terminal column widths (`display_width`).
fn wrap_overlay_text(text: &str, max_width: usize, max_lines: usize) -> Vec<String> {
    let max_w = max_width.max(1);
    if max_lines <= 1 || display_width(text) <= max_w {
        return vec![truncate_by_display_width(text, max_w)];
    }

    let mut lines = Vec::with_capacity(max_lines.min(8));
    let mut current = String::new();
    let mut current_len = 0usize;

    for word in text.split_whitespace() {
        let word_len = display_width(word);
        if current_len == 0 {
            if word_len <= max_w {
                current.push_str(word);
                current_len = word_len;
            } else {
                lines.push(truncate_by_display_width(word, max_w));
                if lines.len() >= max_lines {
                    return lines;
                }
            }
        } else if current_len + 1 + word_len <= max_w {
            current.push(' ');
            current.push_str(word);
            current_len += 1 + word_len;
        } else {
            lines.push(std::mem::take(&mut current));
            if lines.len() >= max_lines {
                return lines;
            }
            if word_len <= max_w {
                current.push_str(word);
                current_len = word_len;
            } else {
                current.push_str(&truncate_by_display_width(word, max_w));
                current_len = max_w;
            }
        }
    }

    if !current.is_empty() && lines.len() < max_lines {
        lines.push(current);
    }

    if lines.is_empty() {
        lines.push(truncate_by_display_width(text, max_w));
    }
    lines
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StyledSeg {
    text: String,
    cols: usize,
    fg: Color,
    bold: bool,
    underline: bool,
    link_url: Option<String>,
}

#[derive(Clone, Debug)]
struct StyledLine {
    segs: Vec<StyledSeg>,
    total_cols: usize,
}

impl StyledLine {
    fn new() -> Self {
        Self {
            segs: Vec::new(),
            total_cols: 0,
        }
    }

    fn push_cluster(
        &mut self,
        base: char,
        comb: &[char],
        fg: Color,
        bold: bool,
        underline: bool,
        link_url: &Option<String>,
    ) {
        if let Some(last) = self.segs.last_mut() {
            if last.fg == fg
                && last.bold == bold
                && last.underline == underline
                && last.link_url == *link_url
            {
                last.text.push(base);
                for &c in comb {
                    last.text.push(c);
                }
                last.cols += 1;
                self.total_cols += 1;
                return;
            }
        }
        let mut text = String::with_capacity(1 + comb.len());
        text.push(base);
        for &c in comb {
            text.push(c);
        }
        self.segs.push(StyledSeg {
            text,
            cols: 1,
            fg,
            bold,
            underline,
            link_url: link_url.clone(),
        });
        self.total_cols += 1;
    }
}

#[derive(Clone)]
struct StyledCluster {
    base: char,
    comb: Vec<char>,
    is_ws: bool,
    fg: Color,
    bold: bool,
    underline: bool,
    link_url: Option<String>,
}

/// Word-wraps a `TextOverlaySpan` preserving inline bold/link/color styling per Unicode codepoint
/// across wrapped terminal lines so cohesive prose paragraphs flow naturally without colliding or dropping words.
fn wrap_styled_overlay_lines(
    span: &TextOverlaySpan,
    max_width: usize,
    max_lines: usize,
    inline_indent: usize,
) -> Vec<StyledLine> {
    let max_w = max_width.max(1);
    let max_l = max_lines.max(1);
    let mut clusters: Vec<StyledCluster> = Vec::new();
    let mut run_cursor = 0usize;
    let has_runs = !span.inline_runs.is_empty();
    let allow_underline_bg = span.bg.map_or(true, |b| b.luma_fast() > 215);

    for (char_idx, ch) in span.text.chars().enumerate() {
        if is_combining_char(ch) {
            if let Some(prev) = clusters.last_mut() {
                prev.comb.push(ch);
            }
            continue;
        }

        let active_run: Option<&InlineStyleRun> = if has_runs {
            while run_cursor < span.inline_runs.len()
                && span.inline_runs[run_cursor].end_char <= char_idx
            {
                run_cursor += 1;
            }
            span.inline_runs
                .get(run_cursor)
                .filter(|r| char_idx >= r.start_char && char_idx < r.end_char)
        } else {
            None
        };

        let is_link = active_run.map_or(span.is_link, |r| r.is_link);
        let raw_fg = active_run.and_then(|r| r.fg).unwrap_or(span.fg);
        let bold = active_run.map_or(span.bold, |r| r.bold);
        let link_url = active_run
            .and_then(|r| r.link_url.clone())
            .or_else(|| if is_link { span.link_url.clone() } else { None });

        let fg = if span.is_input {
            Color::new(40, 90, 195)
        } else if span.is_popup {
            Color::new(32, 33, 36)
        } else if is_link && (raw_fg.luma_fast() < 40 || raw_fg.b <= raw_fg.r.saturating_add(15)) {
            Color::new(26, 86, 219)
        } else {
            raw_fg
        };
        let underline = is_link && !span.is_input && !span.is_popup && allow_underline_bg;

        clusters.push(StyledCluster {
            base: ch,
            comb: Vec::new(),
            is_ws: ch.is_whitespace(),
            fg,
            bold,
            underline,
            link_url,
        });
    }

    if clusters.is_empty() {
        return Vec::new();
    }

    // Preserve exact spacing/truncation for single-line UI controls and inputs
    if max_l <= 1 || clusters.len() <= max_w.saturating_sub(inline_indent) {
        let mut line = StyledLine::new();
        let cap = if inline_indent > 0 && inline_indent < max_w {
            max_w - inline_indent
        } else {
            max_w
        };
        for cl in clusters.into_iter().take(cap) {
            line.push_cluster(
                cl.base,
                &cl.comb,
                cl.fg,
                cl.bold,
                cl.underline,
                &cl.link_url,
            );
        }
        return if line.total_cols > 0 { vec![line] } else { Vec::new() };
    }

    let mut lines: Vec<StyledLine> = Vec::with_capacity(max_l.min(16));
    let mut current = StyledLine::new();
    let mut line0_cap = if inline_indent > 0 && inline_indent + 6 < max_w {
        max_w - inline_indent
    } else {
        max_w
    };
    let mut active_cap = line0_cap;
    let mut pending_space: Option<StyledCluster> = None;

    let mut idx = 0usize;
    let n = clusters.len();
    while idx < n {
        if clusters[idx].is_ws {
            if current.total_cols > 0 && pending_space.is_none() {
                let mut sp = clusters[idx].clone();
                sp.base = ' ';
                pending_space = Some(sp);
            }
            idx += 1;
            continue;
        }

        let start = idx;
        while idx < n && !clusters[idx].is_ws {
            idx += 1;
        }
        let word = &clusters[start..idx];
        let word_w = word.len();
        let space_w = if current.total_cols > 0 && pending_space.is_some() {
            1
        } else {
            0
        };

        if current.total_cols > 0 && current.total_cols + space_w + word_w <= active_cap {
            if let Some(sp) = pending_space.take() {
                let sp_underline = sp.underline && sp.link_url.is_some();
                current.push_cluster(' ', &[], sp.fg, sp.bold, sp_underline, &sp.link_url);
            }
            for cl in word {
                current.push_cluster(
                    cl.base,
                    &cl.comb,
                    cl.fg,
                    cl.bold,
                    cl.underline,
                    &cl.link_url,
                );
            }
        } else {
            if current.total_cols > 0 {
                lines.push(current);
                if lines.len() >= max_l {
                    return lines;
                }
                current = StyledLine::new();
                line0_cap = max_w;
                active_cap = max_w;
                pending_space = None;
            }
            if word_w <= active_cap {
                for cl in word {
                    current.push_cluster(
                        cl.base,
                        &cl.comb,
                        cl.fg,
                        cl.bold,
                        cl.underline,
                        &cl.link_url,
                    );
                }
            } else {
                // Hard-wrap overly long unbroken token across lines
                let mut pos = 0usize;
                while pos < word_w {
                    let rem_cap = active_cap.saturating_sub(current.total_cols).max(1);
                    let take = rem_cap.min(word_w - pos);
                    for cl in &word[pos..pos + take] {
                        current.push_cluster(
                            cl.base,
                            &cl.comb,
                            cl.fg,
                            cl.bold,
                            cl.underline,
                            &cl.link_url,
                        );
                    }
                    pos += take;
                    if pos < word_w {
                        lines.push(current);
                        if lines.len() >= max_l {
                            return lines;
                        }
                        current = StyledLine::new();
                        active_cap = max_w;
                    }
                }
            }
        }
    }

    if current.total_cols > 0 && lines.len() < max_l {
        lines.push(current);
    }
    let _ = line0_cap;
    lines
}

fn print_help() {
    println!("\x1b[1m⚡ TermBlitz\x1b[0m :: Ultra-fast visual terminal browser for Linux & Android Termux");
    println!("\n\x1b[1mUSAGE:\x1b[0m\n  termblitz [OPTIONS] [URL or Search Query]");
    println!("\n\x1b[1mOPTIONS:\x1b[0m");
    println!("  --benchmark                  Run automated headless stress-test & benchmark against live mainstream websites");
    println!("  --benchmark-url=<URL>        Run automated stress-test & benchmark against a specific live URL");
    println!("  --native                     Force Native HTML terminal engine (zero Chromium overhead)");
    println!("  --hybrid / --cdp             Force Chromium Hybrid mode (60 FPS visual background + crisp terminal text)");
    println!("  --raw / --cdp-raw            Force Chromium Raw ANSI mode");
    println!("  --sixel                      Force Sixel graphics protocol");
    println!("  --kitty                      Force Kitty graphics protocol");
    println!("  --ansi                       Force Half-Block TrueColor ANSI protocol");
    println!("  --sextant                    Enable Unicode 13.0 2x3 Sextant sub-cell blitter (6x density, ANSI-6X)");
    println!("  --quadrant                   Enable Unicode 2x2 Quadrant sub-cell blitter (4x density, ANSI-4X)");
    println!("  --auto-sextant               Enable Hybrid Auto-ROI 2x3 Sextant sub-cell blitter (ANSI-6X inside ROIs)");
    println!("  --no-dither                  Disable 8x8 Blue Noise + Floyd-Steinberg dithering");
    println!("  --upscale=<MODE>             Set Super-Resolution mode (off, fsr, neural, combined, multipass)");
    println!("  --pixel-art                  Enable forced Scale2x pixel-art super-resolution");
    println!("  -h, --help                   Show this help message");
}

fn run_live_benchmark(single_url: Option<String>) -> io::Result<()> {
    let targets: Vec<String> = if let Some(u) = single_url {
        vec![u]
    } else {
        vec![
            "https://www.google.com".to_string(),
            "https://m.youtube.com".to_string(),
            "https://www.instagram.com".to_string(),
            "https://en.wikipedia.org".to_string(),
            "https://github.com".to_string(),
        ]
    };

    println!("\x1b[1;36m======================================================================\x1b[0m");
    println!("\x1b[1;32m⚡ TermBlitz Headless Diagnostic & Live Stress-Test Benchmark Loop\x1b[0m");
    println!("\x1b[1;36m======================================================================\x1b[0m");

    let cols = 80u16;
    let rows = 30u16;
    let mut all_passed = true;

    for (idx, url) in targets.iter().enumerate() {
        println!("\n\x1b[1;33m[{}/{}] Navigating to: {}\x1b[0m", idx + 1, targets.len(), url);
        let start_time = Instant::now();

        let port = 9222 + (idx as u16);
        let mut engine = CdpEngine::new(port);
        engine.upscale_mode = UpscaleMode::FsrOnly;
        engine.dithering_enabled = true;

        if let Err(e) = engine.ensure_browser_running(cols, rows) {
            eprintln!("\x1b[1;31m  ✗ Failed to start headless Chromium: {}\x1b[0m", e);
            all_passed = false;
            continue;
        }

        engine.navigate(url, cols, rows);

        let mut frames_captured = 0usize;
        let mut first_frame_time: Option<Duration> = None;
        let mut first_dom_time: Option<Duration> = None;
        let mut last_frame = PixelBuffer::default();

        let nav_wait_start = Instant::now();
        while nav_wait_start.elapsed() < Duration::from_secs(8) {
            std::thread::sleep(Duration::from_millis(60));
            let fc = engine.get_frame_count();
            if fc > frames_captured {
                frames_captured = fc;
                if first_frame_time.is_none() {
                    first_frame_time = Some(nav_wait_start.elapsed());
                    last_frame = engine.get_frame();
                }
            }
            let overlays = engine.get_text_overlays();
            if !overlays.is_empty() && first_dom_time.is_none() {
                first_dom_time = Some(nav_wait_start.elapsed());
            }
            if first_frame_time.is_some() && first_dom_time.is_some() {
                break;
            }
        }

        if last_frame.width == 0 {
            last_frame = engine.get_frame();
        }
        let overlays = engine.get_text_overlays();
        let overlays_count = overlays.len();
        let rois = engine.get_active_rois();
        let rois_count = rois.len();
        let rect_masks = engine.get_rect_masks();
        let popups_count = rect_masks.len();

        println!(
            "  • Captured: Frame {}x{} | Latency: {:.1}ms | Overlays: {} | ROIs: {} | Popups: {}",
            last_frame.width,
            last_frame.height,
            first_frame_time.unwrap_or(nav_wait_start.elapsed()).as_secs_f64() * 1000.0,
            overlays_count,
            rois_count,
            popups_count
        );

        let mut renderer = DiffRenderer::new(cols, rows);
        renderer.upscale_mode = UpscaleMode::FsrOnly;
        renderer.dithering_enabled = true;
        renderer.set_active_rois(rois.clone(), 360, 576);

        // Warm up blit
        renderer.blit_frame_cached(1, || last_frame.clone(), 1, rows);
        let _ = renderer.build_synchronized_frame();

        // High precision benchmark of DiffRenderer blit + VRS Subpixel + FSR + overlay compositing
        let blit_start = Instant::now();
        let iterations = 30;
        let mut total_ansi_bytes = 0usize;
        for _ in 0..iterations {
            renderer.blit_frame(&last_frame, 1, rows);
            for mask in &rect_masks {
                let (mc0, mr0, mc1, mr1) = mask.to_terminal_bounds(cols, rows, 360, 576);
                renderer.fill_popup_mask(mc0, mr0, mc1, mr1, mask.color);
            }
            for span in overlays.iter().take(50) {
                let start_col = ((span.px_x.max(0) as f32 * cols as f32) / 360.0).round().min(cols.saturating_sub(1) as f32) as u16;
                let term_row = 1 + ((span.px_y.max(0) as f32 * (rows - 1) as f32) / 576.0).round() as u16;
                renderer.put_str_with_existing_bg(
                    start_col,
                    term_row.min(rows.saturating_sub(1)),
                    &span.text,
                    span.fg,
                    span.bg,
                    span.bold,
                    span.is_link,
                );
            }
            let bytes = renderer.build_synchronized_frame();
            total_ansi_bytes += bytes;
        }
        let avg_blit_micros = (blit_start.elapsed().as_micros() as f64) / (iterations as f64);
        let avg_blit_ms = avg_blit_micros / 1000.0;
        let avg_ansi_bytes = total_ansi_bytes / iterations;

        println!(
            "  • Render Latency: \x1b[1;32m{:.3}ms\x1b[0m (Subpixel VRS + FSR + TrueColor Dither) | ANSI output: {} bytes",
            avg_blit_ms, avg_ansi_bytes
        );

        // Simulate Scroll Down & Up
        let scroll_start = Instant::now();
        engine.scroll_at(3, cols / 2, rows / 2);
        std::thread::sleep(Duration::from_millis(200));
        let pending_px = engine.get_pending_scroll_px();
        let scroll_pct = engine.get_scroll_pct();
        println!(
            "  • Simulated Scroll: dy=+3 -> offset: {}px, pct: {}% ({:.1}ms)",
            pending_px,
            scroll_pct,
            scroll_start.elapsed().as_secs_f64() * 1000.0
        );

        // Simulate Input / Search focus and typing
        let type_start = Instant::now();
        engine.focus_input(cols / 2, rows / 4);
        engine.type_text("termblitz");
        std::thread::sleep(Duration::from_millis(300));
        let typed_overlays = engine.get_text_overlays();
        let typed_masks = engine.get_rect_masks();
        println!(
            "  • Simulated Input: Typed 'termblitz' -> extractions: {} overlays, {} popup masks ({:.1}ms)",
            typed_overlays.len(),
            typed_masks.len(),
            type_start.elapsed().as_secs_f64() * 1000.0
        );

        let frame_ok = last_frame.width > 0 && last_frame.height > 0;
        let time_ok = avg_blit_ms <= 1.5;
        let status_ok = frame_ok && time_ok;

        if status_ok {
            println!(
                "\x1b[1;32m  ✓ SUCCESS: {} benchmark passed cleanly in {:.2}s\x1b[0m",
                url,
                start_time.elapsed().as_secs_f64()
            );
        } else {
            eprintln!(
                "\x1b[1;31m  ✗ FAIL/SLOW: {} (frame_ok={}, avg_blit_ms={:.2}ms)\x1b[0m",
                url, frame_ok, avg_blit_ms
            );
            all_passed = false;
        }

        engine.stop();
        std::thread::sleep(Duration::from_millis(200));
    }

    println!("\x1b[1;36m======================================================================\x1b[0m");
    if all_passed {
        println!("\x1b[1;32m✓ ALL LIVE WEBSITES BENCHMARKED SUCCESSFULLY (Render times < 1.5ms)!\x1b[0m");
        Ok(())
    } else {
        eprintln!("\x1b[1;31m✗ Some live benchmarks had issues.\x1b[0m");
        Err(io::Error::new(io::ErrorKind::Other, "Live benchmark failed"))
    }
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return Ok(());
    }

    let benchmark_url = args.iter().find_map(|a| {
        a.strip_prefix("--benchmark-url=")
            .or_else(|| a.strip_prefix("--benchmark="))
    }).map(|s| s.to_string());
    let mut has_bench_flag = benchmark_url.is_some() || args.iter().any(|a| a == "--benchmark" || a == "--benchmark-url");
    let mut bench_arg_url = benchmark_url;
    if bench_arg_url.is_none() {
        let mut iter = args.iter().skip(1);
        while let Some(arg) = iter.next() {
            if arg == "--benchmark-url" || arg == "--benchmark" {
                has_bench_flag = true;
                if let Some(next) = iter.next() {
                    if !next.starts_with("--") {
                        bench_arg_url = Some(next.clone());
                    }
                }
                break;
            }
        }
    }

    if has_bench_flag {
        return run_live_benchmark(bench_arg_url);
    }
    let raw_arg = args
        .iter()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "about:home".to_string());
    let initial_url = if raw_arg == "about:home" {
        raw_arg
    } else {
        resolve_input_to_url(&raw_arg)
    };

    let enable_sixel = args.iter().any(|a| a == "--sixel");
    let enable_kitty = args.iter().any(|a| a == "--kitty");
    let enable_ansi = args.iter().any(|a| a == "--ansi");
    let start_in_cdp = args.iter().any(|a| a == "--cdp" || a == "--hybrid");
    let start_in_raw = args.iter().any(|a| a == "--raw" || a == "--cdp-raw");
    let start_in_native = args.iter().any(|a| a == "--native");
    let no_dither = args.iter().any(|a| a == "--no-dither" || a == "--no-dithering");
    let start_pixel_art = args.iter().any(|a| a == "--pixel-art");
    let mut initial_subcell_mode = if args.iter().any(|a| a == "--sextant" || a == "--2x3") {
        SubCellMode::Sextant2x3
    } else if args.iter().any(|a| a == "--quadrant" || a == "--2x2") {
        SubCellMode::Quadrant2x2
    } else if args.iter().any(|a| a == "--auto-sextant" || a == "--roi-sextant") {
        SubCellMode::AutoRoiSextant
    } else {
        SubCellMode::HalfBlock1x2
    };

    let mut initial_upscale_mode = UpscaleMode::FsrOnly;
    let mut arg_iter = args.iter().skip(1).peekable();
    while let Some(arg) = arg_iter.next() {
        if let Some(val) = arg.strip_prefix("--upscale=") {
            if let Some(parsed) = UpscaleMode::from_cli_value(val) {
                initial_upscale_mode = parsed;
            }
        } else if arg == "--upscale" {
            if let Some(next_val) = arg_iter.peek() {
                if let Some(parsed) = UpscaleMode::from_cli_value(next_val) {
                    initial_upscale_mode = parsed;
                }
            }
        } else if let Some(val) = arg.strip_prefix("--subcell=") {
            if let Some(parsed) = SubCellMode::from_cli_value(val) {
                initial_subcell_mode = parsed;
            }
        }
    }

    enable_raw_mode()?;
    let mut out = stdout();
    execute!(out, EnterAlternateScreen, EnableMouseCapture, Hide)?;
    let _ = out.write_all(b"\x1b[2J\x1b[H\x1b[?1000h\x1b[?1002h\x1b[?1006h");
    let _ = out.flush();
    let _guard = TerminalGuard;

    let (mut cols, mut rows) = crossterm::terminal::size().unwrap_or((80, 24));
    let mut renderer = DiffRenderer::new(cols, rows);
    renderer.upscale_mode = initial_upscale_mode;
    renderer.dithering_enabled = !no_dither;
    renderer.pixel_art_mode = start_pixel_art;
    renderer.subcell_mode = initial_subcell_mode;

    if enable_kitty {
        renderer.protocol = GraphicsProtocol::Kitty;
    } else if enable_sixel {
        renderer.protocol = GraphicsProtocol::Sixel;
    } else if enable_ansi {
        renderer.protocol = GraphicsProtocol::HalfBlockAnsi;
    }

    let mut native_engine = NativeEngine::new();
    native_engine.upscale_mode = initial_upscale_mode;
    native_engine.dithering_enabled = !no_dither;

    let mut cdp_engine = CdpEngine::new(9222);
    cdp_engine.upscale_mode = initial_upscale_mode;
    cdp_engine.dithering_enabled = !no_dither;

    let mut input_handler = InputHandler::new();

    let mut mode = if start_in_native || cdp_engine.chromium_bin.is_none() {
        RenderMode::Native
    } else if start_in_raw {
        RenderMode::CdpRaw
    } else if start_in_cdp {
        RenderMode::CdpHybrid
    } else {
        detect_smart_mode(&initial_url, cdp_engine.chromium_bin.is_some())
    };

    let mut status_banner: Option<(String, Instant)> = None;
    let mut history: Vec<String> = vec![initial_url.clone()];
    let mut history_idx: usize = 0;

    if (start_in_cdp || start_in_raw) && cdp_engine.chromium_bin.is_none() {
        status_banner = Some((
            "Chromium missing: falling back to Native Mode".to_string(),
            Instant::now() + Duration::from_secs(4),
        ));
    }

    if mode == RenderMode::Native {
        native_engine.load_url(&initial_url, cols);
    } else {
        native_engine.current_url = initial_url.clone();
        cdp_engine.navigate(&initial_url, cols, rows);
        if mode == RenderMode::CdpRaw {
            cdp_engine.set_hide_text(false);
        }
    }

    let mut fps = 0.0;
    let mut frame_count = 0;
    let mut fps_timer = Instant::now();
    let mut cdp_last_count = 0;
    let mut cdp_fps = 0.0;
    let mut cached_mem_mb = get_memory_mb();
    let mut occupied_cells: Vec<bool> = Vec::with_capacity((cols as usize) * (rows as usize));
    let mut scroll_fast_until = Instant::now() - Duration::from_secs(1);
    let mut quality_settle_pending = false;
    let mut full_rows = rows;

    loop {
        let frame_start = Instant::now();

        if let Ok((w, h)) = crossterm::terminal::size() {
            if w != cols || h != rows {
                let width_changed = w != cols;
                cols = w;
                rows = h;
                // Only update full_rows and resize Chromium when terminal width changes or height grows.
                // When the Android soft keyboard opens (w == cols && h < full_rows), keep full_rows
                // fixed so webpage content is never vertically squished!
                if width_changed || h >= full_rows {
                    full_rows = h;
                    if mode.is_cdp() {
                        cdp_engine.resize(cols, full_rows);
                    }
                }
                renderer.resize(cols, rows);
                if width_changed {
                    native_engine.parse_and_layout(cols);
                }
            }
        }

        // Sync live URL from CDP when navigating via links or search inside headless Chromium
        if mode.is_cdp() {
            let live = cdp_engine.get_live_url();
            if !live.is_empty() && live != "about:blank" && live != native_engine.current_url {
                native_engine.current_url = live.clone();
                cdp_engine.current_url = live;
                input_handler.is_typing_in_page = false;
            }
        }

        let action = input_handler.poll_action(Duration::from_millis(16));
        match action {
            AppAction::Exit => break,
            AppAction::ToggleMode => {
                renderer.force_redraw();
                let has_chromium = cdp_engine.chromium_bin.is_some();
                if !has_chromium {
                    status_banner = Some((
                        "Chromium missing: locked in Native Mode".to_string(),
                        Instant::now() + Duration::from_secs(4),
                    ));
                    mode = RenderMode::Native;
                } else {
                    let old_mode = mode;
                    mode = mode.next(true);

                    match (old_mode, mode) {
                        (RenderMode::Native, RenderMode::CdpHybrid) => {
                            renderer.clear(Color::new(246, 246, 239));
                            cdp_engine.set_hide_text(true);
                            cdp_engine.navigate(&native_engine.current_url, cols, full_rows);
                            status_banner = Some((
                                "Hybrid Mode: Crisp DOM text + visual background".to_string(),
                                Instant::now() + Duration::from_secs(2),
                            ));
                        }
                        (RenderMode::CdpHybrid, RenderMode::CdpRaw) => {
                            cdp_engine.set_hide_text(false);
                            cdp_engine.set_rendered_hitboxes(Vec::new());
                            status_banner = Some((
                                "Raw Mode: Pure bitmap (Press [m] for sharp text)".to_string(),
                                Instant::now() + Duration::from_secs(2),
                            ));
                        }
                        (RenderMode::CdpRaw, RenderMode::Native) => {
                            cdp_engine.set_hide_text(true);
                            renderer.clear(native_engine.body_bg);
                            let curr = if !native_engine.current_url.is_empty() {
                                native_engine.current_url.clone()
                            } else {
                                cdp_engine.current_url.clone()
                            };
                            native_engine.load_url(&curr, cols);
                            status_banner = Some((
                                "Native Mode: Ultra-light pure Rust parser".to_string(),
                                Instant::now() + Duration::from_secs(2),
                            ));
                        }
                        _ => {}
                    }
                }
            }
            AppAction::ToggleAdBlock => {
                let next_state = !cdp_engine.adblock_enabled;
                native_engine.adblock_enabled = next_state;
                cdp_engine.set_adblock(next_state);
                status_banner = Some((
                    if next_state {
                        "[SHIELD] Brave + YouTube Ad-Shield: ENABLED".to_string()
                    } else {
                        "[SHIELD] Brave + YouTube Ad-Shield: DISABLED".to_string()
                    },
                    Instant::now() + Duration::from_secs(2),
                ));
            }
            AppAction::ToggleMute => {
                let muted = cdp_engine.toggle_mute();
                renderer.force_redraw();
                status_banner = Some((
                    if muted {
                        "[MUTE] Browser Audio: MUTED [Alt+m / M]".to_string()
                    } else {
                        "[SND] Browser Audio: UNMUTED [Alt+m / M]".to_string()
                    },
                    Instant::now() + Duration::from_secs(2),
                ));
            }
            AppAction::CycleUpscale => {
                let next_sr = renderer.upscale_mode.next();
                renderer.upscale_mode = next_sr;
                native_engine.upscale_mode = next_sr;
                cdp_engine.upscale_mode = next_sr;
                renderer.force_redraw();
                status_banner = Some((
                    format!("Super-Resolution [u]: {}", next_sr.name()),
                    Instant::now() + Duration::from_secs(2),
                ));
            }
            AppAction::ToggleDither => {
                let next_dither = !renderer.dithering_enabled;
                renderer.dithering_enabled = next_dither;
                native_engine.dithering_enabled = next_dither;
                cdp_engine.dithering_enabled = next_dither;
                renderer.force_redraw();
                status_banner = Some((
                    format!(
                        "Dithering [d]: {}",
                        if next_dither {
                            "ON (8x8 Blue Noise + Serpentine Floyd-Steinberg)"
                        } else {
                            "OFF"
                        }
                    ),
                    Instant::now() + Duration::from_secs(2),
                ));
            }
            AppAction::TogglePixelArt => {
                renderer.pixel_art_mode = !renderer.pixel_art_mode;
                renderer.force_redraw();
                status_banner = Some((
                    if renderer.pixel_art_mode {
                        "Pixel-Art Scale2x Mode [p]: ON (Forced Scale2x)".to_string()
                    } else {
                        "Pixel-Art Scale2x Mode [p]: AUTO (Content-Aware)".to_string()
                    },
                    Instant::now() + Duration::from_secs(2),
                ));
            }
            AppAction::CycleSubCell => {
                renderer.subcell_mode = renderer.subcell_mode.next();
                renderer.force_redraw();
                status_banner = Some((
                    format!(
                        "Sub-Cell Blitter [s]: {} [{}]",
                        renderer.subcell_mode.name(),
                        renderer.protocol_badge()
                    ),
                    Instant::now() + Duration::from_secs(2),
                ));
            }
            AppAction::HistoryBack => {
                input_handler.is_typing_in_page = false;
                if mode.is_cdp() {
                    cdp_engine.history_back();
                    status_banner = Some((
                        "◀ Back".to_string(),
                        Instant::now() + Duration::from_millis(1200),
                    ));
                } else if history_idx > 0 {
                    history_idx -= 1;
                    let target = history[history_idx].clone();
                    native_engine.load_url(&target, cols);
                    status_banner = Some((
                        "◀ Back".to_string(),
                        Instant::now() + Duration::from_millis(1200),
                    ));
                }
            }
            AppAction::HistoryForward => {
                input_handler.is_typing_in_page = false;
                if mode.is_cdp() {
                    cdp_engine.history_forward();
                    status_banner = Some((
                        "▶ Forward".to_string(),
                        Instant::now() + Duration::from_millis(1200),
                    ));
                } else if history_idx + 1 < history.len() {
                    history_idx += 1;
                    let target = history[history_idx].clone();
                    native_engine.load_url(&target, cols);
                    status_banner = Some((
                        "▶ Forward".to_string(),
                        Instant::now() + Duration::from_millis(1200),
                    ));
                }
            }
            AppAction::Navigate(target_url) => {
                input_handler.is_typing_in_page = false;
                let resolved = if target_url.starts_with('/') && !native_engine.current_url.is_empty() {
                    if let Ok(base) = url_base(&native_engine.current_url) {
                        format!("{}{}", base, target_url)
                    } else {
                        target_url
                    }
                } else {
                    target_url
                };

                if history.get(history_idx) != Some(&resolved) {
                    history.truncate(history_idx + 1);
                    history.push(resolved.clone());
                    history_idx = history.len() - 1;
                }

                // Auto-switch to Hybrid if user navigates from about:home to a visual site like Google
                if mode == RenderMode::Native
                    && cdp_engine.chromium_bin.is_some()
                    && detect_smart_mode(&resolved, true) == RenderMode::CdpHybrid
                {
                    mode = RenderMode::CdpHybrid;
                    cdp_engine.set_hide_text(true);
                }

                if mode == RenderMode::Native {
                    native_engine.load_url(&resolved, cols);
                } else {
                    native_engine.current_url = resolved.clone();
                    cdp_engine.navigate(&resolved, cols, full_rows);
                }
            }
            AppAction::TypeInPage(text) => {
                if mode.is_cdp() {
                    cdp_engine.type_text(&text);
                }
            }
            AppAction::PageKey(page_key) => {
                if mode.is_cdp() {
                    cdp_engine.send_page_key(page_key);
                }
            }
            AppAction::UnfocusPageInput => {
                if mode.is_cdp() {
                    cdp_engine.blur_input();
                }
            }
            AppAction::Click(click_x, click_y) => {
                if mode == RenderMode::Native {
                    input_handler.is_typing_in_page = false;
                    if let Some(target) = native_engine.hit_test(click_x, click_y) {
                        let resolved = if target.starts_with('/') {
                            if let Ok(base) = url_base(&native_engine.current_url) {
                                format!("{}{}", base, target)
                            } else {
                                target
                            }
                        } else {
                            target
                        };
                        history.truncate(history_idx + 1);
                        history.push(resolved.clone());
                        history_idx = history.len() - 1;

                        if cdp_engine.chromium_bin.is_some()
                            && detect_smart_mode(&resolved, true) == RenderMode::CdpHybrid
                        {
                            mode = RenderMode::CdpHybrid;
                            cdp_engine.set_hide_text(true);
                            native_engine.current_url = resolved.clone();
                            cdp_engine.navigate(&resolved, cols, full_rows);
                        } else {
                            native_engine.load_url(&resolved, cols);
                        }
                    }
                } else {
                    let layout_max_y = full_rows.max(rows);
                    let (hit_link, hit_input) = cdp_engine.hit_test_overlay(click_x, click_y, cols, layout_max_y);
                    if hit_input {
                        // Focus the DOM input element on the page and type directly into the webpage box
                        input_handler.is_entering_url = false;
                        input_handler.is_typing_in_page = true;
                        cdp_engine.focus_input(click_x, click_y);
                    } else if let Some(url) = hit_link {
                        input_handler.is_typing_in_page = false;
                        if !url.starts_with("javascript:") && !url.is_empty() {
                            native_engine.current_url = url.clone();
                            cdp_engine.navigate(&url, cols, full_rows);
                        } else {
                            cdp_engine.click(click_x, click_y);
                        }
                    } else {
                        input_handler.is_typing_in_page = false;
                        cdp_engine.click(click_x, click_y);
                    }
                }
            }
            AppAction::Scroll(delta) => {
                if mode == RenderMode::Native {
                    if delta > 0 {
                        native_engine.scroll_y = (native_engine.scroll_y + delta as usize).min(native_engine.max_scroll);
                    } else {
                        native_engine.scroll_y = native_engine.scroll_y.saturating_sub((-delta) as usize);
                    }
                } else {
                    let (ptr_col, ptr_row) = input_handler.last_pointer_pos;
                    cdp_engine.scroll_at(delta, ptr_col, ptr_row);
                    scroll_fast_until = Instant::now() + Duration::from_millis(160);
                    quality_settle_pending = true;
                }
            }
            AppAction::PageScroll(delta) => {
                let half_page = (rows as usize / 2).max(1);
                if mode == RenderMode::Native {
                    if delta > 0 {
                        native_engine.scroll_y = (native_engine.scroll_y + half_page).min(native_engine.max_scroll);
                    } else {
                        native_engine.scroll_y = native_engine.scroll_y.saturating_sub(half_page);
                    }
                } else {
                    let (ptr_col, ptr_row) = input_handler.last_pointer_pos;
                    cdp_engine.scroll_at(delta * 3, ptr_col, ptr_row);
                    scroll_fast_until = Instant::now() + Duration::from_millis(160);
                    quality_settle_pending = true;
                }
            }
            AppAction::Reload => {
                input_handler.is_typing_in_page = false;
                renderer.force_redraw();
                let curr = native_engine.current_url.clone();
                if mode == RenderMode::Native {
                    native_engine.load_url(&curr, cols);
                } else {
                    if cdp_engine.get_error().is_some() {
                        cdp_engine.stop();
                        cdp_engine.clear_error();
                    }
                    cdp_engine.navigate(&curr, cols, full_rows);
                }
            }
            AppAction::ZoomIn => {
                if mode.is_cdp() {
                    cdp_engine.zoom_in();
                    status_banner = Some((
                        format!("Zoom: {:.0}%", cdp_engine.zoom * 100.0),
                        Instant::now() + Duration::from_secs(2),
                    ));
                }
            }
            AppAction::ZoomOut => {
                if mode.is_cdp() {
                    cdp_engine.zoom_out();
                    status_banner = Some((
                        format!("Zoom: {:.0}%", cdp_engine.zoom * 100.0),
                        Instant::now() + Duration::from_secs(2),
                    ));
                }
            }
            AppAction::ZoomReset => {
                if mode.is_cdp() {
                    cdp_engine.zoom_reset();
                    status_banner = Some((
                        format!("Zoom: {:.0}%", cdp_engine.zoom * 100.0),
                        Instant::now() + Duration::from_secs(2),
                    ));
                }
            }
            AppAction::FocusUrlBar => {}
            AppAction::None => {}
        }

        let clear_bg = match mode {
            RenderMode::Native => native_engine.body_bg,
            RenderMode::CdpHybrid | RenderMode::CdpRaw => {
                if cdp_engine.get_error().is_some() {
                    Color::new(18, 20, 26)
                } else {
                    Color::WHITE
                }
            }
        };
        renderer.clear(clear_bg);

        // Row 0: Top Status & Controls Header
        // Row 1: Dedicated Separate https:// Address & Search Bar
        // Row 2..rows: Webpage Viewport
        let view_start_y = if rows > 4 { 2u16 } else { 1u16 };
        let view_max_y = rows;
        let layout_max_y = full_rows.max(rows);

        match mode {
            RenderMode::Native => {
                renderer.video_fast_path = false;
                renderer.scroll_fast_path = false;
                native_engine.render_to(&mut renderer, view_start_y, view_max_y);
            }
            RenderMode::CdpHybrid | RenderMode::CdpRaw => {
                let cdp_err = cdp_engine.get_error();
                if let Some(ref err) = cdp_err {
                    renderer.video_fast_path = false;
                    renderer.scroll_fast_path = false;
                    let box_w = (cols.saturating_sub(4)).min(70).max(28);
                    let start_x = (cols.saturating_sub(box_w)) / 2;
                    let start_y = (rows.saturating_sub(10)) / 2;

                    let err_bg = Color::new(45, 15, 15);
                    let err_fg = Color::WHITE;
                    let err_border = Color::RED;

                    let inner_w = (box_w.saturating_sub(2)) as usize;
                    let horiz = "═".repeat(inner_w);
                    let blank_inner = format!("║{}║", " ".repeat(inner_w));
                    let top_border = format!("╔{}╗", horiz);
                    let mid_border = format!("╠{}╣", horiz);
                    let bot_border = format!("╚{}╝", horiz);

                    // Pre-fill all interior rows of the error box so there are never uncolored white gaps
                    for by in 1..9 {
                        renderer.put_str(start_x, start_y + by, &blank_inner, err_border, err_bg, false, false);
                    }

                    renderer.put_str(start_x, start_y, &top_border, err_border, err_bg, true, false);
                    let title_str = truncate_by_display_width(" ⚠️  CDP-WEB Engine Notice", inner_w);
                    renderer.put_str(start_x + 1, start_y + 1, &title_str, err_fg, err_bg, true, false);
                    renderer.put_str(start_x, start_y + 2, &mid_border, err_border, err_bg, false, false);

                    let wrapped_err = wrap_overlay_text(err, inner_w.saturating_sub(2), 4);
                    for (idx, eline) in wrapped_err.iter().take(4).enumerate() {
                        let y = start_y + 3 + (idx as u16);
                        renderer.put_str(start_x + 2, y, eline, Color::LIGHT_GRAY, err_bg, false, false);
                    }

                    let prompt1 = truncate_by_display_width(" • Press [m] to switch to Native Mode", inner_w);
                    let prompt2 = truncate_by_display_width(" • Press [r] to retry Chromium", inner_w);
                    renderer.put_str(start_x + 1, start_y + 7, &prompt1, Color::YELLOW, err_bg, true, false);
                    renderer.put_str(start_x + 1, start_y + 8, &prompt2, Color::CYAN, err_bg, false, false);
                    renderer.put_str(start_x, start_y + 9, &bot_border, err_border, err_bg, true, false);
                } else {
                    let (fw, fh) = cdp_engine.get_frame_dims();
                    let is_loading = cdp_engine.is_loading();
                    let count = cdp_engine.get_frame_count();
                    let video_rects = cdp_engine.get_video_rects();
                    let (vw, vh) = cdp_engine.get_viewport_dims();
                    renderer.set_active_rois(cdp_engine.get_active_rois(), vw, vh);
                    renderer.video_fast_path = !video_rects.is_empty();

                    // Two-Gear Kinetic Frame Pacer:
                    // Gear 1 (< 160ms since last scroll): Kinetic Motion Mode (1-tap Q16.16 blitter)
                    // Gear 2 (>= 160ms after scroll stops): High-Precision Rest Mode (Quality Settling Pass)
                    let in_kinetic_motion = Instant::now() < scroll_fast_until;
                    if !in_kinetic_motion && quality_settle_pending {
                        quality_settle_pending = false;
                        renderer.invalidate_frame_cache();
                    }
                    renderer.scroll_fast_path = in_kinetic_motion;
                    let scroll_delta_px = cdp_engine.get_pending_scroll_px();
                    renderer.pending_scroll_px = scroll_delta_px;

                    if count == 0 || (is_loading && count < 2) {
                        let msg = " [Loading Visual Web Page...] ";
                        let msg_x = cols.saturating_sub(msg.len() as u16) / 2;
                        let msg_y = rows / 2;
                        renderer.put_str(msg_x, msg_y, msg, Color::CYAN, Color::new(28, 32, 42), true, false);
                    } else if fw > 0 && fh > 0 {
                        // 1. Blit visual background frame using dirty-frame cell cache (only runs Super-Resolution
                        // or clones PixelBuffer when `count` or renderer fast-path/settings change)
                        renderer.blit_frame_cached(
                            count,
                            || cdp_engine.get_frame(),
                            view_start_y,
                            view_max_y,
                        );

                        // 2. In Hybrid Mode: Overlay razor-sharp native terminal characters at 1:1 CSS viewport coordinates
                        // while suppressing stale DOM text overlays inside any actively playing <video> bounding box,
                        // and clearing underlying page text beneath elevated dropdown popups (`POPUP_MASK`).
                        if mode == RenderMode::CdpHybrid {
                            let mut overlays = cdp_engine.get_text_overlays();
                            let rect_masks = cdp_engine.get_rect_masks();
                            // Prioritize inputs first, then sticky/fixed headers & popups, then cohesive
                            // prose blocks in strict top-to-bottom document reading order, then standalone links/text.
                            overlays.sort_by_key(|s| {
                                let prio = if s.is_input {
                                    0u8
                                } else if s.is_fixed || s.is_popup {
                                    1u8
                                } else if !s.inline_runs.is_empty() {
                                    2u8
                                } else if s.bold || s.is_link {
                                    3u8
                                } else {
                                    4u8
                                };
                                (prio, s.px_y, s.px_x)
                            });

                            let pw = vw.max(1) as f32;
                            let ph = vh.max(1) as f32;
                            // Use layout_max_y (full_rows) so opening the Android soft keyboard never squishes text vertically!
                            let view_rows = (layout_max_y.saturating_sub(view_start_y)).max(1) as f32;
                            let video_term_boxes: Vec<(u16, u16, u16, u16)> = video_rects
                                .iter()
                                .map(|vr| vr.to_terminal_bounds(cols, layout_max_y, vw, vh))
                                .collect();

                            let cols_usize = cols as usize;
                            occupied_cells.clear();
                            occupied_cells.resize(cols_usize * (rows as usize), false);
                            let mut rendered_hitboxes: Vec<RenderedHitbox> =
                                Vec::with_capacity(overlays.len().min(256));

                            // Pass 2a: Render base document overlays (!span.is_popup), then apply POPUP_MASK popup
                            // clearing, then render elevated popup/autocomplete suggestion items (span.is_popup) on top.
                            for pass_popup in [false, true] {
                                if pass_popup {
                                    for mask in &rect_masks {
                                        let (mc0, mr0, mc1, mr1) =
                                            mask.to_terminal_bounds(cols, layout_max_y, vw, vh);
                                        renderer.fill_popup_mask(mc0, mr0, mc1, mr1.min(view_max_y), mask.color);
                                        for ry in mr0.max(view_start_y)..mr1.min(view_max_y) {
                                            let r_base = (ry as usize) * cols_usize;
                                            let c0_u = (mc0 as usize).min(cols_usize);
                                            let c1_u = (mc1 as usize).min(cols_usize);
                                            for occ in &mut occupied_cells[r_base + c0_u..r_base + c1_u] {
                                                *occ = false;
                                            }
                                        }
                                    }
                                }

                                for span in overlays.iter().filter(|s| s.is_popup == pass_popup) {
                                    let eff_px_y = span.effective_px_y(scroll_delta_px);
                                    let span_px_h = span.px_h.max(12);
                                    if eff_px_y + span_px_h <= 2
                                        || eff_px_y >= (vh as i32).saturating_sub(2)
                                    {
                                        continue;
                                    }
                                    let start_col = ((span.px_x.max(0) as f32 * cols as f32) / pw)
                                        .round()
                                        .min(cols.saturating_sub(1) as f32) as u16;
                                    let start_row_i32 = (view_start_y as i32)
                                        + ((eff_px_y as f32 * view_rows) / ph).round() as i32;
                                    let span_cols = ((span.px_w.max(8) as f32 * cols as f32) / pw)
                                        .round()
                                        .max(1.0) as usize;
                                    let box_rows = ((span_px_h as f32 * view_rows) / ph)
                                        .ceil()
                                        .max(1.0) as usize;

                                    if start_col >= cols || start_row_i32 >= view_max_y as i32 {
                                        continue;
                                    }

                                    let available_cols = (cols - start_col) as usize;
                                    let has_inline_runs = !span.inline_runs.is_empty();
                                    let is_multiline = !span.is_input
                                        && !span.is_popup
                                        && (has_inline_runs || (box_rows >= 2 && span.px_h >= 24));
                                    let text_disp_w = display_width(&span.text);
                                    let wrap_w = if is_multiline {
                                        span_cols.min(available_cols).max(8)
                                    } else {
                                        span_cols.max(text_disp_w).min(available_cols)
                                    };
                                    let max_lines = if is_multiline {
                                        box_rows.max(24).min(32)
                                    } else {
                                        1
                                    };

                                    // For standalone plain text following an inline link sibling on line 0, detect
                                    // existing occupied cells on `start_row_i32`; cohesive blocks start cleanly on their own block row.
                                    let mut inline_indent = 0usize;
                                    if !has_inline_runs
                                        && start_row_i32 >= view_start_y as i32
                                        && start_row_i32 < view_max_y as i32
                                    {
                                        let first_row_base = (start_row_i32 as usize) * cols_usize;
                                        let sc_u = (start_col as usize).min(cols_usize);
                                        let scan_end = (sc_u + wrap_w).min(cols_usize);
                                        if sc_u < scan_end && occupied_cells[first_row_base + sc_u] {
                                            let mut k = sc_u;
                                            while k < scan_end {
                                                if occupied_cells[first_row_base + k] {
                                                    inline_indent = (k + 2).saturating_sub(sc_u);
                                                    k += 1;
                                                } else {
                                                    break;
                                                }
                                            }
                                        }
                                        if inline_indent + 6 >= wrap_w {
                                            inline_indent = 0;
                                        }
                                    }

                                    let wrapped_lines = wrap_styled_overlay_lines(
                                        span,
                                        wrap_w,
                                        max_lines,
                                        inline_indent,
                                    );

                                    // Push cohesive multi-line paragraphs down gently if an earlier paragraph wrapped
                                    // into slightly more terminal rows than CSS height predicted.
                                    let mut vertical_push: i32 = 0;
                                    if has_inline_runs && !span.is_fixed && !span.is_popup && start_row_i32 >= view_start_y as i32 {
                                        let c0_u = (start_col as usize).min(cols_usize);
                                        let c1_u = (start_col as usize + wrap_w.min(12)).min(cols_usize);
                                        while vertical_push < 4 {
                                            let probe_y = start_row_i32 + vertical_push;
                                            if probe_y >= view_max_y as i32 {
                                                break;
                                            }
                                            let r_base = (probe_y as usize) * cols_usize;
                                            if c0_u < c1_u
                                                && occupied_cells[r_base + c0_u..r_base + c1_u]
                                                    .iter()
                                                    .any(|&occ| occ)
                                            {
                                                vertical_push += 1;
                                            } else {
                                                break;
                                            }
                                        }
                                    }

                                    for (line_offset, styled_line) in wrapped_lines.iter().enumerate() {
                                        let target_y_i32 =
                                            start_row_i32 + (line_offset as i32) + vertical_push;
                                        if target_y_i32 < view_start_y as i32 {
                                            continue;
                                        }
                                        if target_y_i32 >= view_max_y as i32 {
                                            break;
                                        }
                                        let target_y = target_y_i32 as u16;
                                        let mut draw_col = if line_offset == 0 && inline_indent > 0 {
                                            start_col.saturating_add(inline_indent as u16).min(cols)
                                        } else {
                                            start_col
                                        };
                                        if draw_col >= cols || styled_line.total_cols == 0 {
                                            continue;
                                        }

                                        let max_rem_cols = (cols - draw_col) as usize;
                                        let line_w = styled_line.total_cols.min(max_rem_cols) as u16;
                                        if line_w == 0 {
                                            continue;
                                        }
                                        let mut line_end_col = draw_col.saturating_add(line_w).min(cols);

                                        let inside_video = video_term_boxes.iter().any(|&(vc0, vr0, vc1, vr1)| {
                                            target_y >= vr0
                                                && target_y < vr1
                                                && draw_col < vc1
                                                && line_end_col > vc0
                                        });
                                        if inside_video {
                                            continue;
                                        }

                                        let row_base = (target_y as usize) * cols_usize;
                                        let check_c0 = (draw_col as usize).min(cols_usize);
                                        let check_c1 = (line_end_col as usize).min(cols_usize);
                                        if check_c0 < check_c1
                                            && occupied_cells[row_base + check_c0..row_base + check_c1]
                                                .iter()
                                                .any(|&occ| occ)
                                        {
                                            if has_inline_runs {
                                                continue;
                                            }
                                            // Scan right on the same row to see if there is unoccupied space after an inline sibling
                                            let mut last_occ = check_c0;
                                            for c_idx in check_c0..cols_usize {
                                                if occupied_cells[row_base + c_idx] {
                                                    last_occ = c_idx;
                                                } else if c_idx > last_occ + 1 {
                                                    break;
                                                }
                                            }
                                            let shifted_col = ((last_occ + 2) as u16).min(cols);
                                            if shifted_col + 4 <= cols
                                                && !occupied_cells[row_base + (shifted_col as usize)..row_base + cols_usize]
                                                    .iter()
                                                    .take(4)
                                                    .any(|&occ| occ)
                                            {
                                                draw_col = shifted_col;
                                                let rem = (cols - draw_col) as usize;
                                                line_end_col = draw_col
                                                    .saturating_add(styled_line.total_cols.min(rem) as u16)
                                                    .min(cols);
                                            } else {
                                                continue;
                                            }
                                        }

                                        let mut cursor_col = draw_col;
                                        for seg in &styled_line.segs {
                                            if cursor_col >= cols {
                                                break;
                                            }
                                            let seg_avail = (cols - cursor_col) as usize;
                                            let clipped_seg = truncate_by_display_width(&seg.text, seg_avail);
                                            let drawn_w = display_width(&clipped_seg) as u16;
                                            if drawn_w == 0 {
                                                continue;
                                            }
                                            let should_underline = seg.underline && !has_complex_script(&clipped_seg);
                                            renderer.put_str_with_existing_bg(
                                                cursor_col,
                                                target_y,
                                                &clipped_seg,
                                                seg.fg,
                                                span.bg,
                                                seg.bold,
                                                should_underline,
                                            );
                                            let seg_end = cursor_col.saturating_add(drawn_w).min(cols);
                                            if span.is_input || seg.link_url.is_some() {
                                                rendered_hitboxes.push(RenderedHitbox {
                                                    col_start: cursor_col,
                                                    col_end: seg_end,
                                                    row: target_y,
                                                    link_url: seg.link_url.clone(),
                                                    is_input: span.is_input,
                                                });
                                            }
                                            cursor_col = seg_end;
                                        }

                                        let mark_c0 = (draw_col as usize).min(cols_usize);
                                        let mark_c1 = (cursor_col.max(line_end_col) as usize).min(cols_usize);
                                        for cell_occ in &mut occupied_cells[row_base + mark_c0..row_base + mark_c1] {
                                            *cell_occ = true;
                                        }
                                    }
                                }
                            }

                            cdp_engine.set_rendered_hitboxes(rendered_hitboxes);
                        }

                        if is_loading {
                            let badge = " [Updating...] ";
                            let bx = cols.saturating_sub(badge.len() as u16 + 1);
                            renderer.put_str(bx, view_start_y, badge, Color::YELLOW, Color::new(28, 32, 42), true, false);
                        }
                    }
                }
            }
        }

        // ─────────────────────────────────────────────────────────────────────────
        // ROW 0: Top Status & Controls Header (Pure 1-col ASCII badges, never wraps)
        // ─────────────────────────────────────────────────────────────────────────
        let bar_bg = Color::new(24, 28, 38);
        let bar_fg = Color::WHITE;
        let is_compact = cols < 72;
        let zoom_pct = (cdp_engine.zoom * 100.0).round() as u32;
        let mode_badge = match mode {
            RenderMode::Native => (
                if is_compact { " NAT ".to_string() } else { " NATIVE ".to_string() },
                Color::BLACK,
                Color::GREEN,
            ),
            RenderMode::CdpHybrid => {
                if cdp_engine.get_error().is_some() {
                    (" ERR ".to_string(), Color::WHITE, Color::RED)
                } else if is_compact {
                    (format!(" HYB {}% ", zoom_pct), Color::BLACK, Color::new(255, 180, 0))
                } else {
                    (format!(" HYBRID [{}%] ", zoom_pct), Color::BLACK, Color::new(255, 180, 0))
                }
            }
            RenderMode::CdpRaw => {
                if cdp_engine.get_error().is_some() {
                    (" ERR ".to_string(), Color::WHITE, Color::RED)
                } else if is_compact {
                    (format!(" RAW {}% ", zoom_pct), Color::WHITE, Color::new(140, 80, 220))
                } else {
                    (format!(" RAW [{}%] ", zoom_pct), Color::WHITE, Color::new(140, 80, 220))
                }
            }
        };

        renderer.put_str(0, 0, &" ".repeat(cols as usize), bar_fg, bar_bg, false, false);
        renderer.put_str(0, 0, &mode_badge.0, mode_badge.1, mode_badge.2, true, false);

        let addr_x = mode_badge.0.len() as u16;
        let has_active_banner = if let Some((ref msg, expiry)) = status_banner {
            if Instant::now() < expiry {
                let max_banner_w = (cols.saturating_sub(addr_x)) as usize;
                let banner = truncate_by_display_width(&format!(" {} ", msg), max_banner_w);
                renderer.put_str(addr_x, 0, &banner, Color::YELLOW, Color::new(50, 40, 20), true, false);
                true
            } else {
                false
            }
        } else {
            false
        };

        if !has_active_banner {
            let mem = cached_mem_mb;
            let scroll_pct = if mode.is_cdp() {
                cdp_engine.get_scroll_pct()
            } else if native_engine.max_scroll > 0 {
                (native_engine.scroll_y * 100) / native_engine.max_scroll
            } else {
                100
            };
            // Pure 1-column ASCII audio indicator so terminal wcwidth is always 100% exact
            let audio_icon = if cdp_engine.audio_muted { "MUTE" } else { "SND" };
            let shield_tag = if cdp_engine.adblock_enabled {
                if is_compact { "SHLD" } else { "SHIELD" }
            } else {
                "OFF"
            };
            let fps_display = if mode.is_cdp() {
                if cdp_fps > 0.0 { format!("{:0.0}fps", cdp_fps) } else { "idle".to_string() }
            } else {
                format!("{:0.0}fps", fps)
            };
            let dither_state = if renderer.dithering_enabled { "ON" } else { "OFF" };
            let pix_tag = if renderer.pixel_art_mode { "+PIX" } else { "" };
            let sr_part = if is_compact {
                format!("[SR:{}{}", renderer.upscale_mode.short_name(), pix_tag)
            } else {
                format!("[SR: {}{}", renderer.upscale_mode.badge_name(), pix_tag)
            };
            let dither_part = if is_compact {
                format!("|D:{}]", dither_state)
            } else {
                format!(" | DITHER: {}]", dither_state)
            };
            let sr_badge = format!("{}{}", sr_part, dither_part);
            let mute_sep = if is_compact { "|" } else { " | " };
            let metrics = if is_compact {
                format!(
                    " {}{}{}|{}|{:0.0}M|{}% ",
                    sr_badge, mute_sep, audio_icon, shield_tag, mem, scroll_pct
                )
            } else {
                format!(
                    " {}{}{} | {} | {} | {:0.1}MB | {} | {}% ",
                    sr_badge,
                    mute_sep,
                    audio_icon,
                    shield_tag,
                    renderer.protocol_badge(),
                    mem,
                    fps_display,
                    scroll_pct
                )
            };
            let metrics_cols = display_width(&metrics) as u16;
            let metrics_x = cols.saturating_sub(metrics_cols).max(addr_x);

            let sr_col_start = metrics_x + 1;
            let sr_col_end = sr_col_start + (display_width(&sr_part) as u16);
            let dither_col_end = sr_col_end + (display_width(&dither_part) as u16);
            let mute_col_start = dither_col_end + (display_width(mute_sep) as u16);
            let mute_col_end = mute_col_start + (display_width(audio_icon) as u16);
            input_handler.set_all_status_hit_zones(
                Some((sr_col_start, sr_col_end)),
                Some((sr_col_end, dither_col_end)),
                Some((mute_col_start, mute_col_end)),
            );
            let clipped_metrics = truncate_by_display_width(&metrics, (cols.saturating_sub(metrics_x)) as usize);
            renderer.put_str(metrics_x, 0, &clipped_metrics, Color::LIGHT_GRAY, bar_bg, false, false);
        } else {
            input_handler.set_all_status_hit_zones(None, None, None);
        }

        // ─────────────────────────────────────────────────────────────────────────
        // ROW 1: Dedicated Separate https:// Address & Search Bar
        // ─────────────────────────────────────────────────────────────────────────
        if rows > 4 {
            let url_bar_bg = Color::new(36, 42, 56);
            let pill_bg = Color::new(16, 20, 28);
            renderer.put_str(0, 1, &" ".repeat(cols as usize), Color::WHITE, url_bar_bg, false, false);

            if input_handler.is_entering_url {
                let prefix = if is_compact { " https:// " } else { " https:// or Search: " };
                renderer.put_str(0, 1, prefix, Color::new(110, 220, 255), pill_bg, true, false);
                let input_x = prefix.len() as u16;
                let avail_w = (cols.saturating_sub(input_x + 1)) as usize;
                let max_input_chars = avail_w.saturating_sub(2).max(4);
                let input_chars: Vec<char> = input_handler.url_input_buffer.chars().collect();
                let visible_input: String = if input_chars.len() > max_input_chars {
                    input_chars[input_chars.len() - max_input_chars..].iter().collect()
                } else {
                    input_handler.url_input_buffer.clone()
                };
                let prompt = format!(" {}█", visible_input);
                renderer.put_str(input_x, 1, &prompt, Color::CYAN, url_bar_bg, true, false);
            } else if input_handler.is_typing_in_page {
                let badge = " INPUT ";
                renderer.put_str(0, 1, badge, Color::BLACK, Color::CYAN, true, false);
                let hint = if is_compact {
                    " Typing in page box [Enter=Go | Esc=Exit]"
                } else {
                    " Typing in webpage input box [Enter = Submit | Esc = Exit]"
                };
                let max_w = (cols.saturating_sub(badge.len() as u16)) as usize;
                let clipped = truncate_by_display_width(hint, max_w);
                renderer.put_str(badge.len() as u16, 1, &clipped, Color::CYAN, url_bar_bg, true, false);
            } else {
                let raw_url = if !native_engine.current_url.is_empty() {
                    native_engine.current_url.as_str()
                } else {
                    cdp_engine.current_url.as_str()
                };
                let scheme_badge = if raw_url.starts_with("http://") {
                    " http:// "
                } else if raw_url.starts_with("about:") {
                    " about: "
                } else {
                    " https:// "
                };
                renderer.put_str(0, 1, scheme_badge, Color::new(110, 230, 180), pill_bg, true, false);
                let url_x = scheme_badge.len() as u16;
                let hint = if is_compact { " [/]" } else { " [Tap or / to Search] " };
                let hint_w = hint.len() as u16;
                let hint_x = cols.saturating_sub(hint_w);
                if hint_x > url_x + 4 {
                    renderer.put_str(hint_x, 1, hint, Color::GRAY, url_bar_bg, false, false);
                }
                let max_url_w = (hint_x.saturating_sub(url_x)) as usize;
                let clean_host_path = format!(" {}", format_clean_url(raw_url));
                let clipped_url = truncate_by_display_width(&clean_host_path, max_url_w);
                renderer.put_str(url_x, 1, &clipped_url, Color::WHITE, url_bar_bg, true, false);
            }
        }

        let _ = renderer.flush();

        frame_count += 1;
        let elapsed_secs = fps_timer.elapsed().as_secs_f32();
        if elapsed_secs >= 0.5 {
            fps = (frame_count as f32) / elapsed_secs;
            cached_mem_mb = get_memory_mb();
            let cdp_curr = cdp_engine.get_frame_count();
            if cdp_curr >= cdp_last_count {
                cdp_fps = ((cdp_curr - cdp_last_count) as f32) / elapsed_secs;
            }
            cdp_last_count = cdp_curr;
            frame_count = 0;
            fps_timer = Instant::now();
        }

        let target_frame_time = if renderer.video_fast_path || renderer.scroll_fast_path || input_handler.is_typing_in_page {
            Duration::from_millis(16)
        } else {
            Duration::from_millis(33)
        };
        let rendered_cdp_count = cdp_engine.get_frame_count();
        let was_scroll_fast = renderer.scroll_fast_path;
        while frame_start.elapsed() < target_frame_time {
            if mode.is_cdp() && cdp_engine.get_frame_count() != rendered_cdp_count {
                break;
            }
            if was_scroll_fast && Instant::now() >= scroll_fast_until {
                break;
            }
            let remaining = target_frame_time.saturating_sub(frame_start.elapsed());
            std::thread::sleep(remaining.min(Duration::from_millis(2)));
        }
    }

    Ok(())
}

fn detect_smart_mode(url: &str, has_chromium: bool) -> RenderMode {
    if !has_chromium {
        return RenderMode::Native;
    }
    let u = url.to_lowercase();
    let is_text_dense = u.starts_with("about:")
        || u.ends_with(".txt")
        || u.ends_with(".md")
        || u.ends_with(".json")
        || u.ends_with(".csv")
        || u.ends_with(".log")
        || u.contains("news.ycombinator.com")
        || u.contains("lobste.rs")
        || u.contains("docs.rs")
        || u.contains("text.")
        || u.contains("lite.");

    if is_text_dense {
        RenderMode::Native
    } else {
        RenderMode::CdpHybrid
    }
}

fn url_base(url: &str) -> Result<String, ()> {
    if let Some(pos) = url.find("://") {
        let after_scheme = &url[pos + 3..];
        if let Some(slash_pos) = after_scheme.find('/') {
            Ok(url[..pos + 3 + slash_pos].to_string())
        } else {
            Ok(url.to_string())
        }
    } else {
        Err(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cohesive_block_wrap_with_inline_runs_and_citations() {
        let text = "India, officially the Republic of India,[20] is a country in South Asia.";
        let runs = vec![
            InlineStyleRun {
                start_char: 0,
                end_char: 5,
                fg: Some(Color::BLACK),
                bold: true,
                is_link: false,
                link_url: None,
            },
            InlineStyleRun {
                start_char: 22,
                end_char: 39,
                fg: Some(Color::BLACK),
                bold: true,
                is_link: false,
                link_url: None,
            },
            InlineStyleRun {
                start_char: 40,
                end_char: 44,
                fg: Some(Color::new(51, 102, 204)),
                bold: false,
                is_link: true,
                link_url: Some("https://en.wikipedia.org/wiki/India#cite_note-20".to_string()),
            },
            InlineStyleRun {
                start_char: 61,
                end_char: 71,
                fg: Some(Color::new(51, 102, 204)),
                bold: false,
                is_link: true,
                link_url: Some("https://en.wikipedia.org/wiki/South_Asia".to_string()),
            },
        ];
        let span = TextOverlaySpan {
            text: text.to_string(),
            px_x: 16,
            px_y: 120,
            px_w: 328,
            px_h: 48,
            fg: Color::BLACK,
            bg: Some(Color::WHITE),
            bold: false,
            is_link: false,
            is_input: false,
            link_url: None,
            is_fixed: false,
            is_popup: false,
            inline_runs: runs,
        };

        let wrapped = wrap_styled_overlay_lines(&span, 34, 8, 0);
        assert!(wrapped.len() >= 2 && wrapped.len() <= 4);

        let rejoined: String = wrapped
            .iter()
            .map(|l| l.segs.iter().map(|s| s.text.as_str()).collect::<String>())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(rejoined.contains("India, officially the"));
        assert!(rejoined.contains("India,[20]"));
        assert!(rejoined.contains("South Asia."));

        let all_segs: Vec<&StyledSeg> = wrapped.iter().flat_map(|l| l.segs.iter()).collect();
        assert!(all_segs.iter().any(|s| s.bold && s.text.contains("India")));
        assert!(all_segs.iter().any(|s| {
            s.underline
                && s.link_url.as_deref() == Some("https://en.wikipedia.org/wiki/South_Asia")
        }));
    }
}
