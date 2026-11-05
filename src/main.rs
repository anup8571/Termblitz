mod renderer;
mod engine;
mod input;

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

use engine::{CdpEngine, NativeEngine, RenderMode};
use input::{resolve_input_to_url, AppAction, InputHandler};
use renderer::{Color, DiffRenderer, GraphicsProtocol};

struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let mut out = stdout();
        let _ = out.write_all(b"\x1b[?1006l\x1b[?1002l\x1b[?1000l\x1b_Ga=d,d=A\x1b[0m\x1b[2J\x1b[H");
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

/// Word-wraps multi-line DOM text overlays within their bounding box column width (`max_width`)
/// across up to `max_lines` rows so cards, snippets, and side-by-side columns never truncate or collide.
fn wrap_overlay_text(text: &str, max_width: usize, max_lines: usize) -> Vec<String> {
    let max_w = max_width.max(1);
    if max_lines <= 1 || text.chars().count() <= max_w {
        return vec![text.chars().take(max_w).collect()];
    }

    let mut lines = Vec::with_capacity(max_lines.min(8));
    let mut current = String::new();
    let mut current_len = 0usize;

    for word in text.split_whitespace() {
        let word_len = word.chars().count();
        if current_len == 0 {
            if word_len <= max_w {
                current.push_str(word);
                current_len = word_len;
            } else {
                let clipped: String = word.chars().take(max_w).collect();
                lines.push(clipped);
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
                current.push_str(&word.chars().take(max_w).collect::<String>());
                current_len = max_w;
            }
        }
    }

    if !current.is_empty() && lines.len() < max_lines {
        lines.push(current);
    }

    if lines.is_empty() {
        lines.push(text.chars().take(max_w).collect());
    }
    lines
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
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

    enable_raw_mode()?;
    let mut out = stdout();
    execute!(out, EnterAlternateScreen, EnableMouseCapture, Hide)?;
    let _ = out.write_all(b"\x1b[2J\x1b[H\x1b[?1000h\x1b[?1002h\x1b[?1006h");
    let _ = out.flush();
    let _guard = TerminalGuard;

    let (mut cols, mut rows) = crossterm::terminal::size().unwrap_or((80, 24));
    let mut renderer = DiffRenderer::new(cols, rows);

    if enable_kitty {
        renderer.protocol = GraphicsProtocol::Kitty;
    } else if enable_sixel {
        renderer.protocol = GraphicsProtocol::Sixel;
    } else if enable_ansi {
        renderer.protocol = GraphicsProtocol::HalfBlockAnsi;
    }

    let mut native_engine = NativeEngine::new();
    let mut cdp_engine = CdpEngine::new(9222);
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

    loop {
        let frame_start = Instant::now();

        if let Ok((w, h)) = crossterm::terminal::size() {
            if w != cols || h != rows {
                cols = w;
                rows = h;
                renderer.resize(cols, rows);
                native_engine.parse_and_layout(cols);
                if mode.is_cdp() {
                    cdp_engine.resize(cols, rows);
                }
            }
        }

        // Sync live URL from CDP when navigating via links or search inside headless Chromium
        if mode.is_cdp() {
            let live = cdp_engine.get_live_url();
            if !live.is_empty() && live != "about:blank" && live != native_engine.current_url {
                native_engine.current_url = live.clone();
                cdp_engine.current_url = live;
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
                            cdp_engine.navigate(&native_engine.current_url, cols, rows);
                            status_banner = Some((
                                "Hybrid Mode: Crisp DOM text + visual background".to_string(),
                                Instant::now() + Duration::from_secs(2),
                            ));
                        }
                        (RenderMode::CdpHybrid, RenderMode::CdpRaw) => {
                            cdp_engine.set_hide_text(false);
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
                        "🛡️ Brave Ad & Tracker Shield: ENABLED".to_string()
                    } else {
                        "🛡️ Brave Ad & Tracker Shield: DISABLED".to_string()
                    },
                    Instant::now() + Duration::from_secs(2),
                ));
            }
            AppAction::HistoryBack => {
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
                    cdp_engine.navigate(&resolved, cols, rows);
                }
            }
            AppAction::Click(click_x, click_y) => {
                if mode == RenderMode::Native {
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
                            cdp_engine.navigate(&resolved, cols, rows);
                        } else {
                            native_engine.load_url(&resolved, cols);
                        }
                    }
                } else {
                    let (hit_link, hit_input) = cdp_engine.hit_test_overlay(click_x, click_y, cols, rows);
                    if hit_input {
                        // Tapping a search bar or input field on the page opens the search/URL input bar!
                        input_handler.is_entering_url = true;
                        input_handler.url_input_buffer.clear();
                    } else if let Some(url) = hit_link {
                        if !url.starts_with("javascript:") && !url.is_empty() {
                            native_engine.current_url = url.clone();
                            cdp_engine.navigate(&url, cols, rows);
                        } else {
                            cdp_engine.click(click_x, click_y);
                        }
                    } else {
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
                    cdp_engine.scroll(delta);
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
                    cdp_engine.scroll(delta * (rows as i32 / 2));
                }
            }
            AppAction::Reload => {
                renderer.force_redraw();
                let curr = native_engine.current_url.clone();
                if mode == RenderMode::Native {
                    native_engine.load_url(&curr, cols);
                } else {
                    cdp_engine.navigate(&curr, cols, rows);
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
            RenderMode::CdpHybrid | RenderMode::CdpRaw => Color::WHITE,
        };
        renderer.clear(clear_bg);

        let view_start_y = 1u16;
        let view_max_y = rows;

        match mode {
            RenderMode::Native => {
                native_engine.render_to(&mut renderer, view_start_y, view_max_y);
            }
            RenderMode::CdpHybrid | RenderMode::CdpRaw => {
                let cdp_err = cdp_engine.get_error();
                if let Some(ref err) = cdp_err {
                    let box_w = (cols.saturating_sub(6)).min(70);
                    let start_x = (cols.saturating_sub(box_w)) / 2;
                    let start_y = (rows.saturating_sub(10)) / 2;

                    let err_bg = Color::new(45, 15, 15);
                    let err_fg = Color::WHITE;
                    let err_border = Color::RED;

                    let horiz = "═".repeat((box_w.saturating_sub(2)) as usize);
                    let top_border = format!("╔{}╗", horiz);
                    let bot_border = format!("╚{}╝", horiz);

                    renderer.put_str(start_x, start_y, &top_border, err_border, err_bg, true, false);
                    renderer.put_str(start_x, start_y + 1, "║ ⚠️  CDP-WEB Engine Notice", err_fg, err_bg, true, false);
                    renderer.put_str(start_x + box_w - 1, start_y + 1, "║", err_border, err_bg, true, false);

                    let mid_border = format!("╠{}╣", horiz);
                    renderer.put_str(start_x, start_y + 2, &mid_border, err_border, err_bg, false, false);

                    let lines: Vec<&str> = err.lines().collect();
                    for (idx, eline) in lines.iter().take(4).enumerate() {
                        let y = start_y + 3 + (idx as u16);
                        let pad = " ".repeat((box_w as usize).saturating_sub(eline.chars().count() + 4));
                        let content = format!("║  {}{}", eline, pad);
                        renderer.put_str(start_x, y, &content, Color::LIGHT_GRAY, err_bg, false, false);
                        renderer.put_str(start_x + box_w - 1, y, "║", err_border, err_bg, false, false);
                    }

                    let prompt_y = start_y + 7;
                    renderer.put_str(start_x, prompt_y, "║  • Press [m] to switch to Native Mode", Color::YELLOW, err_bg, true, false);
                    renderer.put_str(start_x + box_w - 1, prompt_y, "║", err_border, err_bg, false, false);

                    let prompt_y2 = start_y + 8;
                    renderer.put_str(start_x, prompt_y2, "║  • Press [r] to retry Chromium", Color::CYAN, err_bg, false, false);
                    renderer.put_str(start_x + box_w - 1, prompt_y2, "║", err_border, err_bg, false, false);

                    renderer.put_str(start_x, start_y + 9, &bot_border, err_border, err_bg, true, false);
                } else {
                    let px_buf = cdp_engine.get_frame();
                    let is_loading = cdp_engine.is_loading();
                    let count = cdp_engine.get_frame_count();

                    if count == 0 || (is_loading && count < 2) {
                        let msg = " 🌐 Connecting & Loading Visual Web Page... ⏳ ";
                        let msg_x = cols.saturating_sub(msg.len() as u16) / 2;
                        let msg_y = rows / 2;
                        renderer.put_str(msg_x, msg_y, msg, Color::CYAN, Color::new(28, 32, 42), true, false);
                    } else if px_buf.width > 0 && px_buf.height > 0 {
                        // 1. Blit visual background frame (cards, buttons, logos, images)
                        renderer.blit_frame(&px_buf, view_start_y, view_max_y);

                        // 2. In Hybrid Mode: Overlay razor-sharp native terminal characters at 1:1 CSS viewport coordinates
                        if mode == RenderMode::CdpHybrid {
                            let overlays = cdp_engine.get_text_overlays();
                            let (vw, vh) = cdp_engine.get_viewport_dims();
                            let pw = vw.max(1) as f32;
                            let ph = vh.max(1) as f32;
                            let view_rows = (view_max_y.saturating_sub(view_start_y)).max(1) as f32;

                            for span in &overlays {
                                let start_col = ((span.px_x.max(0) as f32 * cols as f32) / pw)
                                    .round()
                                    .min(cols.saturating_sub(1) as f32) as u16;
                                let term_row = view_start_y
                                    + ((span.px_y.max(0) as f32 * view_rows) / ph).round() as u16;
                                let span_cols = ((span.px_w.max(8) as f32 * cols as f32) / pw)
                                    .round()
                                    .max(1.0) as usize;
                                let box_rows = ((span.px_h.max(12) as f32 * view_rows) / ph)
                                    .ceil()
                                    .max(1.0) as usize;

                                if term_row >= view_start_y && term_row < view_max_y && start_col < cols {
                                    let available_cols = (cols - start_col) as usize;
                                    let is_multiline = box_rows >= 2 && span.px_h >= 24;
                                    let wrap_w = if is_multiline {
                                        span_cols.min(available_cols).max(8)
                                    } else {
                                        span_cols.max(span.text.chars().count()).min(available_cols)
                                    };
                                    let max_lines = if is_multiline { box_rows.min(6) } else { 1 };
                                    let wrapped_lines = wrap_overlay_text(&span.text, wrap_w, max_lines);

                                    let link_fg = if span.is_input {
                                        Color::new(30, 80, 190)
                                    } else if span.is_link && span.fg.luma_fast() < 40 {
                                        Color::new(26, 86, 219)
                                    } else {
                                        span.fg
                                    };

                                    let fallback_bg = span.bg.unwrap_or(Color::WHITE);
                                    for (line_offset, line_text) in wrapped_lines.iter().enumerate() {
                                        let target_y = term_row + (line_offset as u16);
                                        if target_y >= view_max_y {
                                            break;
                                        }
                                        renderer.put_str_with_existing_bg(
                                            start_col,
                                            target_y,
                                            line_text,
                                            link_fg,
                                            fallback_bg,
                                            span.bold,
                                            span.is_link,
                                        );
                                    }
                                }
                            }
                        }

                        if is_loading {
                            let badge = " ⏳ Updating... ";
                            let bx = cols.saturating_sub(badge.len() as u16 + 2);
                            renderer.put_str(bx, 1, badge, Color::YELLOW, Color::new(28, 32, 42), true, false);
                        }
                    }
                }
            }
        }

        // Top 1-line Status & Address bar
        let bar_bg = Color::new(28, 32, 42);
        let bar_fg = Color::WHITE;
        let mode_badge = match mode {
            RenderMode::Native => (" NATIVE ".to_string(), Color::BLACK, Color::GREEN),
            RenderMode::CdpHybrid => {
                if cdp_engine.get_error().is_some() {
                    (" CDP-ERR ".to_string(), Color::WHITE, Color::RED)
                } else {
                    (format!(" HYBRID [{:.0}%] ", cdp_engine.zoom * 100.0), Color::BLACK, Color::new(255, 180, 0))
                }
            }
            RenderMode::CdpRaw => {
                if cdp_engine.get_error().is_some() {
                    (" CDP-ERR ".to_string(), Color::WHITE, Color::RED)
                } else {
                    (format!(" RAW [{:.0}%] ", cdp_engine.zoom * 100.0), Color::WHITE, Color::new(140, 80, 220))
                }
            }
        };

        renderer.put_str(0, 0, &" ".repeat(cols as usize), bar_fg, bar_bg, false, false);
        renderer.put_str(0, 0, &mode_badge.0, mode_badge.1, mode_badge.2, true, false);

        let addr_x = (mode_badge.0.len() + 1) as u16;
        let has_active_banner = if let Some((ref msg, expiry)) = status_banner {
            if Instant::now() < expiry {
                let banner = format!(" {} ", msg);
                renderer.put_str(addr_x, 0, &banner, Color::YELLOW, Color::new(50, 40, 20), true, false);
                true
            } else {
                false
            }
        } else {
            false
        };

        if !has_active_banner {
            if input_handler.is_entering_url {
                let prompt = format!("Search / URL: {}█", input_handler.url_input_buffer);
                renderer.put_str(addr_x, 0, &prompt, Color::CYAN, bar_bg, true, false);
            } else {
                let url_display = format!(" {} ", native_engine.current_url);
                renderer.put_str(addr_x, 0, &url_display, Color::WHITE, bar_bg, false, false);
            }
        }

        let mem = get_memory_mb();
        let scroll_pct = if mode.is_cdp() {
            cdp_engine.get_scroll_pct()
        } else if native_engine.max_scroll > 0 {
            (native_engine.scroll_y * 100) / native_engine.max_scroll
        } else {
            100
        };
        let shield_tag = if cdp_engine.adblock_enabled { "SHIELD" } else { "NO-AD" };
        let fps_display = if mode.is_cdp() {
            if cdp_fps > 0.0 { format!("{:0.0}fps", cdp_fps) } else { "idle".to_string() }
        } else {
            format!("{:0.0}fps", fps)
        };
        let metrics = format!(" {} | {} | {:0.1}MB | {} | {}% ", shield_tag, renderer.protocol.short_name(), mem, fps_display, scroll_pct);
        let metrics_x = cols.saturating_sub(metrics.len() as u16);
        renderer.put_str(metrics_x, 0, &metrics, Color::LIGHT_GRAY, bar_bg, false, false);

        let _ = renderer.flush();

        frame_count += 1;
        let elapsed_secs = fps_timer.elapsed().as_secs_f32();
        if elapsed_secs >= 0.5 {
            fps = (frame_count as f32) / elapsed_secs;
            let cdp_curr = cdp_engine.get_frame_count();
            if cdp_curr >= cdp_last_count {
                cdp_fps = ((cdp_curr - cdp_last_count) as f32) / elapsed_secs;
            }
            cdp_last_count = cdp_curr;
            frame_count = 0;
            fps_timer = Instant::now();
        }

        let elapsed = frame_start.elapsed();
        let target_frame_time = Duration::from_millis(33);
        if elapsed < target_frame_time {
            std::thread::sleep(target_frame_time - elapsed);
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
