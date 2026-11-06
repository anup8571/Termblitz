use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageKey {
    Backspace,
    Enter,
    Tab,
    Left,
    Right,
    Delete,
    Home,
    End,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AppAction {
    Scroll(i32),
    PageScroll(i32),
    Navigate(String),
    TypeInPage(String),
    PageKey(PageKey),
    UnfocusPageInput,
    ToggleMode,
    ToggleAdBlock,
    ToggleMute,
    CycleUpscale,
    ToggleDither,
    TogglePixelArt,
    HistoryBack,
    HistoryForward,
    Reload,
    ZoomIn,
    ZoomOut,
    ZoomReset,
    Click(u16, u16),
    FocusUrlBar,
    Exit,
    None,
}

fn url_encode_query(q: &str) -> String {
    let mut out = String::with_capacity(q.len() * 2);
    for b in q.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            b' ' => out.push('+'),
            _ => {
                out.push('%');
                out.push(
                    char::from_digit((b >> 4) as u32, 16)
                        .unwrap_or('0')
                        .to_ascii_uppercase(),
                );
                out.push(
                    char::from_digit((b & 0x0F) as u32, 16)
                        .unwrap_or('0')
                        .to_ascii_uppercase(),
                );
            }
        }
    }
    out
}

/// Checks if a single-token string without scheme looks like a real browser hostname/URL
/// (e.g. `google.com`, `en.wikipedia.org/wiki/Rust`, `localhost:8080`, `127.0.0.1:3000`)
/// vs. normal search text or dotted terms (`rust 1.80`, `v1.2.3`, `node.js`, `main.rs`, `what.is`).
fn looks_like_direct_url(trimmed: &str) -> bool {
    if trimmed.is_empty() || trimmed.chars().any(|c| c.is_whitespace()) {
        return false;
    }

    // Split off path, query, or fragment to inspect the authority/host
    let authority = trimmed
        .split(&['/', '?', '#'][..])
        .next()
        .unwrap_or(trimmed);
    if authority.is_empty() {
        return false;
    }

    // Strip optional :port
    let (host, port_opt) = if let Some((h, p)) = authority.rsplit_once(':') {
        if p.chars().all(|c| c.is_ascii_digit()) && !p.is_empty() {
            (h, Some(p))
        } else {
            return false;
        }
    } else {
        (authority, None)
    };

    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }

    // Check valid IPv4 address (e.g. 127.0.0.1 or 192.168.1.1)
    let octets: Vec<&str> = host.split('.').collect();
    if octets.len() == 4 && octets.iter().all(|o| o.parse::<u8>().is_ok()) {
        return true;
    }

    // Hostname must have at least two labels separated by '.'
    if octets.len() < 2 {
        return false;
    }

    for label in &octets {
        if label.is_empty()
            || label.starts_with('-')
            || label.ends_with('-')
            || !label.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
        {
            return false;
        }
    }

    let tld = octets.last().copied().unwrap_or("").to_ascii_lowercase();
    // Common file/code extensions that users search for on Google rather than visiting as TLDs
    // (unless explicitly followed by a `/` path or `:port`)
    let has_explicit_path_or_port = trimmed.contains('/') || port_opt.is_some();
    if !has_explicit_path_or_port
        && matches!(
            tld.as_str(),
            "js" | "ts" | "rs" | "py" | "rb" | "go" | "c" | "cpp" | "h" | "java"
                | "kt" | "swift" | "css" | "html" | "json" | "toml" | "yaml" | "yml"
                | "md" | "txt" | "pdf" | "zip" | "tar" | "gz" | "exe" | "apk" | "so"
        )
    {
        return false;
    }

    const KNOWN_TLDS: &[&str] = &[
        "com", "org", "net", "edu", "gov", "mil", "int", "io", "co", "ai", "app", "dev", "me",
        "tv", "gg", "sh", "xyz", "info", "biz", "tech", "online", "site", "store", "blog",
        "cloud", "page", "link", "news", "live", "pro", "club", "design", "wiki", "fm", "to",
        "ly", "gl", "is", "it", "us", "uk", "ca", "au", "de", "fr", "in", "jp", "cn", "br",
        "ru", "es", "nl", "se", "no", "fi", "dk", "pl", "ch", "at", "be", "ie", "nz", "sg",
        "kr", "tw", "hk", "mx", "ar", "za", "id", "my", "ph", "th", "vn", "tr", "ua", "cz",
        "gr", "pt", "hu", "ro", "il", "ae", "sa", "eg", "pk", "bd", "ng", "ke", "asia", "mobi",
        "name", "museum", "coop", "aero", "xxx", "idv", "ovh", "one", "icu", "top", "vip",
        "work", "beer", "art", "fun", "space", "website", "press", "host", "world", "today",
        "digital", "media", "network", "systems", "solutions", "services", "company", "group",
        "studio", "agency", "center", "academy", "institute", "foundation", "gle", "youtube",
        "google",
    ];

    KNOWN_TLDS.contains(&tld.as_str()) || (has_explicit_path_or_port && tld.len() >= 2 && tld.chars().all(|c| c.is_ascii_alphabetic()))
}

/// Chrome-style Omnibox resolver:
/// - Explicit schemes (`https://`, `http://`, `about:`) or recognized domains/URLs navigate directly.
/// - Normal words, sentences, questions, or non-URL terms search on Google (`https://www.google.com/search?q=...`).
pub fn resolve_input_to_url(raw_input: &str) -> String {
    let trimmed = raw_input.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    if trimmed.starts_with("http://")
        || trimmed.starts_with("https://")
        || trimmed.starts_with("about:")
    {
        return trimmed.to_string();
    }
    if looks_like_direct_url(trimmed) {
        format!("https://{}", trimmed)
    } else {
        format!("https://www.google.com/search?q={}", url_encode_query(trimmed))
    }
}

pub struct InputHandler {
    pub is_entering_url: bool,
    pub is_typing_in_page: bool,
    pub url_input_buffer: String,
    pub last_pointer_pos: (u16, u16),
    pub sr_hit_range: Option<(u16, u16)>,
    pub dither_hit_range: Option<(u16, u16)>,
    pub mute_hit_range: Option<(u16, u16)>,
    drag_start: Option<(u16, u16)>,
    last_drag_y: Option<u16>,
    has_dragged: bool,
}

impl InputHandler {
    pub fn new() -> Self {
        Self {
            is_entering_url: false,
            is_typing_in_page: false,
            url_input_buffer: String::new(),
            last_pointer_pos: (40, 12),
            sr_hit_range: None,
            dither_hit_range: None,
            mute_hit_range: None,
            drag_start: None,
            last_drag_y: None,
            has_dragged: false,
        }
    }

    pub fn set_status_hit_zones(
        &mut self,
        sr_range: Option<(u16, u16)>,
        dither_range: Option<(u16, u16)>,
    ) {
        self.sr_hit_range = sr_range;
        self.dither_hit_range = dither_range;
    }

    pub fn set_all_status_hit_zones(
        &mut self,
        sr_range: Option<(u16, u16)>,
        dither_range: Option<(u16, u16)>,
        mute_range: Option<(u16, u16)>,
    ) {
        self.sr_hit_range = sr_range;
        self.dither_hit_range = dither_range;
        self.mute_hit_range = mute_range;
    }

    /// Resets any latched touch/mouse drag state so stale drag deltas can never loop.
    #[inline]
    pub fn reset_drag_state(&mut self) {
        self.drag_start = None;
        self.last_drag_y = None;
        self.has_dragged = false;
    }

    /// Polls for terminal input and drains all queued `crossterm::event::poll(Duration::ZERO)`
    /// events in the current frame tick. Coalesces and clamps scroll deltas into a single bounded
    /// scroll step per frame so scrolling stops the exact millisecond the user's finger or wheel stops.
    pub fn poll_action(&mut self, timeout: Duration) -> AppAction {
        let Ok(true) = event::poll(timeout) else {
            return AppAction::None;
        };
        let Ok(first_ev) = event::read() else {
            return AppAction::None;
        };

        let mut events = Vec::with_capacity(16);
        events.push(first_ev);

        // Drain all queued events with Duration::ZERO up to a safe cap
        let mut drained = 0usize;
        while drained < 128 {
            if let Ok(true) = event::poll(Duration::ZERO) {
                if let Ok(ev) = event::read() {
                    events.push(ev);
                    drained += 1;
                    continue;
                }
            }
            break;
        }

        self.process_event_batch(events)
    }

    /// Processes a batch of events from a single frame tick, coalescing scroll deltas and
    /// consecutive in-page typed characters while immediately returning high-priority actions.
    pub fn process_event_batch(&mut self, events: impl IntoIterator<Item = Event>) -> AppAction {
        let mut coalesced_scroll: i32 = 0;
        let mut has_scroll = false;
        let mut typed_buffer = String::new();
        let mut pending_action = AppAction::None;

        for ev in events {
            let act = self.handle_event(ev);
            match act {
                AppAction::None => {}
                AppAction::Scroll(delta) => {
                    has_scroll = true;
                    // Preserve intentional Home/End full-document jumps (-100000 / +100000)
                    if delta.abs() >= 10_000 {
                        coalesced_scroll = delta;
                    } else if coalesced_scroll.abs() < 10_000 {
                        coalesced_scroll = (coalesced_scroll + delta).clamp(-12, 12);
                    }
                }
                AppAction::TypeInPage(ch_str) => {
                    typed_buffer.push_str(&ch_str);
                }
                discrete => {
                    // Discrete action supersedes accumulated scroll in this frame and resets drag latching.
                    self.reset_drag_state();
                    if !typed_buffer.is_empty() && matches!(discrete, AppAction::PageKey(PageKey::Enter)) {
                        // Keep typed characters if user hit Enter right after typing in same batch
                        pending_action = discrete;
                        break;
                    }
                    pending_action = discrete;
                    if matches!(pending_action, AppAction::Exit | AppAction::Navigate(_)) {
                        return pending_action;
                    }
                }
            }
        }

        if !typed_buffer.is_empty() && pending_action == AppAction::None {
            AppAction::TypeInPage(typed_buffer)
        } else if pending_action != AppAction::None {
            pending_action
        } else if has_scroll && coalesced_scroll != 0 {
            AppAction::Scroll(coalesced_scroll)
        } else {
            AppAction::None
        }
    }

    fn handle_event(&mut self, ev: Event) -> AppAction {
        match ev {
            Event::Key(key) => self.handle_key(key),
            Event::Mouse(mouse) => self.handle_mouse(mouse),
            Event::Resize(_, _) => {
                self.reset_drag_state();
                AppAction::None
            }
            _ => AppAction::None,
        }
    }

    fn handle_key(&mut self, key: KeyEvent) -> AppAction {
        if key.modifiers.contains(KeyModifiers::CONTROL)
            && (key.code == KeyCode::Char('c') || key.code == KeyCode::Char('d'))
        {
            return AppAction::Exit;
        }

        // Alt+u, Alt+d, Alt+p, and Alt+m work globally, even when focused inside the URL bar or page input
        if key.modifiers.contains(KeyModifiers::ALT) {
            match key.code {
                KeyCode::Char('u') | KeyCode::Char('U') => return AppAction::CycleUpscale,
                KeyCode::Char('d') | KeyCode::Char('D') => return AppAction::ToggleDither,
                KeyCode::Char('p') | KeyCode::Char('P') => return AppAction::TogglePixelArt,
                KeyCode::Char('m') | KeyCode::Char('M') => return AppAction::ToggleMute,
                _ => {}
            }
        }

        if self.is_entering_url {
            match key.code {
                KeyCode::Enter => {
                    self.is_entering_url = false;
                    self.is_typing_in_page = false;
                    let raw = std::mem::take(&mut self.url_input_buffer);
                    let target = resolve_input_to_url(&raw);
                    if target.is_empty() {
                        AppAction::None
                    } else {
                        AppAction::Navigate(target)
                    }
                }
                KeyCode::Esc => {
                    self.is_entering_url = false;
                    self.url_input_buffer.clear();
                    AppAction::None
                }
                KeyCode::Backspace => {
                    self.url_input_buffer.pop();
                    AppAction::None
                }
                KeyCode::Char(c) => {
                    self.url_input_buffer.push(c);
                    AppAction::None
                }
                _ => AppAction::None,
            }
        } else if self.is_typing_in_page {
            match key.code {
                KeyCode::Esc => {
                    self.is_typing_in_page = false;
                    AppAction::UnfocusPageInput
                }
                KeyCode::Enter => {
                    self.is_typing_in_page = false;
                    AppAction::PageKey(PageKey::Enter)
                }
                KeyCode::Backspace => AppAction::PageKey(PageKey::Backspace),
                KeyCode::Delete => AppAction::PageKey(PageKey::Delete),
                KeyCode::Tab => AppAction::PageKey(PageKey::Tab),
                KeyCode::Left => AppAction::PageKey(PageKey::Left),
                KeyCode::Right => AppAction::PageKey(PageKey::Right),
                KeyCode::Home => AppAction::PageKey(PageKey::Home),
                KeyCode::End => AppAction::PageKey(PageKey::End),
                KeyCode::Up => {
                    self.reset_drag_state();
                    AppAction::Scroll(-1)
                }
                KeyCode::Down => {
                    self.reset_drag_state();
                    AppAction::Scroll(1)
                }
                KeyCode::Char(c) => AppAction::TypeInPage(c.to_string()),
                _ => AppAction::None,
            }
        } else {
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => AppAction::Exit,
                KeyCode::Char('u') | KeyCode::Char('U') => AppAction::CycleUpscale,
                KeyCode::Char('d') | KeyCode::Char('D') => AppAction::ToggleDither,
                KeyCode::Char('p') | KeyCode::Char('P') => AppAction::TogglePixelArt,
                KeyCode::Char('o') | KeyCode::Char('/') => {
                    self.is_entering_url = true;
                    self.is_typing_in_page = false;
                    self.url_input_buffer.clear();
                    AppAction::FocusUrlBar
                }
                KeyCode::Char('m') | KeyCode::F(2) => AppAction::ToggleMode,
                KeyCode::Char('M') => AppAction::ToggleMute,
                KeyCode::Char('a') => AppAction::ToggleAdBlock,
                KeyCode::Char('[') | KeyCode::Char('H') => AppAction::HistoryBack,
                KeyCode::Char(']') | KeyCode::Char('L') => AppAction::HistoryForward,
                KeyCode::Char('r') => AppAction::Reload,
                KeyCode::Char('+') | KeyCode::Char('=') => AppAction::ZoomIn,
                KeyCode::Char('-') | KeyCode::Char('_') => AppAction::ZoomOut,
                KeyCode::Char('0') => AppAction::ZoomReset,
                KeyCode::Char('j') | KeyCode::Down => {
                    self.reset_drag_state();
                    AppAction::Scroll(1)
                }
                KeyCode::Char('k') | KeyCode::Up => {
                    self.reset_drag_state();
                    AppAction::Scroll(-1)
                }
                KeyCode::PageDown | KeyCode::Char(' ') => {
                    self.reset_drag_state();
                    AppAction::PageScroll(1)
                }
                KeyCode::Char('b') | KeyCode::PageUp => {
                    self.reset_drag_state();
                    AppAction::PageScroll(-1)
                }
                KeyCode::Char('g') | KeyCode::Home => {
                    self.reset_drag_state();
                    AppAction::Scroll(-100000)
                }
                KeyCode::Char('G') | KeyCode::End => {
                    self.reset_drag_state();
                    AppAction::Scroll(100000)
                }
                _ => AppAction::None,
            }
        }
    }

    /// Universal SGR 1006 mouse and touch gesture handling (vertical drag scroll, horizontal swipe back/forward, top-bar touch)
    fn handle_mouse(&mut self, mouse: MouseEvent) -> AppAction {
        self.last_pointer_pos = (mouse.column, mouse.row);
        match mouse.kind {
            MouseEventKind::Down(MouseButton::Left) => {
                self.drag_start = Some((mouse.column, mouse.row));
                self.last_drag_y = Some(mouse.row);
                self.has_dragged = false;
                AppAction::None
            }
            MouseEventKind::Drag(MouseButton::Left) => {
                if let Some((sx, sy)) = self.drag_start {
                    let dx = (mouse.column as i32 - sx as i32).abs();
                    let dy = (mouse.row as i32 - sy as i32).abs();
                    if dx > 2 || dy > 1 {
                        self.has_dragged = true;
                    }
                }
                if let Some(last_y) = self.last_drag_y {
                    let diff = (last_y as i32) - (mouse.row as i32);
                    // Update anchor row immediately and clamp per-event drag step
                    self.last_drag_y = Some(mouse.row);
                    if diff != 0 {
                        return AppAction::Scroll(diff.clamp(-6, 6));
                    }
                }
                AppAction::None
            }
            MouseEventKind::Up(MouseButton::Left) => {
                let start = self.drag_start.take();
                let dragged = self.has_dragged;
                // Always clear drag state on touch/mouse release so touch_start never stays latched
                self.reset_drag_state();

                if let Some((sx, sy)) = start {
                    let dx = mouse.column as i32 - sx as i32;
                    let dy = mouse.row as i32 - sy as i32;

                    // Horizontal swipe gesture (Swipe Right = Back, Swipe Left = Forward)
                    if dragged && dx.abs() >= 10 && dx.abs() > dy.abs() * 2 {
                        self.is_typing_in_page = false;
                        if dx > 0 {
                            return AppAction::HistoryBack;
                        } else {
                            return AppAction::HistoryForward;
                        }
                    }

                    // Single tap (without drag)
                    if !dragged && dx.abs() <= 2 && dy.abs() <= 1 {
                        if mouse.row == 0 {
                            self.is_typing_in_page = false;
                            if mouse.column < 14 {
                                return AppAction::ToggleMode;
                            }
                            if let Some((sr_start, sr_end)) = self.sr_hit_range {
                                if mouse.column >= sr_start && mouse.column < sr_end {
                                    return AppAction::CycleUpscale;
                                }
                            }
                            if let Some((d_start, d_end)) = self.dither_hit_range {
                                if mouse.column >= d_start && mouse.column < d_end {
                                    return AppAction::ToggleDither;
                                }
                            }
                            if let Some((m_start, m_end)) = self.mute_hit_range {
                                if mouse.column >= m_start && mouse.column < m_end {
                                    return AppAction::ToggleMute;
                                }
                            }
                            self.is_entering_url = true;
                            self.url_input_buffer.clear();
                            return AppAction::FocusUrlBar;
                        } else {
                            self.is_entering_url = false;
                            return AppAction::Click(mouse.column, mouse.row);
                        }
                    }
                }
                AppAction::None
            }
            MouseEventKind::ScrollDown => {
                // Reset any active touch/drag state immediately after emitting ScrollDown
                self.reset_drag_state();
                AppAction::Scroll(3)
            }
            MouseEventKind::ScrollUp => {
                // Reset any active touch/drag state immediately after emitting ScrollUp
                self.reset_drag_state();
                AppAction::Scroll(-3)
            }
            MouseEventKind::Up(_) => {
                self.reset_drag_state();
                AppAction::None
            }
            _ => AppAction::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chrome_style_google_search_vs_url() {
        // Direct URLs and known TLDs
        assert_eq!(resolve_input_to_url("google.com"), "https://google.com");
        assert_eq!(
            resolve_input_to_url("https://en.wikipedia.org/wiki/Rust"),
            "https://en.wikipedia.org/wiki/Rust"
        );
        assert_eq!(
            resolve_input_to_url("localhost:3000"),
            "https://localhost:3000"
        );
        assert_eq!(
            resolve_input_to_url("127.0.0.1:8080/test"),
            "https://127.0.0.1:8080/test"
        );

        // Normal text, queries with spaces, code files, and dotted non-domains must search Google
        assert_eq!(
            resolve_input_to_url("rust terminal browser"),
            "https://www.google.com/search?q=rust+terminal+browser"
        );
        assert_eq!(
            resolve_input_to_url("youtube"),
            "https://www.google.com/search?q=youtube"
        );
        assert_eq!(
            resolve_input_to_url("node.js"),
            "https://www.google.com/search?q=node.js"
        );
        assert_eq!(
            resolve_input_to_url("main.rs"),
            "https://www.google.com/search?q=main.rs"
        );
        assert_eq!(
            resolve_input_to_url("v1.2.3"),
            "https://www.google.com/search?q=v1.2.3"
        );
    }

    #[test]
    fn test_in_page_input_typing_mode() {
        let mut handler = InputHandler::new();
        handler.is_typing_in_page = true;

        // Typing 'r', 'u', 's', 't' in a single batch coalesces into TypeInPage("rust")
        // instead of triggering Reload ('r') or CycleUpscale ('u')!
        let batch = vec![
            Event::Key(KeyEvent::new(KeyCode::Char('r'), KeyModifiers::NONE)),
            Event::Key(KeyEvent::new(KeyCode::Char('u'), KeyModifiers::NONE)),
            Event::Key(KeyEvent::new(KeyCode::Char('s'), KeyModifiers::NONE)),
            Event::Key(KeyEvent::new(KeyCode::Char('t'), KeyModifiers::NONE)),
        ];
        assert_eq!(
            handler.process_event_batch(batch),
            AppAction::TypeInPage("rust".to_string())
        );
        assert!(handler.is_typing_in_page);

        // Pressing Enter submits the page input and exits typing mode
        let enter_ev = vec![Event::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        ))];
        assert_eq!(
            handler.process_event_batch(enter_ev),
            AppAction::PageKey(PageKey::Enter)
        );
        assert!(!handler.is_typing_in_page);
    }

    #[test]
    fn test_scroll_coalescing_and_drag_unlatch() {
        let mut handler = InputHandler::new();

        // Simulate a burst of 15 wheel ScrollDown events in a single frame tick
        let mut burst = Vec::new();
        for _ in 0..15 {
            burst.push(Event::Mouse(MouseEvent {
                kind: MouseEventKind::ScrollDown,
                column: 10,
                row: 10,
                modifiers: KeyModifiers::NONE,
            }));
        }

        let action = handler.process_event_batch(burst);
        // Must be coalesced and clamped to a single bounded scroll step (+12)
        assert_eq!(action, AppAction::Scroll(12));
        assert!(handler.drag_start.is_none());
        assert!(handler.last_drag_y.is_none());

        // Simulate touch Down -> Drag -> Up and verify drag state is completely cleared on Up
        let touch_seq = vec![
            Event::Mouse(MouseEvent {
                kind: MouseEventKind::Down(MouseButton::Left),
                column: 20,
                row: 15,
                modifiers: KeyModifiers::NONE,
            }),
            Event::Mouse(MouseEvent {
                kind: MouseEventKind::Drag(MouseButton::Left),
                column: 20,
                row: 11,
                modifiers: KeyModifiers::NONE,
            }),
            Event::Mouse(MouseEvent {
                kind: MouseEventKind::Up(MouseButton::Left),
                column: 20,
                row: 11,
                modifiers: KeyModifiers::NONE,
            }),
        ];

        let drag_action = handler.process_event_batch(touch_seq);
        assert_eq!(drag_action, AppAction::Scroll(4));
        assert!(handler.drag_start.is_none(), "drag_start must be cleared after Up");
        assert!(handler.last_drag_y.is_none(), "last_drag_y must be cleared after Up");
        assert!(!handler.has_dragged, "has_dragged must be cleared after Up");
    }

    #[test]
    fn test_mute_hotkey_and_status_hit_zone() {
        let mut handler = InputHandler::new();
        handler.set_all_status_hit_zones(Some((20, 28)), Some((28, 40)), Some((41, 45)));

        // Alt+m toggles mute even while entering a URL
        handler.is_entering_url = true;
        let alt_m = Event::Key(KeyEvent::new(KeyCode::Char('m'), KeyModifiers::ALT));
        assert_eq!(handler.process_event_batch(vec![alt_m]), AppAction::ToggleMute);
        handler.is_entering_url = false;

        // Status bar tap on mute hit-zone (col 42, row 0) triggers ToggleMute
        let tap_mute = vec![
            Event::Mouse(MouseEvent {
                kind: MouseEventKind::Down(MouseButton::Left),
                column: 42,
                row: 0,
                modifiers: KeyModifiers::NONE,
            }),
            Event::Mouse(MouseEvent {
                kind: MouseEventKind::Up(MouseButton::Left),
                column: 42,
                row: 0,
                modifiers: KeyModifiers::NONE,
            }),
        ];
        assert_eq!(handler.process_event_batch(tap_mute), AppAction::ToggleMute);
    }

    #[test]
    fn test_pixel_art_hotkey() {
        let mut handler = InputHandler::new();
        let key_p = Event::Key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE));
        assert_eq!(
            handler.process_event_batch(vec![key_p]),
            AppAction::TogglePixelArt
        );

        handler.is_entering_url = true;
        let alt_p = Event::Key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::ALT));
        assert_eq!(
            handler.process_event_batch(vec![alt_p]),
            AppAction::TogglePixelArt
        );
    }
}
