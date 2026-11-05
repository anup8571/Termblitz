use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use std::time::Duration;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AppAction {
    Scroll(i32),
    PageScroll(i32),
    Navigate(String),
    ToggleMode,
    ToggleAdBlock,
    CycleUpscale,
    ToggleDither,
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
                out.push( char::from_digit((b >> 4) as u32, 16).unwrap_or('0').to_ascii_uppercase() );
                out.push( char::from_digit((b & 0x0F) as u32, 16).unwrap_or('0').to_ascii_uppercase() );
            }
        }
    }
    out
}

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
    // Treat as search query if it contains spaces or has no domain dot (unless localhost)
    if trimmed.contains(' ') || (!trimmed.contains('.') && !trimmed.starts_with("localhost")) {
        format!("https://www.google.com/search?q={}", url_encode_query(trimmed))
    } else {
        format!("https://{}", trimmed)
    }
}

pub struct InputHandler {
    pub is_entering_url: bool,
    pub url_input_buffer: String,
    drag_start: Option<(u16, u16)>,
    last_drag_y: Option<u16>,
    has_dragged: bool,
}

impl InputHandler {
    pub fn new() -> Self {
        Self {
            is_entering_url: false,
            url_input_buffer: String::new(),
            drag_start: None,
            last_drag_y: None,
            has_dragged: false,
        }
    }

    pub fn poll_action(&mut self, timeout: Duration) -> AppAction {
        if let Ok(true) = event::poll(timeout) {
            if let Ok(ev) = event::read() {
                return self.handle_event(ev);
            }
        }
        AppAction::None
    }

    fn handle_event(&mut self, ev: Event) -> AppAction {
        match ev {
            Event::Key(key) => self.handle_key(key),
            Event::Mouse(mouse) => self.handle_mouse(mouse),
            _ => AppAction::None,
        }
    }

    fn handle_key(&mut self, key: KeyEvent) -> AppAction {
        if key.modifiers.contains(KeyModifiers::CONTROL)
            && (key.code == KeyCode::Char('c') || key.code == KeyCode::Char('d'))
        {
            return AppAction::Exit;
        }

        // Alt+u and Alt+d work globally, even when focused inside the URL/search input bar
        if key.modifiers.contains(KeyModifiers::ALT) {
            match key.code {
                KeyCode::Char('u') | KeyCode::Char('U') => return AppAction::CycleUpscale,
                KeyCode::Char('d') | KeyCode::Char('D') => return AppAction::ToggleDither,
                _ => {}
            }
        }

        if self.is_entering_url {
            match key.code {
                KeyCode::Enter => {
                    self.is_entering_url = false;
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
        } else {
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => AppAction::Exit,
                KeyCode::Char('u') | KeyCode::Char('U') => AppAction::CycleUpscale,
                KeyCode::Char('d') | KeyCode::Char('D') => AppAction::ToggleDither,
                KeyCode::Char('o') | KeyCode::Char('/') => {
                    self.is_entering_url = true;
                    self.url_input_buffer.clear();
                    AppAction::FocusUrlBar
                }
                KeyCode::Char('m') | KeyCode::F(2) => AppAction::ToggleMode,
                KeyCode::Char('a') => AppAction::ToggleAdBlock,
                KeyCode::Char('[') | KeyCode::Char('H') => AppAction::HistoryBack,
                KeyCode::Char(']') | KeyCode::Char('L') => AppAction::HistoryForward,
                KeyCode::Char('r') => AppAction::Reload,
                KeyCode::Char('+') | KeyCode::Char('=') => AppAction::ZoomIn,
                KeyCode::Char('-') | KeyCode::Char('_') => AppAction::ZoomOut,
                KeyCode::Char('0') => AppAction::ZoomReset,
                KeyCode::Char('j') | KeyCode::Down => AppAction::Scroll(1),
                KeyCode::Char('k') | KeyCode::Up => AppAction::Scroll(-1),
                KeyCode::PageDown | KeyCode::Char(' ') => AppAction::PageScroll(1),
                KeyCode::Char('b') | KeyCode::PageUp => AppAction::PageScroll(-1),
                KeyCode::Char('g') | KeyCode::Home => AppAction::Scroll(-100000),
                KeyCode::Char('G') | KeyCode::End => AppAction::Scroll(100000),
                _ => AppAction::None,
            }
        }
    }

    /// Universal SGR 1006 mouse and touch gesture handling (vertical drag scroll, horizontal swipe back/forward, top-bar touch)
    fn handle_mouse(&mut self, mouse: MouseEvent) -> AppAction {
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
                    if diff != 0 {
                        self.last_drag_y = Some(mouse.row);
                        return AppAction::Scroll(diff);
                    }
                }
                AppAction::None
            }
            MouseEventKind::Up(MouseButton::Left) => {
                let start = self.drag_start.take();
                self.last_drag_y = None;
                let dragged = self.has_dragged;
                self.has_dragged = false;

                if let Some((sx, sy)) = start {
                    let dx = mouse.column as i32 - sx as i32;
                    let dy = mouse.row as i32 - sy as i32;

                    // Horizontal swipe gesture (Swipe Right = Back, Swipe Left = Forward)
                    if dragged && dx.abs() >= 10 && dx.abs() > dy.abs() * 2 {
                        if dx > 0 {
                            return AppAction::HistoryBack;
                        } else {
                            return AppAction::HistoryForward;
                        }
                    }

                    // Single tap (without drag)
                    if !dragged && dx.abs() <= 2 && dy.abs() <= 1 {
                        if mouse.row == 0 {
                            if mouse.column < 14 {
                                return AppAction::ToggleMode;
                            } else {
                                self.is_entering_url = true;
                                self.url_input_buffer.clear();
                                return AppAction::FocusUrlBar;
                            }
                        } else {
                            return AppAction::Click(mouse.column, mouse.row);
                        }
                    }
                }
                AppAction::None
            }
            MouseEventKind::ScrollDown => AppAction::Scroll(3),
            MouseEventKind::ScrollUp => AppAction::Scroll(-3),
            _ => AppAction::None,
        }
    }
}
