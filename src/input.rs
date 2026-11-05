use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use std::time::Duration;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AppAction {
    Scroll(i32),
    PageScroll(i32),
    Navigate(String),
    ToggleMode,
    Reload,
    ZoomIn,
    ZoomOut,
    ZoomReset,
    Click(u16, u16),
    FocusUrlBar,
    Exit,
    None,
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
        // Universal Ctrl+C / Ctrl+D exit
        if key.modifiers.contains(KeyModifiers::CONTROL)
            && (key.code == KeyCode::Char('c') || key.code == KeyCode::Char('d'))
        {
            return AppAction::Exit;
        }

        if self.is_entering_url {
            match key.code {
                KeyCode::Enter => {
                    self.is_entering_url = false;
                    let url = std::mem::take(&mut self.url_input_buffer);
                    if url.is_empty() {
                        AppAction::None
                    } else if !url.starts_with("http://")
                        && !url.starts_with("https://")
                        && !url.starts_with("about:")
                    {
                        AppAction::Navigate(format!("https://{}", url))
                    } else {
                        AppAction::Navigate(url)
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
                KeyCode::Char('u') | KeyCode::Char('o') => {
                    self.is_entering_url = true;
                    self.url_input_buffer.clear();
                    AppAction::FocusUrlBar
                }
                KeyCode::Char('m') | KeyCode::F(2) => AppAction::ToggleMode,
                KeyCode::Char('r') => AppAction::Reload,
                KeyCode::Char('+') | KeyCode::Char('=') => AppAction::ZoomIn,
                KeyCode::Char('-') | KeyCode::Char('_') => AppAction::ZoomOut,
                KeyCode::Char('0') => AppAction::ZoomReset,
                KeyCode::Char('j') | KeyCode::Down => AppAction::Scroll(1),
                KeyCode::Char('k') | KeyCode::Up => AppAction::Scroll(-1),
                KeyCode::Char('d') | KeyCode::PageDown | KeyCode::Char(' ') => AppAction::PageScroll(1),
                KeyCode::Char('b') | KeyCode::PageUp => AppAction::PageScroll(-1),
                KeyCode::Char('g') | KeyCode::Home => AppAction::Scroll(-100000), // Jump to top
                KeyCode::Char('G') | KeyCode::End => AppAction::Scroll(100000),   // Jump to bottom
                _ => AppAction::None,
            }
        }
    }

    /// Universal SGR 1006 mouse and touch event handling (desktop mouse + mobile touch drag/swipe)
    fn handle_mouse(&mut self, mouse: MouseEvent) -> AppAction {
        match mouse.kind {
            MouseEventKind::Down(MouseButton::Left) => {
                self.drag_start = Some((mouse.column, mouse.row));
                self.last_drag_y = Some(mouse.row);
                self.has_dragged = false;
                AppAction::None
            }
            MouseEventKind::Drag(MouseButton::Left) => {
                if let Some(last_y) = self.last_drag_y {
                    let diff = (last_y as i32) - (mouse.row as i32);
                    if diff != 0 {
                        self.has_dragged = true;
                        self.last_drag_y = Some(mouse.row);
                        return AppAction::Scroll(diff);
                    }
                }
                AppAction::None
            }
            MouseEventKind::Up(MouseButton::Left) => {
                let start = self.drag_start.take();
                self.last_drag_y = None;

                // If user tapped without dragging, trigger Click
                if !self.has_dragged {
                    if let Some((sx, sy)) = start {
                        let dist_x = (mouse.column as i32 - sx as i32).abs();
                        let dist_y = (mouse.row as i32 - sy as i32).abs();
                        if dist_x <= 1 && dist_y <= 1 {
                            if mouse.row == 0 {
                                self.is_entering_url = true;
                                self.url_input_buffer.clear();
                                return AppAction::FocusUrlBar;
                            } else {
                                return AppAction::Click(mouse.column, mouse.row);
                            }
                        }
                    }
                }
                self.has_dragged = false;
                AppAction::None
            }
            MouseEventKind::ScrollDown => AppAction::Scroll(3),
            MouseEventKind::ScrollUp => AppAction::Scroll(-3),
            _ => AppAction::None,
        }
    }
}
