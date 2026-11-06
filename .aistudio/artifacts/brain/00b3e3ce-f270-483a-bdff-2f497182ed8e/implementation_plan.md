# TermBlitz: Smooth Adaptive Scrolling, Direct Webpage Input Typing, Chrome-Style Google Search & Lag Optimization

This plan resolves the four core usability and performance bottlenecks in TermBlitz: responsive adaptive scrolling across all web interfaces, direct typing into webpage input boxes without hijacking focus to the top URL bar, Chrome-style Google Search fallback in the address bar, and background CPU/WebSocket optimizations that eliminate frame lag.

## User Review & Critical Decisions

> [!IMPORTANT]
> All three core UX and performance decisions have been confirmed and incorporated into the architecture below:

- **Confirmed Decision 1 — Webpage Input Box Typing**: Tapping an `<input>`, `<textarea>`, or search box inside a webpage focuses the element on the page and lets you type directly into the web box via CDP (`Esc` or tapping outside exits page-input mode; `/` or `o` or tapping row 0 opens the top URL/Search bar).
- **Confirmed Decision 2 — Adaptive Web-Interface Scrolling**: Scrolling dispatches both native CDP `mouseWheel` events at the pointer position and an adaptive DOM container scroller that detects nested scrollable containers (`overflow-y: auto/scroll`, YouTube mobile, modals, sidebars) before falling back to the document root.
- **Confirmed Decision 3 — Background Lag Elimination**: Combines dirty-frame cell caching (never re-running Super-Resolution or cloning `PixelBuffer` when the CDP frame hasn't changed) with a 60 FPS fast-scroll motion bypass that refines to full Super-Resolution the moment scrolling stops.

---

## 1. Overview & Core Concept

- **What It Does**:
  1. **Adaptive Smooth Scrolling**: Whether browsing a standard article, YouTube Mobile (`ytm-app`), a web app with nested scroll containers, or a modal dialog, scrolling adapts to the active container under the cursor/touch point with zero queue lag.
  2. **Direct In-Page Form & Search Box Typing**: Clicking any input field on a webpage focuses that DOM element and forwards keystrokes (`Char`, `Backspace`, `Space`, `Enter`, `Tab`, arrows) directly into the webpage rather than redirecting text to the top URL bar.
  3. **Chrome-Style Google Search Omnibox**: Entering normal words, phrases, or dotted terms that aren't valid domains/URLs in the top bar searches Google (`https://www.google.com/search?q=...`) instead of attempting broken `https://` navigation.
  4. **Zero-Lag Background Pipeline**: Eliminates redundant per-tick Super-Resolution recomputation, removes the 12ms WebSocket read stall, and throttles heavy DOM TreeWalker scans during active scrolling.
- **Target Audience / Persona**: Terminal power users and Android Termux / Linux users who want a fast, responsive browser experience with real page interaction.
- **Key Value**: Instantaneous input response, 60 FPS tear-free scrolling, and dramatic reduction in idle and active CPU usage.

---

## 2. User Experience & Interaction Flows

- **Flow 1 — Typing into a Webpage Input Box**:
  1. User taps or clicks a search bar or form field on a webpage (e.g., Google, YouTube, Wikipedia, GitHub).
  2. TermBlitz dispatches the click to Chromium to focus the input element and activates **Page Input Mode** (`[INPUT: typing on page | Esc to exit]` indicator in the top bar).
  3. Every keystroke is sent directly to the webpage's focused input box; pressing `Enter` submits the form or triggers the site's search handler.
  4. Pressing `Esc` or clicking outside an input box returns to normal navigation hotkeys.
- **Flow 2 — Chrome-Style Omnibox (URL vs. Google Search)**:
  1. User presses `/` or `o` (or taps the top bar) to open the Omnibox.
  2. If the user enters a valid URL or domain with a known TLD (e.g., `wikipedia.org`, `https://rust-lang.org`, `localhost:8080`), TermBlitz navigates directly to that URL.
  3. If the user enters natural text, multiple words, or non-domain dotted strings (e.g., `best rust tui libraries`, `python 3.12`, `what.is`), TermBlitz automatically queries Google Search.
- **Flow 3 — Smooth Adaptive Scrolling**:
  1. User scrolls via touch drag, mouse wheel, `j`/`k`, or `Space`/`PageDown`.
  2. During motion, TermBlitz switches to a 60 FPS fast-scroll path and scrolls the exact scrollable container under the pointer.
  3. Within 180ms of stopping, TermBlitz performs a single high-precision Super-Resolution pass and updates the cached frame buffer.

---

## 3. Key Product Decisions & Trade-Offs

- **Decision 1: Dirty-Frame Cell Cache + Motion Fast-Path**
  - *Chosen Approach*: Store the last rendered CDP `frame_id` and terminal cell snapshot inside `DiffRenderer`. Skip `get_frame()` cloning and Super-Resolution upscaling entirely when `frame_id` and renderer settings are unchanged. While scrolling is active (`< 180ms` since last scroll event), bypass heavy multi-pass neural convolution and refine once scrolling pauses.
  - *Why*: Previously, the main loop re-cloned the frame and re-ran full neural/FSR upscaling 30 times per second even on a static page, saturating the CPU and causing input lag.
- **Decision 2: Dual Native Wheel + DOM Scrollable Container Discovery**
  - *Chosen Approach*: Combine CDP `Input.dispatchMouseEvent` (`mouseWheel`) with an injected adaptive scroll function that checks `document.elementFromPoint(x, y)` for scrollable ancestors (`scrollHeight > clientHeight` with `overflow-y: auto | scroll | overlay`) or the largest scrollable container on screen.
  - *Why*: `window.scrollBy` fails on single-page apps and mobile layouts whose main feed lives inside a scrollable `<div>` or custom element.
- **Decision 3: Event-Driven Style Injection & Non-Blocking WebSocket Reads**
  - *Chosen Approach*: Lower WebSocket poll read timeout to `2ms` when idle and return immediately upon frame assembly; only inject `build_page_style_js` on navigation or mode changes instead of every 90ms.
  - *Why*: Eliminates unnecessary main-thread JS evaluation in Chromium and removes socket read stalls in the Rust worker thread.

---

## 4. Technical Architecture & Data Strategy

### Architecture & Event Flow Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        Terminal Input & Gesture Layer                      │
│  • Event Drain & Scroll Coalescer (tracks pointer col/row & drag delta)    │
│  • Mode Router:                                                            │
│     ├─ Top Omnibox Mode (`/`, `o`, row 0 tap) -> Chrome-style URL/Google   │
│     ├─ Page Input Mode (tap on <input>/<textarea>) -> Direct CDP Keystrokes│
│     └─ Browser Command Mode -> Scroll, Click, Zoom, SR/Dither/PixelArt     │
└───────────────────┬────────────────────────────────────┬───────────────────┘
                    │                                    │
                    ▼                                    ▼
┌───────────────────────────────────────┐  ┌─────────────────────────────────┐
│      Non-Blocking CDP Worker          │  │   Cached Super-Res Renderer     │
│  • 2ms Socket Poll (no 12ms stall)    │  │  • Frame-ID Dirty Check (O(1)   │
│  • Adaptive Scroll Engine:            │  │    skip when frame unchanged)   │
│    - CDP mouseWheel at pointer (x,y)  │  │  • 60 FPS Motion Fast-Path      │
│    - DOM scrollable ancestor lookup   │  │    during active scroll bursts  │
│  • Direct DOM Input Dispatcher:       │  │  • Full SR + RCAS refinement    │
│    - Input.insertText / KeyEvents     │  │    180ms after scroll settles   │
│    - Enter -> form submit / keydown   │  │  • DEC Mode 2026 Sync Output    │
└───────────────────────────────────────┘  └─────────────────────────────────┘
```

### Interactive State & Handler Mapping

1. **Omnibox URL vs. Google Search Classifier (`resolve_input_to_url`)**:
   - Validates explicit schemes (`http://`, `https://`, `about:`), `localhost`, IPv4 addresses, and single-token hostnames with valid TLDs.
   - Routes all other queries (spaces, questions, numeric versions like `1.20`, unknown dotted strings) to `https://www.google.com/search?q=<urlencoded>`.
2. **In-Page Input Focus & Typing (`is_typing_in_page`)**:
   - Clicking a webpage input field sends `CdpWorkerCommand::FocusAndClickInput(col, row)` which clicks and focuses the underlying `<input>`/`<textarea>`/`[contenteditable]` element.
   - Keystrokes emit `AppAction::TypeInPage(String)` (`Input.insertText` + input event), `AppAction::PageKey(PageKeyAction::Backspace | Enter | Tab | Left | Right | Escape)`.
   - `Enter` dispatches `keydown`/`keypress`/`keyup` (`Enter`, keyCode 13) and triggers form submission if inside a `<form>`, automatically unfocusing `is_typing_in_page` when navigation starts.
3. **Adaptive Scroll Container Engine (`CdpWorkerCommand::ScrollAt`)**:
   - Passes `(delta, pointer_col, pointer_row)` to the CDP worker.
   - Executes native `Input.dispatchMouseEvent` (`type: "mouseWheel"`) and adaptive container scroll in JS that locates the active scrollable element at `(px_x, px_y)` or the primary scrollable feed container, scrolling it instantaneously and reporting accurate `scroll_pct`.
4. **Dirty-Frame Caching & Background Optimization**:
   - Tracks `last_rendered_frame_id`, `last_rendered_mode`, and `scroll_fast_until` in the main loop and `DiffRenderer`.
   - Avoids cloning `PixelBuffer` or running `blit_frame` when no new frame or state change has occurred, and pauses DOM `TreeWalker` extraction during active scroll frames until scrolling settles.
