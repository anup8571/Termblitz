# ⚡ Termblitz

> **Blazing-Fast Visual Web Browser for Android Termux & Cross-Platform Terminals**  
> Unified 3-Mode Architecture: Pure-Rust Native Parser (<1MB RAM) & Headless Chromium CDP with 1:1 Razor-Sharp DOM Overlay & Brave-Style Ad Blocker.

Termblitz is a lightweight, mobile-optimized terminal browser crafted specifically for portrait screens on Android (Termux), Linux, macOS, and Windows. It combines a zero-overhead pure-Rust HTML layout engine with an advanced Headless Chromium hybrid renderer.

---

## 🚀 Key Features

* **Unified 3-Mode Rendering Architecture**:
  * 🟢 **Native Mode (`NATIVE`)**: Ultra-fast pure-Rust HTML/DOM parser and table layout engine. Zero Chromium overhead, **< 1MB RAM**, instant startup, 0% idle CPU.
  * 🟡 **CDP Hybrid Mode (`HYBRID [100%]`)**: Headless Chromium with 1:1 aspect-ratio-locked mobile portrait reflow (360px). Hides rasterized glyph blur via `-webkit-text-fill-color: transparent` while preserving true CSS computed text colors, overlaying DOM text and `<input>` fields as **razor-sharp native terminal characters** with multi-line word wrapping and guaranteed luminance contrast.
  * 🟣 **CDP Raw Mode (`RAW [100%]`)**: Full ANSI half-block / Kitty / Sixel raster bitmap rendering for viewing full-fidelity canvases, WebGL, or raw images without overlays.
* **🛡️ Integrated Brave-Style Ad & Tracker Shield (`SHIELD`)**:
  * Blocks ad networks, telemetry, and trackers at the Chromium network layer (`Network.setBlockedURLs`) plus cosmetic CSS ad-element hiding and Native HTML link filtering.
* **Touch Gestures & Hit-Testing**:
  * **Tap** links, buttons, or search `<input>` boxes directly on your phone screen.
  * **Swipe Up / Down** to scroll smoothly.
  * **Swipe Right / Left** for Back / Forward history navigation.
  * **Tap Top-Left Mode Badge** to cycle rendering modes by touch; **Tap Top Bar** to enter a URL or search query.

---

## ⌨️ Controls & Navigation

| Key / Gesture | Action |
| :--- | :--- |
| `m` / `F2` / **Tap Badge** | **Cycle Rendering Mode** (`Native` → `Hybrid` → `Raw` → `Native`) |
| `u` / `o` / `/` / **Tap Bar** | **Focus Search / Address Bar** (enter any URL or Google search term) |
| `a` | **Toggle Brave Ad & Tracker Shield** (`SHIELD` / `NO-AD`) |
| `[` / `]` or **Swipe R/L** | **History Back / Forward** |
| `j` / `k` or `↑` / `↓` | Scroll up / down (vim keys & touch drag supported) |
| `PgUp` / `PgDn` / `Space` | Half-page scroll |
| `+` / `-` / `0` | Zoom In / Zoom Out / Reset Zoom (CDP modes) |
| `r` | Reload page / retry connection |
| `q` / `Esc` | Exit Termblitz |

---

## 📦 Building & Running in Termux

### Prerequisites
Inside Termux / Debian proot:
```bash
pkg install -y proot-distro
proot-distro login debian
apt update && apt install -y build-essential cargo chromium curl
```

### Build Release Binary
```bash
cargo build --release
```
The optimized binary is compiled to `target/release/termblitz`.

### Run
```bash
./target/release/termblitz https://www.google.com
```

---

## 📄 License
MIT License.
