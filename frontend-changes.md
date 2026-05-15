# Frontend Changes — Dark/Light Theme Toggle

## Summary

Added a dark/light theme toggle button that allows users to switch between the existing dark theme and a new light theme. The preference persists across page reloads via `localStorage`.

---

## Files Modified

### `frontend/index.html`

- Bumped cache-busting version from `v=11` to `v=12` on both `style.css` and `script.js`.
- Added a `<button id="themeToggle">` element with two SVG icons:
  - **Sun icon** — displayed in dark mode; clicking switches to light mode.
  - **Moon icon** — displayed in light mode; clicking switches back to dark mode.
- The button is positioned as a fixed overlay (top-right corner) using CSS so it is always visible.

### `frontend/style.css`

**New CSS variables (`:root` additions):**

Added variables for source-link colors and code block backgrounds so they can be overridden per theme without duplicating selectors:
- `--source-link-color`, `--source-link-bg`, `--source-link-border`
- `--source-link-hover-bg`, `--source-link-hover-border`, `--source-link-hover-color`
- `--code-bg`

**New `html[data-theme="light"]` block:**

Overrides all background, surface, text, and border variables for light mode:
| Variable | Light value |
|---|---|
| `--background` | `#f8fafc` |
| `--surface` | `#ffffff` |
| `--surface-hover` | `#f1f5f9` |
| `--text-primary` | `#0f172a` |
| `--text-secondary` | `#64748b` |
| `--border-color` | `#e2e8f0` |
| `--shadow` | lighter `rgba(0,0,0,0.1)` |
| `--welcome-bg` | `#dbeafe` |
| Source link colors | dark blue (`#1d4ed8`) for contrast on white |
| `--code-bg` | `rgba(0,0,0,0.05)` |

Primary/accent colors (`--primary-color`, `--primary-hover`, `--focus-ring`) remain unchanged in both themes.

**Smooth transition rule:**

Added `transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease` to `body`, `.sidebar`, `.chat-main`, `.chat-container`, `.chat-messages`, `.chat-input-container`, `.message-content`, `.stat-item`, `.theme-toggle`, and `#chatInput` so all backgrounds and text fade smoothly when the theme changes.

**Converted hardcoded colors to CSS variables:**

- `.sources-content a` — replaced `#93c5fd` and hardcoded `rgba(37,99,235,...)` values with `var(--source-link-*)` variables.
- `.message-content code` and `.message-content pre` — replaced `rgba(0,0,0,0.2)` with `var(--code-bg)`.

**Theme toggle button styles (`.theme-toggle`):**

- Fixed position: `top: 1rem; right: 1rem; z-index: 1000`.
- 42×42 px circle with `var(--surface)` background and `var(--border-color)` border.
- Hover: scales up 1.1× and shifts color to `var(--text-primary)`.
- Focus: `box-shadow` focus ring using `var(--focus-ring)`.
- Icon switching: `.sun-icon` is shown by default (dark mode); `html[data-theme="light"]` hides sun and shows `.moon-icon`.
- `@keyframes spin-once` + `.theme-toggle.spinning` class: rotates the button 360° on each click for tactile feedback.

### `frontend/script.js`

**Immediate IIFE at top of file:**

```js
(function () {
    if (localStorage.getItem('theme') === 'light') {
        document.documentElement.setAttribute('data-theme', 'light');
    }
})();
```

Runs before `DOMContentLoaded` to set the saved theme before the first paint, preventing a flash of wrong theme.

**`toggleTheme()` function:**

- Reads `data-theme` attribute on `document.documentElement` (`<html>`) to determine the current theme.
- Toggles the attribute and writes the new value to `localStorage` (`'light'` or `'dark'`).
- Restarts the `spinning` CSS animation on the button via a forced reflow (`void btn.offsetWidth`) and removes the class after `animationend`.

**`setupEventListeners()` update:**

Added `document.getElementById('themeToggle').addEventListener('click', toggleTheme)` alongside the existing event listeners.

---

## Behaviour

| State | Icon shown | `data-theme` on `<html>` | `localStorage.theme` |
|---|---|---|---|
| Dark (default) | ☀ Sun | _(absent)_ | `'dark'` or absent |
| Light | 🌙 Moon | `"light"` | `'light'` |

- Clicking the toggle switches theme instantly with a 0.3 s CSS transition on all themed surfaces.
- The button spins 360° on each click as visual feedback.
- Preference survives page reload via `localStorage`.
- All existing UI elements (sidebar, chat bubbles, inputs, suggested questions, source links, code blocks, welcome message) adapt to both themes through the CSS variable system.
- Keyboard accessible: the button is a native `<button>` element with `aria-label="Toggle theme"` and a visible focus ring.
