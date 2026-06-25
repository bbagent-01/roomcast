# bbase-landing-template

A minimal, deploy-ready landing page template for Cloudflare Pages.  
Clone → edit → push → live at `yourproject.bbase.dev`.

## Stack
- Static HTML + Tailwind CSS (via CDN)
- Plus Jakarta Sans + JetBrains Mono
- BB Base Theme (dark hero / light body)
- Zero build step — Cloudflare serves as-is

## Quick Start (VS Code)

1. **Clone from GitHub** — Use VS Code's Command Palette (`Cmd+Shift+P`) → `Git: Clone` → paste the repo URL
2. **Open the project** — VS Code opens automatically
3. **Edit `index.html`** — Change the content to match your project
4. **Push to deploy** — Click the Source Control icon (branch) in the sidebar → Stage → Commit → Push
5. **Live in ~60 seconds** at `yourproject.bbase.dev`

## File Structure
```
├── index.html          ← Your landing page
├── /assets
│   ├── style.css       ← Custom styles (Tailwind handles most)
│   └── /img            ← Images go here
├── /pages              ← Additional pages (about, pricing, etc.)
└── README.md
```

## Adding Pages
Create new `.html` files in `/pages`. They'll be available at `yourproject.bbase.dev/pages/filename`.

## Custom Domain
After first deploy, go to Cloudflare Pages → your project → Custom domains → Add `yourproject.bbase.dev`.
