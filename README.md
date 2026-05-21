# Project Rolo

> Meet Rolo. A free, open-source desktop companion here to make your day a little brighter.

[![Status: alpha](https://img.shields.io/badge/status-alpha-orange)](#status)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Built with Tauri](https://img.shields.io/badge/built%20with-Tauri%202-24C8DB)](https://tauri.app/)

Rolo lives on your screen and keeps you company throughout everyday tasks. He runs locally on your machine, respects your privacy, and is happiest when you're happy.

> **Rolo is not an AI assistant or tool.** He will politely decline any requests that fall outside his purpose: bringing joy and whimsy to your screen.

---

## How Rolo works

Rolo lives locally on your machine and respects your privacy. He's powered by three pillars:

- **Rolo's voice** — a fine-tuned [Gemma 3 4B](https://deepmind.google/models/gemma/gemma-3/) model. A small open-source language model trained on hand-seeded examples of how Rolo speaks.
- **Rolo's status** — his mood is driven by four bars (hunger, social, energy, happiness) combined with his perception of what's happening on your screen. The result gives flavor and relevance to everything he says.
- **Rolo's memory** — built through ongoing interactions. He remembers the names, events, and preferences you share and stores them while he sleeps and dreams.

---

## Install Rolo (macOS)

Grab the latest build — no compiler required:

**[⬇ Download Project-Rolo.dmg](https://github.com/larawashington/Project-Rolo/releases/latest/download/Project-Rolo.dmg)**

1. Open the `.dmg` and drag Rolo into Applications.
2. Install [Ollama](https://ollama.com/) if you don't already have it.
3. Pull Rolo's brain (the fine-tuned model that gives him his voice, ~3 GB):

   ```bash
   ollama pull hf.co/larawashington/rolo-brain
   ```

4. Launch Rolo from Applications.

> **First launch on macOS:** the alpha build isn't notarized yet, so Gatekeeper will say *"Apple could not verify Project-Rolo is free of malware."* Right-click Rolo in Applications and choose **Open** the first time — macOS will remember the exception. Notarization is on the roadmap.

> **Fallback:** if `rolo-brain` isn't on your machine, Rolo automatically falls back to base `gemma3:4b` with a system prompt. The fallback works but Rolo sounds less like himself — `rolo-brain` is the recommended path.

---

## Build from source

For contributors and the curious.

**Prereqs:** [Rust](https://rustup.rs/), [Node 20+](https://nodejs.org/), [pnpm](https://pnpm.io/), [Ollama](https://ollama.com/).

```bash
git clone https://github.com/larawashington/Project-Rolo.git
cd Project-Rolo
pnpm install
ollama pull hf.co/larawashington/rolo-brain
pnpm tauri dev
```

To produce an app bundle (output in `src-tauri/target/release/bundle/`):

```bash
pnpm tauri build
```

---

## What's inside

- `src/` — Rolo's window, speech bubbles, chat, and command center (React + TypeScript)
- `src-tauri/src/` — Rolo's brain stem (Rust): state machine, animation tick loop, LLM clients, vault, tools, perception
- `ASSETS/` — every hand-made 64×64 sprite frame and the `animation_meta.json` that drives it
- `dist/hf-rolo-brain/` — the public model card and Modelfile for Rolo's fine-tuned voice

## Tech stack

- **Frontend** — React 19 + TypeScript, Vite
- **Backend** — Rust (Tauri 2.x), Tokio
- **Local model runtime** — [Ollama](https://ollama.com/) (default: `gemma3:4b`)
- **Memory store** — local SQLite + embeddings (everything stays on your machine)
- **Sprites** — hand-drawn 64×64 pixel art

---

## Status

**Alpha — building in public.** macOS only for now. Expect a few rough edges, the occasional crash, and Rolo wandering into corners. The fine-tuned voice is live; everything else is functional and getting steadily friendlier.

## Contributing

PRs, bug reports, and sprite-art suggestions welcome. A `CONTRIBUTING.md` with the full contribution workflow is on the way.

## License

MIT — see [LICENSE](LICENSE).

Rolo's brain artifact (the fine-tuned model on HuggingFace) is governed by the [NOTICE](dist/hf-rolo-brain/NOTICE) in `dist/hf-rolo-brain/`, which preserves attribution to the upstream Gemma weights.

---

_Rolo's website lives at [rolo.dev](https://rolo.dev). This repo is his code._
