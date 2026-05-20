# Contributing to Project Rolo

Thank you for helping Rolo grow. Every contribution is appreciated and makes Rolo healthier. 

This is a small and personal project. Lara Washington reviews and merges every PR personally. Apologies if it takes a while to get back to you.

---

## What's welcome

- 🐛 **Bug reports** — open an issue with what you did, what you expected, and what Rolo did instead. A short screen recording will be helpful.
- 🎨 **Sprite art** — new animations, alt palettes, accessories for Rolo, accessibility-friendly variants. Keep frames 64×64 unless the existing animation says otherwise; include an `animation_meta.json` so it slots in.
- ✨ **Features** — Always open to new ideas. Open an issue or get in touch if you want to discuss it first. 
- 🛠️ **Refactors** — yes, but please scope them. One concern per PR.
- 📝 **Docs improvements** — always welcome.

## What's not a fit

- Turning Rolo into an AI assistant, coding helper, or productivity tool. He has politely declined this future.
- Cloud integrations that move user data off the local machine. Rolo lives locally on purpose.

---

## Setting up your dev environment

See [Build from source](README.md#build-from-source) in the README.

After cloning, confirm both test suites are green before you start:

```bash
cargo test --manifest-path src-tauri/Cargo.toml
pnpm test
```

---

## Working on a change

1. **Branch.** Create a branch named after the change: `fix-speech-bubble-offset`, `feat-evening-mood`, etc. (No need to prefix with your username.)
2. **Small commits.** Each commit should leave the tree compiling and the tests passing. Easier to review, easier to revert.
3. **Tests.** Every behavior change needs a test. Rust changes go in `src-tauri/tests/` or the closest `#[cfg(test)]` module; frontend changes go in `src/**/*.test.ts` / `*.test.tsx`.
4. **Format & lint** before pushing:
   ```bash
   cargo fmt --manifest-path src-tauri/Cargo.toml
   cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets -- -D warnings
   pnpm lint  # (if configured)
   ```
5. **Open a PR** against `main`. Reference any related issue with `Fixes #N` so it auto-closes on merge.

## Commit message conventions

A light-touch conventional style:

```
feat(speech): add evening idle line for sleepy users
fix(tick): clamp drag delta when mouse leaves screen
docs(readme): correct ollama pull command
chore: bump tauri to 2.1.0
```

Keep the first line under 72 characters. The body (if any) is plain English — explain *why*, not *what* (the diff shows *what*).

## Authorship policy

Project Rolo is human-authored. **Do not add `Co-Authored-By: <AI assistant>` or "Generated with …" trailers to commits**, even if your editor tries to. AI tools may assist you with editing — that's fine — but they are not credited as co-authors for licensing or attribution.

## Review timeline

This is a personal project. PRs typically get a first response within a week. Big PRs may take longer. If yours has gone quiet for more than two weeks, comment on the thread — a nudge is appreciated. 

---

## Filing bug reports

Open an [issue](https://github.com/larawashington/Project-Rolo/issues/new) with:

- **OS + version** (e.g. macOS 15.4, Apple Silicon).
- **Rolo version** — from the app menu or the release tag you downloaded.
- **Steps to reproduce** — the more specific the better.
- **What you expected** vs. **what happened**.
- **Logs**, if any — the Rolo console window has a "Copy logs" button.
- **Screen recording** for visual bugs (drag, animation, drawing artifacts).

For security issues, see [SECURITY.md](SECURITY.md) — please don't open a public issue.

---

## Code of conduct

Be kind to other contributors and to Rolo. No harassment, no slurs, no dragging people in private. Rolo is for everyone who likes a little whimsy on their screen; the contributor space should match.

The maintainer reserves the right to lock threads, decline PRs, and ban accounts that make this project a worse place to spend time.
