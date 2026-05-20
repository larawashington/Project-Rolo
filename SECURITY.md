# Security Policy

Thank you for taking the time to look at Rolo's security.

## Supported versions

Project Rolo is in alpha. Only the **latest tagged release** receives security fixes. Older `.dmg` builds will not be patched — please upgrade.

| Version | Supported |
|---|---|
| Latest release on `main` | ✅ |
| Older releases | ❌ |

## What's in scope

The following parts of the project are in scope for security reports:

- The desktop application (the contents of this repo and the published `.dmg`).
- The local data Rolo writes to disk (vault files, settings, chat history).
- The bundled provider clients (Ollama, OpenAI-compatible, Anthropic, Gemini, OpenAI, HuggingFace, DeepInfra) — specifically, how Rolo *uses* them. We're not responsible for the providers themselves.
- The fine-tuned model published at `hf.co/larawashington/rolo-brain` insofar as the model weights or Modelfile encode something exploitable.

## What's out of scope

- Issues that require physical access to an already-unlocked machine.
- Social engineering of the maintainer.
- Vulnerabilities in third-party services Rolo connects to (Ollama, HuggingFace, the LLM providers). Report those upstream.
- DoS against the local app.

## How to report

**Please do not open a public GitHub issue for security problems.**

Two private channels, pick whichever is easier:

1. **GitHub private vulnerability report** (preferred): visit the [Security tab](https://github.com/larawashington/Project-Rolo/security/advisories/new) and submit a private advisory. This creates a private thread between you and the maintainer.
2. **Email**: send to the address listed on the [maintainer's GitHub profile](https://github.com/larawashington). Subject prefix: `[Rolo security]`.

Please include:

- A description of the vulnerability and its impact.
- Steps to reproduce, or a proof-of-concept.
- Your assessment of severity and any suggested remediation.
- Whether you'd like to be credited in the advisory when it's published.


