---
license: gemma
base_model: google/gemma-3-4b-it
tags:
  - project-rolo
  - rolo-brain
  - desktop-pet
  - desktop-companion
  - gemma
  - gemma-3
  - lora
  - gguf
  - ollama
library_name: gguf
pipeline_tag: text-generation
---

# rolo-brain

A fine-tune of **Gemma 3 4B Instruction-Tuned** for
[Project Rolo](https://github.com/larawashington/Desktop-Pet), a free,
open-source desktop companion to brighten your day. This is the model 
specifically trained to give Rolo his unique voice and personality.

The file to run is `rolo-brain-Q5_K_M.gguf` (≈3 GB) via Ollama.

```
ollama pull hf.co/larawashington/rolo-brain
```

## What this is

`rolo-brain` is a LoRA fine-tune fused into Gemma 3 4B IT (4-bit MLX
base) and converted to GGUF Q5_K_M with an imatrix computed from the
training corpus's calibration set. The whole stack runs locally; no
data leaves your machine at inference time.

The model is purpose-built for [Project Rolo](https://github.com/larawashington/Desktop-Pet) — 
it ships with a **specific first-user-turn contract** (see 
[Inference contract](#inference-contract) below).

## Quickstart

After pulling, you can talk to him from the command line:

```
ollama run hf.co/larawashington/rolo-brain
```

For application use the chat API at `http://localhost:11434/api/chat`,
sending the first-user-turn body documented in `runtime_contract.md`.

A minimal example (chat channel):

```json
{
  "model": "hf.co/larawashington/rolo-brain",
  "messages": [
    {
      "role": "user",
      "content": "[Mood: content | Energy: medium | Social: ok | Time: 8:30 AM]\n\nmorning rolo"
    }
  ]
}
```

A minimal example (proactive speech channel, no user message):

```json
{
  "model": "hf.co/larawashington/rolo-brain",
  "messages": [
    {
      "role": "user",
      "content": "[Mood: content | Energy: medium | Social: ok | Time: 2:30 PM | Hunger: just ate]\n[State: Rolo is sitting idle]\n\n<idle>"
    }
  ]
}
```

## Inference contract

The model was trained on a strict prompt layout. The deployed
inference path must produce byte-identical first-user-turns to the
training distribution or output quality degrades silently.

In short:

- **No `system` role.** Persona is in the weights.
- **One `user` turn.** Inside it: a state header (one or two bracketed
  lines), then a single blank line, then either the user's message
  (chat channel) or the literal sentinel `<idle>` (proactive speech).
- **No emoji.** The model was trained emoji-free.
- **Replies are short.** Most turns are 1–3 sentences. The training
  corpus capped responses at 30 words.

The full contract — including the state-vocabulary table, the 5
speech-eligible pet-state stems, and the idle sentinel — lives in
`runtime_contract.md` in this repository.

## Sampling

The bundled `Modelfile` sets sampling defaults that match the
parameters used to validate the fine-tune:

| Parameter | Value |
|-----------|-------|
| `temperature` | 0.85 |
| `top_p` | 0.95 |
| `num_predict` | 64 |
| `stop` | `<end_of_turn>`, `\n\n`, list-bullet prefixes |

The stop sequences trade a small amount of recall for tight bubbles —
the model is trained to end naturally, and the extra stops kill list
rambles that occasionally leak through.

## Persona summary

Rolo is a desktop companion who lives on your screen and keeps you 
company through everyday tasks. He is:

- Affectionate and encouraging, providing a positive voice and energy. 
- Snarky when you make suboptimal choices, to softly reprimand you.
- Primarily concerned with your wellbeing. But also food-motivated.
- Self-aware about his own appetite, naps, and need for attention.
- Capped at short, punchy replies (1–3 sentences typically).

Rolo is **NOT** an assistant or tool. He will decline any requests 
that contradict his character and purpose of bringing joy and whimsy.

## Training summary

| Stage | What happened |
|-------|---------------|
| 1. Gold seed | ~280 hand-written rows in Rolo's voice by Lara Washington |
| 2. Distillation | DeepSeek V4-Pro generated ~5,000 candidate rows from the gold seed + 50 personas |
| 3. Filter + judge | Kimi K2.6 graded surviving rows against a 4-axis rubric (`in_character`, `length`, `reaction_sanity`, `no_banned_phrases`) |
| 4. Format + mix | ~7% Dolly-15k mix-in for general instruction-following; held-out test set (~280 rows) split BEFORE format pass |
| 5. SFT | LoRA r=16 scale=2.0 on `mlx-community/gemma-3-4b-it-4bit`, completion-only loss, seed 20260515 |
| 6. Fuse → GGUF | Fused into the 4-bit base, dequantized to fp16, converted via `llama.cpp/convert_hf_to_gguf.py` |
| 7. Quantize | Q5_K_M with an importance matrix built from in-domain calibration prompts |

The training-data provenance, license posture, and base-model
citation are in [NOTICE](./NOTICE).

## What this is not

- It is **not** a general assistant. Don't use it for coding,
  summarization, or factual lookups — the response-length cap and
  pet-persona conditioning will reject this.
- It is **not** the only way to talk to Rolo. The Desktop-Pet app
  ships with a fallback path that uses base `gemma3:4b` with an
  explicit system prompt; if you don't `ollama pull` this model the
  app degrades to that path automatically.
- It is **not** a chatbot for end users to integrate into other
  products. The persona is opinionated and the inference contract is
  narrow.

## License

The LoRA delta and surrounding scripts are released under the **Apache
License 2.0**.

The base model retains the **Gemma Terms of Use**
(https://ai.google.dev/gemma/terms). Downstream users of this fused
model are bound by those terms — see [NOTICE](./NOTICE) for full
attribution.

## Acknowledgements

- **Google** for releasing Gemma 3 under terms that permit
  derivative fine-tunes.
- **DeepSeek-AI** for V4-Pro and the open distillation license.
- **Moonshot AI** for Kimi K2.6 as a judge model.
- **Apple MLX** team for the training stack that makes M-series
  fine-tunes practical on consumer hardware.

— Authored and created by Lara Washington.
