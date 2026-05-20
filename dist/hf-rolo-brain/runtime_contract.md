# rolo-brain — runtime prompt contract

This document specifies the exact first-user-turn layout the model was
trained on. If you are wiring `rolo-brain` into your own runtime, the
assembled prompt should match this grammar; deviations move inference
off the training distribution and reduce output quality.

You don't need this file if you're using `ollama run` interactively —
the REPL sends each line as a plain user message, and Gemma's
instruction-following keeps the conversation coherent. But the
state-conditioned voice the fine-tune adds (mood, energy, animation
state) only activates when the prompt includes the bracketed state
header described below. This contract is for developers calling
`/api/chat` from code who want the full conditioned behavior.

---

## 1. The contract in one sentence

There is no system role. Each inference call sends exactly one `user`
message containing a bracketed state header, a blank line, and either
the user's message (chat) or the literal sentinel `<idle>` (proactive
speech).

---

## 2. First-user-turn grammar

After Ollama applies the chat template, the prompt fed to the model is:

```
<start_of_turn>user
<state_block>
<pet_state_block_if_present>

<user_message_or_idle_sentinel><end_of_turn>
<start_of_turn>model
```

The portion you assemble and put in `messages[0].content` is:

```
<state_block>
<pet_state_block_if_present>

<user_message_or_idle_sentinel>
```

Rules:

- `<state_block>` is required and follows the grammar in §3.
- `<pet_state_block_if_present>` is omitted on the chat channel and
  required on the speech channel. On chat, the state block is followed
  immediately by the blank-line separator.
- Exactly one blank line (two `\n` in a row) separates the header(s)
  from the message body, on both channels.
- `<user_message_or_idle_sentinel>` is the user's typed message
  verbatim on chat, or the literal string `<idle>` on speech (see §5).

An optional `[Observation: …]` line may appear between the pet-state
block and the blank-line separator on the speech channel.

---

## 3. State header vocabulary

The state block has the form:

```
[Mood: <mood> | Energy: <energy> | Social: <social> | Time: <time>]
```

with an optional trailing `| Hunger: <hunger>` field. Field order is
fixed. The separator is space-pipe-space.

| Field    | Accepted values |
|----------|-----------------|
| `Mood`   | `happy`, `content`, `neutral`, `low`, `sad`. May carry a parenthetical sass modifier: `sad (snarky)`, `low (very snarky)`. |
| `Energy` | `high`, `medium`, `low` |
| `Social` | `engaged`, `ok`, `lonely` |
| `Time`   | 12-hour clock, e.g. `8:30 AM`, `2:00 PM`, `1:30 AM` |
| `Hunger` | `just ate` or `peckish`. Omit the field entirely at neutral hunger. |

Example state blocks:

```
[Mood: content | Energy: high | Social: ok | Time: 9:00 AM]
[Mood: sad (snarky) | Energy: low | Social: lonely | Time: 2:00 PM | Hunger: peckish]
```

Values outside this vocabulary won't raise an error, but the model
hasn't seen them and output quality will drop.

---

## 4. Pet-state block (speech channel only)

Proactive-speech calls include a second bracketed line documenting
Rolo's current animation state:

```
[State: <stem>]
```

The five accepted stems are:

| Animation state | Stem |
|-----------------|------|
| Idle            | `Rolo is sitting idle` |
| Walking         | `Rolo is walking around` |
| Sniffing food   | `Rolo is sniffing a file` |
| Just ate        | `Rolo just finished eating and is satisfied` |
| Food declined   | `Rolo was offered food but it was declined` |

The training corpus contains only these five stems on the speech
channel. Other animation states (sleeping, dragging, mid-eating) are
suppressed upstream and don't reach inference.

---

## 5. The idle sentinel

On the speech channel, the body of the user turn is the literal string:

```
<idle>
```

No quotes, no whitespace padding, no alternative spellings. This is
the token sequence the model was trained to interpret as "no user
message — say something appropriate to the current state."

---

## 6. Output expectations

- **Length.** Replies are 1–3 sentences. Training data was capped at
  30 words per response; sampling uses `num_predict=64`, which leaves
  headroom for stop-token cleanup.
- **No emoji.** The training corpus was emoji-free.
- **No lists.** The bundled Modelfile stops on bullet/list markers
  (`\n* `, `\n- `, `\n1. `, `\n2. `) as a safeguard. The model rarely
  emits them.
- **No follow-up question loop.** Rolo is a desktop pet, not an
  assistant — he reacts, comments, and stops.

Long, list-shaped, or assistant-like output usually indicates a prompt
that violates §2 or §3 — most often, a system message that shouldn't
be there.

---

## 7. Worked examples

### Chat channel — typed user message

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

Note the `\n\n` between the state header and `morning rolo` — one
blank line, two newlines.

Expected output: a short greeting in Rolo's voice, 1–2 sentences.

### Speech channel — proactive `<idle>`

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

Note: two bracketed lines separated by a single `\n`, then `\n\n`,
then the literal `<idle>`.

Expected output: a short, in-character remark appropriate to the mood
and state.

### Speech channel — with perception observation

```
[Mood: low | Energy: low | Social: lonely | Time: 11:42 PM]
[State: Rolo is sitting idle]
[Observation: cursor has not moved in 12 minutes]

<idle>
```

Observations are short free-form clauses. They appeared in a minority
of training rows, and the model handles their absence without issue.

---

## 8. Common mistakes

| Mistake | Symptom |
|---------|---------|
| Adding a `system` role | Generic, list-prone, assistant-shaped replies |
| Hardcoding `<IDLE>`, `[idle]`, `idle`, etc. | Model treats it as user content; replies go off-tone |
| Two blank lines between header and body | Output quality drop from a tokenization shift at the boundary |
| Missing the `[State: …]` line on speech | Replies lose grounding in Rolo's current animation |
| Using a mood word outside §3 | Output looks fluent but is off-distribution |
| Sending emoji in the user message | Model may echo them and degrade voice |

If output looks wrong, log the exact contents of `messages[0].content`
and compare against the examples in §7.
